## ResNet 與 PyTorch-Frame 五個 stage 對照說明

本檔案根據 `models/pytorch_frame/resnet.py` 的實作與 ResNet 論文（Revisiting）整理 ResNet 的 pipeline 與 PyTorch-Frame 五個 stage（start、materialize、encoding、columnwise、decoding）之對應，以及 GNN 插入點的語意與實務注意事項。

### 一、精要
- ResNet 在 PyTorch-Frame 實作中包含：Materialization（TensorFrame 與 loader）、Encoding（`StypeWiseFeatureEncoder`）、Backbone（多層 fully-connected residual blocks，作用於 flatten 後的 row-level 向量）、Decoder（simple MLP head）。
- 程式支援在五個位置插入 GNN：`start`、`materialize`、`encoding`、`columnwise`、`decoding`。其中 `start/materialize/decoding` 為 row-level graph 操作；`encoding/columnwise`（程式有 hook）則可被用來把每個 row 的 columns 當作節點來做 column-level GNN（但 ResNet 的 backbone 是對 flatten 向量操作，因此 column-level GNN 的語意與效果需謹慎評估）。

### 二、ResNet 原始 pipeline（如何處理 input data）
1. start (dummy)
   - 呼叫：`start_fn(train_df, val_df, test_df)`（預設直接回傳輸入）。
   - 用途：提供在最前端插入 precompute GNN 的 hook（`gnn_after_start_fn`）。

2. materialize
   - 呼叫：`materialize_fn(train_df, val_df, test_df, dataset_results, config)`。
   - 行為：呼叫 `Yandex(...).materialize()` 來生成 `TensorFrame`，然後建立 `DataLoader`，設定數值 encoder 類型與 `stype_encoder_dict`，以及度量器（AUROC/Accuracy/MSE）。
   - 輸出：`train_tensor_frame/val_tensor_frame/test_tensor_frame` 與 `train_loader/val_loader/test_loader` 等。

3. encoding
   - 實作：`encoder = StypeWiseFeatureEncoder(...)`，在前向呼叫 `x, _ = encoder(tf)`。
   - 輸出：`x` 形狀為 `[batch, num_cols, channels]`。

4. backbone (ResNet 的 columnwise/interaction 等價)
   - 實作：先將 `x` flatten 為 `[batch, num_cols * channels]`（`x.view(x.size(0), math.prod(x.shape[1:]))`），然後餵入 `backbone = Sequential(*[FCResidualBlock(...) ...])`（多層全連接 residual blocks）。
   - 說明：不同於 Transformer-style 的 column-wise message passing，ResNet 將所有欄位展平後以 MLP/residual-blocks 處理，這等價於在展平的向量空間做交互。

5. decoding
   - 實作：`decoder = Sequential(LayerNorm, ReLU, Linear)`，在前向回傳 `out = decoder(x)`（x 為 backbone 的輸出）。

### 三、五個 stage 與程式對應（簡明表）

| PyTorch-Frame stage | ResNet 中的對應位置 (檔案/函式/物件) | 資料型態 / 說明 |
|---|---:|---|
| start | `start_fn(...)` + `gnn_after_start_fn(...)` | 原始 pandas DataFrame；`gnn_after_start_fn` 對 concat 的 all_df 建 KNN graph，訓練 row-level GCN（重構式 MSE），回寫 row-level embedding 至 df（離線 precompute）。 |
| materialize | `materialize_fn(...)` | 產生 `TensorFrame` 與 `DataLoader`，並建立 `stype_encoder_dict`。 |
| encoding | `StypeWiseFeatureEncoder`；呼叫：`x, _ = encoder(tf)` | 輸出 `x` shape `[batch, num_cols, channels]`。 |
| columnwise/backbone | `backbone = Sequential(*FCResidualBlock(...))`；在 forward 先 `x.view(... )` 再送 backbone | ResNet 在 flatten 後對 row-level 向量做 residual MLP 處理；此處相當於 model 的主要交互 body。 |
| decoding | `decoder = Sequential(...)`；`out = decoder(x)` | 產生 logits / predictions。 |

### 四、GNN 插入點行為（程式對應與語意）
- start: `gnn_after_start_fn` 把 train/val/test concat，對 rows 建 KNN 圖並用 `SimpleGCN` 訓練（loss = MSE），最後把 `final_emb` 分回 train/val/test 並回寫為新的 numeric columns，再交由 `materialize_fn` 處理。
- materialize: `gnn_after_materialize_fn` 把 `TensorFrame` 轉成 df（取 numerical features），再對 rows 建 KNN 與 GCN 訓練，回寫並重新物化供模型使用。
- encoding: 程式在 `model_forward` 有 `if gnn_stage == 'encoding'` 的 hook（會在 encoder 之後對 `x` 做 reshape 與 GNN）；語意為把每個 row 的 columns 當作 nodes（node = each column embedding）做 column-level GNN，然後再 flatten 送入 backbone。這是 joint-train（GNN 參數包含於 optimizer，梯度由最終 loss 提供）。
- columnwise: ResNet 沒有傳統的 column-conv block（它在 flatten 後做 MLP），程式仍保留 `if gnn_stage == 'columnwise'` hook，這個 hook 在模型 flatten 後或 backbone 前後插入 GNN會比較不直觀；通常 columnwise GNN 在 ResNet context 意味著在 encoder 輸出（未 flatten）上做 column-level message passing，然後再 flatten 進入 backbone。
- decoding: `gnn_decoding_eval` 會使用 `get_all_embeddings_and_targets`（該函式會取 encoder -> flatten -> backbone 的輸出）收集 row-level embeddings，對 rows 建 KNN graph 並在該 graph 上以 supervised loss（BCE/CE/MSE）訓練 GNN，並以 validation metric 早停。此 GNN 是 post-hoc decoder（把 GNN 當成替代 decoder）。

### 五、實務注意事項與建議（摘要）
- Data leakage：start/materialize/decoding 若在訓練或建圖時包含 test nodes 或使用 test labels，容易造成資訊洩漏。建議在 GNN 訓練時只使用 train 範例（或採用 inductive inference）。
- column-level GNN 在 ResNet 的上下文需謹慎：因 ResNet 的 main body 是展平後的 MLP，若在 encoder 後立刻用 column-level GNN 處理，效果取決於 GNN 是否能提供比直接 flatten 更有利的組合表示；此外，計算複雜度會隨欄位數 O(num_cols^2) 增長。
- 評估一致性：比較不同插入點時，務必固定 seed、split、GNN 建圖參數（k）、GNN 架構與訓練資源，並匯報訓練成本（時間、epoch、early-stop epoch）。
- 建議改進：把 `gnn_after_*` 的訓練限制為僅使用 training 範例（提供 `train_only=True` flag）；對 encoding/columnwise 的 dense edges 改為 top-k sparse edges；在 start/materialize 的重構 loss 改為更 task-aligned的 self-supervised loss（contrastive、masked reconstruction），或直接用 semi-supervised supervised loss（若不會造成 leakage）。

---

如果你想，我可以幫你：
- 自動修改 `gnn_after_materialize_fn` 與 `gnn_after_start_fn`，使其僅用 train nodes 做 GNN 訓練；
- 在 `model_forward` 的 `encoding` hook 中，提供一個 top-k edges 的示例來替換完整的 dense complete graph；
- 或生成一個比較五個插入點效能與成本的實驗 runner。

檔案位置：`models/pytorch_frame/resnet_pytorch_frame_mapping.md`


## ResNet — 模型 pipeline（上游 → 下游）範例

範例設定：房價回歸，5000 筆、25 欄，train/val/test = 3500/750/750。

1) Start
   - 讀入 DataFrame；可選 precompute row-level GNN 在 `gnn_after_start_fn`。

2) Materialize
   - 轉為 `TensorFrame` 與 DataLoader；建立數值編碼與標準化。

3) Encoding
   - 用 stype-wise encoder 取得 `[B, num_cols, channels]` 表示，然後 flatten 為 row-level 向量。

4) Backbone (ResNet style)
   - 在 flatten 向量上套用多層 residual MLP（backbone），進行交互與表徵學習。

5) Decoding
   - MLP head 輸出回歸值；可用 decoding GNN 作後處理但需小心 leakage。

PyTorch-Frame 對應：start → materialize → encoding → columnwise/backbone → decoding。
