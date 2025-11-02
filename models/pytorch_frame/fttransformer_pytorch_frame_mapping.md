## FT-Transformer 與 PyTorch-Frame 五個 stage 對照說明

以下說明基於 `models/pytorch_frame/fttransformer.py` 的實作與 FT-Transformer 論文（Revisiting），整理 FT-Transformer 原始 pipeline 如何處理輸入資料，並把它對應到 PyTorch Frame 的 stage（start、materialize、encoding、columnwise、decoding）。

### 一、簡短摘要
- FT-Transformer 在此實作中遵循 PyTorch-Frame 的分層抽象：Materialization 將原始表格轉成 `TensorFrame` 與 `DataLoader`；Encoding 使用 `StypeWiseFeatureEncoder` 將每欄位投影到共享維度；Column-wise interaction 為 `FTTransformerConvs`（backbone）；Decoding 為一個簡單 MLP head（LayerNorm → ReLU → Linear）。
- 檔案同時實作了五個可選 GNN 插入點（start、materialize、encoding、columnwise、decoding），其中 start/materialize/decoding 屬 row-level graph，encoding/columnwise 屬 columns-as-nodes graph（per-row）。

### 二、FT-Transformer 原始 pipeline（如何處理 input data）
1. start (dummy)
   - 函式：`start_fn(train_df, val_df, test_df)`（目前為 passthrough）。
   - 用途：提供在最前面插入 GNN 的 hook（`gnn_after_start_fn`）。

2. materialize
   - 函式：`materialize_fn(train_df, val_df, test_df, dataset_results, config)`。
   - 行為：建立 `Yandex` dataset、呼叫 `dataset.materialize()` 產生 `TensorFrame`；建立 `DataLoader`；定義 `stype_encoder_dict`（categorical/numerical encoder）；設定 metric 與輸出通道數。
   - 輸出：`material_outputs`（包含 `train_tensor_frame`、`train_loader`、`col_stats`、`stype_encoder_dict` 等）。

3. encoding
   - 實作：`encoder = StypeWiseFeatureEncoder(...)`。
   - 呼叫位置：`x, _ = encoder(tf)`（在 `model_forward`）。
   - 輸出 shape：`[batch, num_cols, channels]`。

4. column-wise interaction
   - 實作：`backbone = FTTransformerConvs(channels, num_layers)`。
   - 呼叫位置：`x, x_cls = backbone(x)`，其中 `x` 為欄位級表示，`x_cls` 為聚合後的 row-level 表示（供 decoder 使用）。

5. decoding
   - 實作：`decoder = Sequential(LayerNorm, ReLU, Linear)`。
   - 呼叫位置：`out = decoder(x_cls)`。
   - 功能：把 backbone 的 row-level表示輸出為 logits / 預測。

### 三、五個 stage 與程式對應（簡明表）

| PyTorch-Frame stage | FT-Transformer 中的對應位置 (檔案/函式/物件) | 資料型態 / 說明 |
|---|---:|---|
| start | `start_fn(...)` + `gnn_after_start_fn(...)` | 原始 pandas DataFrame；`gnn_after_start_fn` 對 concat 的 all_df 建 KNN graph，訓練 row-level GCN（重構式 MSE），輸出 row embeddings 回寫為新 numeric columns（離線）。 |
| materialize | `materialize_fn(...)` | 產生 `TensorFrame` 與 `DataLoader`；同時產生 `stype_encoder_dict`。 |
| encoding | `StypeWiseFeatureEncoder`；呼叫：`x, _ = encoder(tf)` | 輸出 `x` shape `[batch, num_cols, channels]`。 |
| columnwise | `FTTransformerConvs` (backbone)，呼叫：`x, x_cls = backbone(x)` | `x` 為欄位表示（供後續處理），`x_cls` 為 row-level 聚合表示（decoder 輸入）。 |
| decoding | `decoder = Sequential(...)`；`out = decoder(x_cls)` | 產生 row-level logits / predictions。 |

### 四、GNN 插入點行為（程式對應與語意）
- start: `gnn_after_start_fn` 把三個 df concat，對 row 建 KNN 圖並用 `SimpleGCN` 訓練，loss 為 MSE(out, x)，最後回寫 embedding 到 pandas df，供 `materialize_fn` 重新物化。
- materialize: `gnn_after_materialize_fn` 把 `TensorFrame` 轉回 df（取 numerical features），同樣做 row-level KNN 與 GCN 訓練，回寫並重新物化成 TensorFrame 與 loaders。
- encoding: `if gnn_stage == 'encoding'` 在 `model_forward`，在 `encoder(tf)` 後把 `x` reshape 為 `(batch * num_cols, channels)`，構造 per-row complete graph 的 `edge_index`（每個 row 的 columns 作為節點），並呼叫 `gnn(x_reshape, edge_index)`；該 GNN 與其他參數 joint-train（包含在 optimizer 內）。
- columnwise: `if gnn_stage == 'columnwise'` 已在程式中預留（目前為 noop / x = x），若啟用會在 backbone 後對 columns-as-nodes 做 GNN，與 backbone/encoder joint-train。
- decoding: `gnn_decoding_eval` 會把 `encoder+backbone` 的 pooled embeddings（row-level）收集成 all_emb，對 rows 建 KNN graph 並用 supervised loss（依 task 選擇 BCE / CE / MSE）訓練 GNN（early stop based on val），把 GNN 當作 post-hoc decoder。

### 五、訓練語意比較與注意事項

- 風險與建議：

---

## FT-Transformer — 模型 pipeline（上游 → 下游）範例

範例設定：醫療保險資料，1000 筆、30 欄，train/val/test = 700/150/150。

1) Start
   - 讀入 DataFrame，預留 `gnn_after_start_fn` 以便做離線 row-level GNN。

2) Materialize
   - 物化成 `TensorFrame`，建立 `stype_encoder_dict`、DataLoader。

3) Encoding
   - 用 `StypeWiseFeatureEncoder` 把每欄位投影為向量，輸出 `[B, num_cols, channels]`。

4) Column-wise / Backbone
   - `FTTransformerConvs` 進行欄位交互 (Transformer-like)，聚合為 row-level 表示供 decoder 使用。

5) Decoding
   - MLP head 輸出 logits；或收集 embeddings 以 decoding-stage GNN 做後處理。

PyTorch-Frame 對應：start / materialize / encoding / columnwise / decoding。
如果你想，我可以：
- 幫你把 `gnn_after_*` 改為只在 train nodes 上訓練；
- 或把 encoding/columnwise 的 dense edges 改成 top-k 範例；
- 或生成統一的實驗 runner 來系統性比較插入點效果。

檔案位置：`models/pytorch_frame/fttransformer_pytorch_frame_mapping.md`
