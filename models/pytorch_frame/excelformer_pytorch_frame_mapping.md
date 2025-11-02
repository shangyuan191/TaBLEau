## ExcelFormer 與 PyTorch-Frame 五個 stage 對照說明

以下說明基於 `models/pytorch_frame/excelformer.py` 的實作與 ExcelFormer 論文，整理 ExcelFormer 原始 pipeline如何處理輸入資料，並把它對應到 PyTorch Frame 的 stage（start、materialize、encoding、columnwise、decoding）。

### 一、簡短摘要
- ExcelFormer 的程式主要分為：資料物化（materialize）、欄位編碼（encoding）、列內交互（column-wise interaction）、以及讀出（decoding）。
- 你新增的 `start` 為 dummy stage（在 `start_fn`），程式中也支援把 GNN 插在五個位置：`start`、`materialize`、`encoding`、`columnwise`、`decoding`。

### 二、ExcelFormer 原始 pipeline（如何處理 input data）
1. start (dummy)
   - 函式：`start_fn(train_df, val_df, test_df)`（目前為 passthrough）
   - 輸入：pandas DataFrame（train/val/test）。

2. materialize
   - 函式：`materialize_fn(train_df, val_df, test_df, dataset_results, config)`。
   - 行為：建立 `Yandex` dataset wrapper、呼叫 `dataset.materialize()` 產生 `TensorFrame`，並執行 `CatToNumTransform` 與 `MutualInformationSort`，最後建立 `DataLoader`。
   - 輸出：`train_tensor_frame/val_tensor_frame/test_tensor_frame` 與 `train_loader/val_loader/test_loader` 等。

3. encoding
   - 實作：`encoder = StypeWiseFeatureEncoder(...)` 與 `ExcelFormerEncoder(...)`。
   - 呼叫位置：`x, _ = encoder(tf)`（在 `model_forward`）。
   - 輸出 shape 範例：`[batch, num_cols, channels]`（例如 (32, 20, 256)）。

4. column-wise interaction
   - 實作：`convs = ModuleList([ExcelFormerConv(...) for _ in range(num_layers)])`。
   - 呼叫位置：`for conv in convs: x = conv(x)`。
   - 功能：在欄位嵌入空間執行多層互動 (message passing / attention-like mixing)。

5. decoding
   - 實作：`decoder = ExcelFormerDecoder(...)`。
   - 呼叫位置：`out = decoder(x)`。
   - 功能：把欄位嵌入聚合成 row-level 預測 logits（或 row embedding）。

### 三、五個 stage 與程式對應（簡明表）

| PyTorch-Frame stage | ExcelFormer 中的對應位置 (檔案/函式/物件) | 資料型態 / 說明 |
|---|---:|---|
| start | `start_fn(...)` + `gnn_after_start_fn(...)` | 原始 pandas DataFrame；`gnn_after_start_fn` 在 concat 全表後用 row-level KNN 建圖並訓練 GNN，結果回寫為新的 numeric columns（離線 precompute）。 |
| materialize | `materialize_fn(...)` | 產生 `TensorFrame`（`train_tensor_frame` 等）與 `DataLoader`；包含 `CatToNumTransform`、`MutualInformationSort` 等預處理。 |
| encoding | `StypeWiseFeatureEncoder` / `ExcelFormerEncoder`；呼叫：`x, _ = encoder(tf)` | 把 TensorFrame 映射成 `x` shape `[batch, num_cols, channels]`。 |
| columnwise | `convs` (list of `ExcelFormerConv`)；`for conv in convs: x = conv(x)` | 多層 column-interaction，輸入/輸出仍為 `[batch, num_cols, channels]`。 |
| decoding | `ExcelFormerDecoder`；`out = decoder(x)` | 聚合欄位嵌入成 row-level logits / predictions。 |

### 四、GNN 插入點的行為（程式對應與語意）


---

## 模型 pipeline（上游 → 下游）與 PyTorch-Frame 對應 — 範例導覽

簡短說明：下面以一個具體例子（銀行信用資料，1000 筆，20 個欄位，其中 3 個是 category）說明 ExcelFormer 的資料處理流程，並把每一步對應回 PyTorch-Frame 的 stage。

- 範例資料：train_df (700 rows), val_df (150 rows), test_df (150 rows)，欄位 = ['age','income',...,'cat_feature1','target']。

1) Start
   - 動作：讀入三個 DataFrame，可能執行簡單檢查或填值。對 ExcelFormer 而言通常是 pass-through。
   - PyTorch-Frame 對應：start_fn。

2) Materialize
   - 動作：把 DataFrame 轉成 model 所需的 tensors/encodings。ExcelFormer 使用 `StypeWiseFeatureEncoder`（數值標準化、類別 embedding 建立、缺值指示符等），並產生 per-column embeddings 與 per-row mask。範例：把 'age' 標準化成 float tensor，把 'cat_feature1' 轉為 embedding index 再做 embedding lookup。
   - PyTorch-Frame 對應：materialize_fn（產生 DataLoader、dataset、encoder 的 metadata）。

3) Encoding
   - 動作：每個 column 先透過欄位型別適配器（stype-wise）轉成向量表示（embedding），再合併成 row-level tensor。範例：'age' → vector(16)，'cat_feature1' → embedding(16)，組合成 row vector (num_cols × 16)。
   - PyTorch-Frame 對應：encoding 階段（StypeWiseFeatureEncoder 的 forward）。

4) Column-wise interaction / Backbone
   - 動作：對 column-level或row-level表示進行互動（ExcelFormer 的 conv 層或 transformer block），可在此插入 GNN（per-batch 或 precompute）。範例：對每個 row 的 per-column vectors 做多層 conv，產生新的 column-aware 表示，再做 pooling。
   - PyTorch-Frame 對應：column-wise stage (column interactions、block stacks)、或 backbone。

5) Decoding
   - 動作：把融合後的 representation 平均/concatenate，經過 MLP head 做最終預測（classification/regression）。範例：pooling → 2-layer MLP → sigmoid 得到信用違約機率。
   - PyTorch-Frame 對應：decoding stage（decoder head / downstream evaluation）。

注意：如果在 start/materialize 做 precompute GNN，則 materialize 的輸出會被替換（embedding DataFrame 或新的 tensors），要確保在 precompute 訓練中只用 train 節點計算損失以避免洩漏。
### 五、重要注意事項與建議（摘要）
- Data leakage 風險：start/materialize/decoding 若在建圖或訓練時包含 test nodes 或 test labels，會導致資訊洩漏。建議：在 GNN 訓練時只用 train nodes（或將 test nodes視為 unseen inference nodes）。
- Computational cost：encoding/columnwise 的 per-row columns 完全圖會造成 O(num_cols^2) 的邊數，若欄位數多需稀疏化（top-k、learnable mask）。
- 訓練 regime 差異：start/materialize/decoding 多為離線或 post-hoc（通常訓練數百 epoch），而 encoding/columnwise 則為 joint-train（和主模型同時訓練）。比較策略效果時應固定 seed、split、graph 建構參數與 GNN 容量。

---

如需後續處理，可選擇：
- 將 `gnn_after_*` 改為只在 train nodes 上訓練（避免 leakage）；
- 把 encoding/columnwise 的 dense edge 構造改為 top-k 範例；
- 或生成一個對照實驗 runner（統一比較五個插入點）。

檔案位置：`models/pytorch_frame/excelformer_pytorch_frame_mapping.md`
