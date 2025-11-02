## Trompt 與 PyTorch-Frame 五個 stage 對照說明

本檔案根據 `models/pytorch_frame/trompt.py` 的實作整理 Trompt（TromPT）模型的 pipeline，並把程式中的關鍵位置對應到 PyTorch-Frame 的五個 stage（start、materialize、encoding、columnwise、decoding）。同時說明可插入 GNN 的位置、語意、發現的風險，以及推薦的改進方向。

### 一、一句話精要
- Trompt 在 `materialize_fn` 中物化資料；在 `trompt_core_fn` 裡建立多層 `StypeWiseFeatureEncoder`（encoding）、一系列 `TromptConv` 層負責 prompt-based column-wise interaction（columnwise），並以 `TromptDecoder` 把提示向量解碼為預測（decoding）。程式提供 `gnn_after_start_fn`、`gnn_after_materialize_fn`、`gnn_decoding_eval`，以及在 `encode_batch`（encoding）與 `process_batch_interaction`（columnwise）中的 `gnn_stage` hooks 可做 joint-train GNN。

### 二、Trompt 的 pipeline（程式如何處理 input）
1. start
   - `start_fn(train_df, val_df, test_df)`：預設直接回傳；`gnn_after_start_fn` 可在此與 `materialize_fn` 之間進行 row-level precompute GNN（把 embedding 回寫為新的數值欄位）。

2. materialize
   - `materialize_fn(...)`：使用 `Yandex(...)` 將 DataFrame 物化為 `TensorFrame` 並建立 `DataLoader`。回傳 `col_stats`、`train_loader/val_loader/test_loader`、`metric_computer` 等。

3. encoding
   - 在 `trompt_core_fn` 中，為每一層建立一個 `StypeWiseFeatureEncoder`，`encode_batch(tf)` 會呼叫每個 encoder 並返回該層的輸出（shape [batch, num_cols, channels]）。程式在 `encode_batch` 中若 `gnn_stage=='encoding'`，會於第一層對 encoder 的輸出（各 column embedding）重塑並套用 GNN（目前以每 row 的 columns 作為節點，並在 batch 內做重複的全連接 edges）。

4. columnwise interaction
   - `trompt_convs = ModuleList([TromptConv(...) ...])` 與 `process_batch_interaction(encoded_features, batch_size)`：透過多層 `TromptConv` 以提示向量（prompt）作為核心做 column-wise interaction；每層輸出為 `updated_prompt` shape `[batch, num_prompts, channels]`。
   - 在 `process_batch_interaction` 裡，如果 `gnn_stage=='columnwise'` 並開啟 GNN，程式會把 prompt reshape 成 `[batch * num_prompts, channels]`，在 batch 內建立全連接邊並套用 GNN（把 prompt 當作圖上的節點）。

5. decoding
   - `trompt_decoder = TromptDecoder(channels, out_channels, num_prompts)`，在 `forward_stacked` 中把每層的 prompt 傳入 decoder 產生 per-layer 預測，再取平均作最終輸出。程式也提供 `gnn_decoding_eval(...)`，收集由 prompts 生出的 embeddings（final prompt flatten），建立 KNN graph，訓練 supervised GNN 作為 post-hoc decoder。

### 三、程式中 GNN 插入點與語意（簡表）

| PyTorch-Frame stage | 程式對應 (file / fn / 變數) | 實作語意 |
|---|---:|---|
| start | `start_fn` + `gnn_after_start_fn` | row-level precompute：concat train/val/test，建 KNN，訓練 GNN（MSE 重構），把 embedding 分回並回寫為新的數值欄位（離線注入）。注意是否限定 train-only loss（避免 leakage）。 |
| materialize | `materialize_fn` + `gnn_after_materialize_fn` | TensorFrame 層級的離線 GNN：把 numerical features 轉為 DataFrame、建圖、訓練 GNN、回寫並重新 materialize。 |
| encoding | `encode_batch`（多層 encoders）；`gnn_stage=='encoding'` hook（只在 layer 0 應用） | 程式把 layer 0 的 encoder 輸出 reshape 為 `[batch * num_cols, channels]`，對每個批次內的 column-nodes 建全連接 edge（目前為 dense all-to-all），再套用 GNN；語意上這是每 row 內 columns-as-nodes 的 column-level GNN，但目前為了效率/實作會在 batch 範圍內合併做處理。 |
| columnwise | `process_batch_interaction` + `gnn_stage=='columnwise'` hook | 把 prompt vectors（per-layer）視為節點並進行 batch-內 GNN（reshape prompts -> `[batch * num_prompts, channels]`）；這代表對 prompt-node 的跨-row message passing，也可被視為一種 columnwise enhancement。 |
| decoding | `trompt_decoder` + `gnn_decoding_eval` | post-hoc row-level GNN：收集所有 final-prompt embeddings、建 KNN graph、訓練 supervised GNN（使用 train_mask 作 loss 計算）作為 decoder 的補充或替代。 |

### 四、發現的風險與實務建議
- 資訊洩漏風險：
  - `gnn_after_start_fn` / `gnn_after_materialize_fn` 的實作會 concat train/val/test 並訓練 GNN；務必在 loss 計算時只使用 train 範例（目前兩函式建立 train/val/test mask，但實際 loss 在某些地方並未被限制），否則會造成泄漏。建議將重構或 supervised loss 限制到 train_mask，val/test 僅做 inference。 

- Graph 稀疏化與效能：
  - 程式在 encoding/columnwise hooks 中用 dense all-to-all edges（雙 loop 建 edge_index，O(n^2)），在 batch 大或 prompt/col 數大時會導致記憶體與計算瓶頸。建議改為 top-k KNN（在 embedding 空間或按相似度）或限制每個 node 的鄰居數（k<<batch_size），使用 `NearestNeighbors` 或 GPU 加速庫 (faiss)。

- column-level vs row-level 語意：
  - encoding hook 的實作透過 reshape 把 columns 當作節點（columns-as-nodes），這是合理的 column-level GNN 實作，但在程式中它是以 batch 內合併的方式實作（因此會跨不同 rows 的 columns 互連，除非 offset 被正確處理）；務必確保 edge_index 的 offset/分簇正確，或改為 per-row小圖建立以避免 row 間不當混合。程式目前為每 batch 建 offset 以分開每 row，但若改用 KNN，需要注意不要跨 row 混合 columns。 

- decoding GNN：
  - `gnn_decoding_eval` 的流程是較安全的：它先訓練/提取標準模型 embeddings（可限制 samples），然後在 embedding space 建 KNN 並僅在 train_mask 上訓練 GNN supervised loss（早停），這種 post-hoc GNN 通常更容易避免泄漏且能較清楚量化貢獻。 

### 五、可立即執行的改進（我可以幫你做）
1. 修 patch：在 `gnn_after_start_fn` / `gnn_after_materialize_fn` 裡把 GNN 訓練 loss 限制到 train_mask，並使 val/test 僅用作 inference （避免 leakage）。
2. 把 encoding/columnwise 中的 dense all-to-all edge 建置替換為 top-k KNN（在每個小圖/每個 batch offset 範圍內計算），提供一個 `safe_knn_edges` helper。 
3. 提供一 safe 範例：如何做 per-row column-level GNN（對每 row 的 columns 構建小圖並 batch-execute），並估算計算成本與記憶體需求。 
4. 幫你產生一個實驗 runner，系統化比較五個插入點（固定 seed、split、GNN 架構與超參），並匯出 metric + compute cost。 

### 六、快速小結
- 對應關係：
  - start → `start_fn` + `gnn_after_start_fn`
  - materialize → `materialize_fn` + `gnn_after_materialize_fn`
  - encoding → `encode_batch` (layer encoders) + encoding-GNN hook
  - columnwise → `process_batch_interaction` (TromptConv prompt loop) + columnwise-GNN hook
  - decoding → `trompt_decoder` + `gnn_decoding_eval`

檔案位置：`models/pytorch_frame/trompt_pytorch_frame_mapping.md`


## Trompt — 模型 pipeline（上游 → 下游）範例

範例設定：小型分類資料集，800 筆、20 欄，train/val/test = 560/120/120。

1) Start
  - 讀入 DataFrame；可在此階段插入 precompute row-level GNN。

2) Materialize
  - 物化為 `TensorFrame` 與 DataLoader，準備多層 encoder 與 prompt 結構。

3) Encoding
  - 多層 `StypeWiseFeatureEncoder` 產生 per-layer column embedding；layer0 可插入 encoding-stage GNN（columns-as-nodes）。

4) Column-wise interaction
  - `TromptConv` 層透過 prompt 向量做 column-wise message passing，產生 prompt 表示供 decoder 使用。

5) Decoding
  - `TromptDecoder` 把 prompt 聚合成預測；可用 decoding GNN 做後處理評估。

PyTorch-Frame 對應：start → materialize → encoding → columnwise → decoding。
