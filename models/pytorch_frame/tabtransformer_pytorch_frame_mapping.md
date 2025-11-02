## TabTransformer 與 PyTorch-Frame 五個 stage 對照說明

本檔案基於 `models/pytorch_frame/tabtransformer.py` 的實作整理 TabTransformer 的 pipeline，並把程式中的主要構件（編碼、列間互動、解碼）對應到 PyTorch-Frame 的 stage（start、materialize、encoding、columnwise、decoding）。同時說明檔案中提供的 GNN 插入點、語意與實務建議。

### 一、精要
- TabTransformer 實作：
  - Materialization：`materialize_fn` 建立 `TensorFrame` 與 `DataLoader`；
  - Encoding：`EmbeddingEncoder` (categorical) + `StackEncoder` (numerical)，在 `encode_batch(tf)` 中產生類別與數值的 embedding；
  - Column-wise interaction：`TabTransformerConv` 列層（只對 categorical features），封裝於 `process_batch_interaction`；
  - Decoding：MLP (`decoder`) 對拼接後的向量輸出結果；
  - 程式同時提供 `gnn_after_start_fn`、`gnn_after_materialize_fn`、`gnn_decoding_eval`，以及 `gnn_stage` hooks (`encoding`、`columnwise`) 在前向流程中啟用 GNN。 

### 二、程式如何處理 input（pipeline 步驟）
1. start
   - `start_fn(train_df, val_df, test_df)`：目前只是 pass-through，但程式中有 `gnn_after_start_fn` 可在此與 `materialize_fn` 之間插入 GNN（row-level precompute）。

2. materialize
   - `materialize_fn(...)`：使用 `Yandex(...)` 包裝 DataFrame，呼叫 `dataset.materialize()`，並根據 `split_col` 產生 `train_tensor_frame/val_tensor_frame/test_tensor_frame` 與 `DataLoader`。回傳 `col_stats`、`metric_computer` 等供後續使用。

3. encoding
   - `tabtransformer_core_fn` 內的 `encode_batch(tf)`：
     - categorical: `cat_encoder = EmbeddingEncoder(...)` → `x_cat` + `pad_embedding` (positional pad)；
     - numerical: `num_encoder = StackEncoder(...)` → `x_num` → layer norm；
     - `encode_batch` 回傳一個 list of tuples [(tensor, 'categorical'|'numerical')]

4. column-wise interaction
   - `process_batch_interaction(encoded_features)`：對 categorical features 使用多層 `TabTransformerConv`（即 Transformer-like column interaction）並返回處理後的 features。這即 PyTorch-Frame 的 column-wise interaction 階段。

5. decoding
   - `decoder = Sequential(...)`：把 flatten 後的 categorical 與 numerical 拼接進 MLP，輸出 logits/regression 值。

### 三、程式中的 GNN 插入點（對應與語意）

| PyTorch-Frame stage | 程式對應 (file / 函式 / 變數) | 語意與實作摘要 |
|---|---:|---|
| start | `start_fn(...)` + `gnn_after_start_fn(...)` | row-level precompute：把 train/val/test concat，對 rows 建 KNN graph，訓練 GNN（目前用 MSE 重構）並把 embedding 回寫為新 numeric 欄位，再交給 `materialize_fn`。注意：目前函式內會看到 train/val/test mask，但若訓練 loss 未限制在 train 範圍會造成 leakage。 |
| materialize | `materialize_fn(...)` + `gnn_after_materialize_fn(...)` | row-level precompute on TensorFrame：`gnn_after_materialize_fn` 會把 numerical features 從 `TensorFrame` 轉回 df、建圖、訓練 GNN、把 embedding 回寫並重新 materialize dataset。此為一種離線注入（offline feature injection）。 |
| encoding | `encode_batch(tf)`；`tabtransformer_core_fn` 中 `if gnn_stage == 'encoding'` hook | joint-train 或 batch-level GNN：程式建立 `gnn`（`SimpleGCN_INTERNAL`），並在 `forward` 的 `apply_gnn_to_features` 中針對 categorical（shape [B, num_cols, channels]）與 numerical（shape [B, num_features]）做 batch-internal 全連接圖 / KNN 並套用 GNN。語意上此實作是跨 batch 中的 row-message-passing（node = row in batch）或 per-column GNN（對 categorical 每 column 做 GNN），視實作分支而定。 |
| columnwise | `process_batch_interaction(...)` + `if gnn_stage == 'columnwise'` hook (仍使用 `apply_gnn_to_features`) | 程式把 TabTransformerConv 的輸出（categorical features）或 numerical feature 送入 `apply_gnn_to_features`，同樣以 batch 內全連接/knn graph 做 GNN（預設每 column 分別做 GNN 或把 numerical 投影至 channels 再做 GNN）。此處的實作語意通常是跨-rows 的 GNN，而非每 row 內的 column-node GNN（若你想做 columns-as-nodes，需另外 reshape）。 |
| decoding | `decoder(...)` / `gnn_decoding_eval(...)` | `gnn_decoding_eval` 收集 train/val/test 的 embeddings（透過 `get_embeddings_batch_wise`），建 KNN graph，再做 supervised GNN 訓練（BCE/CE/MSE），用來作為 post-hoc GNN decoder 或 ensemble 修正。此階段的 GNN training loss 明確限定在 train_mask。 |

### 四、發現的風險與實務建議
- 資訊洩漏（最重要）
  - `gnn_after_start_fn` / `gnn_after_materialize_fn` 的目前實作把 train/val/test concat 再進行重構式訓練（MSE），如果 loss 沒有限制在 train 範圍或使用了 test labels，會造成嚴重 leakage。建議在 precompute 階段：
    - 僅用 train 範例訓練 GNN（loss 只計算 train_mask 範圍），
    - 或只在 train 上做 self-supervised pretraining，對 val/test 以推論 (inference) 方式取得 embeddings。

- encoding/columnwise GNN 的語意差異
  - 目前 `encoding`/`columnwise` hook 的實作重點是 batch-内 row-message-passing；若目標是 column-level message passing（columns as nodes inside a row），需要在 encoding 後把 categorical output 由 `[B, num_cols, channels]` reshape 成 `[B * num_cols, channels]` 並為每個 row 構建小圖（num_cols 節點），或以更複雜的 batching 策略合併多個 row 的 column graphs。計算成本會顯著上升。 

- 建圖與稀疏化
  - 程式中部分地方使用全連接雙重 loop 建 edge_index（O(B^2)），在 batch 大時會很慢。建議改為 top-k 或 KNN（embedding space）取代全連接，並使用 `NearestNeighbors` 或 GPU-friendly top-k libs（faiss）以加速。 

- 一致性與可比較性
  - 系統性對照不同插入點時，務必固定 random seed、data split、GNN 架構、k 值與訓練策略（precompute vs joint-train），並報告 compute cost (precompute time, extra params). 

### 五、可立即實作的改進（我可以直接幫你做）
1. 把 `gnn_after_start_fn` / `gnn_after_materialize_fn` 的訓練改為只在 train 範例上計算 loss（避免 leakage）；
2. 在 `tabtransformer_core_fn` 的 `apply_gnn_to_features` 中用 top-k KNN 取代全連接 edge 建置（示例 helper + patch）；
3. 如果你想做 column-level GNN（columns-as-nodes），我可以提供一個 safe reshape + batching 範例以及成本估算；
4. 建立一個簡單的實驗 runner 來比較五個插入點（固定 seed/split/GNN 超參），並紀錄評估指標與時間成本。

### 六、快速小結
- 對應關係：
  - start → `start_fn` + `gnn_after_start_fn`
  - materialize → `materialize_fn` (+ optional `gnn_after_materialize_fn`)
  - encoding → `encode_batch(tf)` + `apply_gnn_to_features` (when `gnn_stage=='encoding'`)
  - columnwise → `process_batch_interaction(...)` + `apply_gnn_to_features` (when `gnn_stage=='columnwise'`)
  - decoding → `decoder` + `gnn_decoding_eval`

檔案位置：`models/pytorch_frame/tabtransformer_pytorch_frame_mapping.md`


## TabTransformer — 模型 pipeline（上游 → 下游）範例

範例：分類任務，1200 筆、15 欄（多個 categorical + numerical），train/val/test = 840/180/180。

1) Start
  - 讀入 DataFrame；`gnn_after_start_fn` 可插入行級離線 GNN。

2) Materialize
  - `materialize_fn` 產生 `TensorFrame`，建立 categorical embedding 與 numerical encoder。

3) Encoding
  - categorical 使用 `EmbeddingEncoder`，numerical 使用 `StackEncoder`，合併為 model 輸入。

4) Column-wise interaction
  - `TabTransformerConv` 層在 categorical 上執行 column interaction（Transformer-like）。

5) Decoding
  - 拼接後送入 MLP decoder；`gnn_decoding_eval` 可做 post-hoc GNN 評估。

PyTorch-Frame 對應：start → materialize → encoding → columnwise → decoding。
