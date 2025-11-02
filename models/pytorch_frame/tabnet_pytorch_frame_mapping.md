## TabNet 與 PyTorch-Frame 五個 stage 對照說明

此檔案根據 `models/pytorch_frame/tabnet.py` 的實作整理 TabNet 的 pipeline，並把程式中的關鍵部分對應到 PyTorch-Frame 的五個 stage（start、materialize、encoding、columnwise、decoding）。另外列出在各插入點加入 GNN 的語意、程式對應位置、以及實務注意事項與建議。

### 一、精要（一句話）
- TabNet 實作在 `materialize_fn` 產生 `TensorFrame/DataLoader`，在 `tabnet_core_fn` 中使用 `StypeWiseFeatureEncoder`（encoding）、再用 TabNet 特有的 FeatureTransformer + AttentiveTransformer 決策步驟（column-wise interaction），最後以線性層輸出（decoding）。程式提供了 `gnn_after_start_fn`、`gnn_after_materialize_fn`、`gnn_decoding_eval`，以及 `gnn_stage` hooks（`encoding`、`columnwise`）可以在不同位置插入 GNN。

### 二、TabNet 原始 pipeline（程式如何處理 input）
1. start
   - 檔案/函式：`start_fn(train_df, val_df, test_df)`。
   - 行為：預設直接回傳 DataFrame，作為在 pipeline 開頭插入 precompute row-level GNN 的 hook（對應 `gnn_after_start_fn`）。

2. materialize
   - 檔案/函式：`materialize_fn(...)`。
   - 行為：使用 `Yandex(...)` 包裝並呼叫 `dataset.materialize()`，然後根據 `split_col` 取得 `train_tensor_frame/val_tensor_frame/test_tensor_frame` 與相對應的 `DataLoader`。同時建立 `col_stats`、`metric_computer` 等。這個階段等價於 PyTorch-Frame 的 Materialization。

3. encoding
   - 程式位置：`tabnet_core_fn` 內的 `feature_encoder = StypeWiseFeatureEncoder(...)` 與內部 `encode_batch(tf)`。
   - 行為：將 `TensorFrame` 轉成 shape 為 `[batch, num_cols, cat_emb_channels]` 的張量，然後 flatten 為 `[batch, num_cols * cat_emb_channels]` 並做 BatchNorm，回傳作為 TabNet controller 的輸入。

4. column-wise interaction (TabNet decision steps)
   - 程式位置：`process_batch_interaction(x, return_reg=False)`；內含 `feat_transformers` 與 `attn_transformers`，在迴圈中累積每個 decision-step 的 `feature_x`，最後以 sum 聚合並送入輸出層。
   - 行為：這裡是 TabNet 的核心——透過 AttentiveTransformer 產生 attention mask（per-feature selection），再把 mask 應用到 x 並用 FeatureTransformer 處理，代表欄位之間的交互與選擇機制（PyTorch-Frame 的 column-wise interaction）。

5. decoding
   - 程式位置：`lin = Linear(split_feat_channels, out_channels)` 與 `out = lin(out)`。
   - 行為：對聚合後的特徵做線性投影得到最終 logits 或回歸預測（PyTorch-Frame 的 decoder）。

### 三、程式內的 GNN 插入 hook 與語意（model file 對應）

| PyTorch-Frame stage | 程式對應 (file / 函式 / 變數) | GNN 在此的語意 |
|---|---:|---|
| start | `start_fn(...)` + `gnn_after_start_fn(...)` | row-level precompute：把 train/val/test concat，對 rows 建 KNN graph，訓練小 GNN（目前用 MSE 重構）並把 embedding 回寫為新 numeric 欄位；適用於離線特徵注入。 |
| materialize | `materialize_fn(...)` + `gnn_after_materialize_fn(...)` | row-level 在 TensorFrame 之後做的 precompute：會把 TensorFrame 轉為 df、建 KNN、訓練 GNN，然後重新 materialize（回寫 embedding 並建立新的 tensor frame）。 |
| encoding | `feature_encoder = StypeWiseFeatureEncoder(...)` 與 `encode_batch(tf)`；`gnn_stage=='encoding'` hook 建立 `gnn_encoding` 並在 `forward` (per-batch) 上呼叫。 | joint-train 的 row-level GNN（或可改為 column-level）：此實作把整個 batch 的 flattened encoding 當成 node features，並用 batch 內的全連接/knn graph 做 GNN。語意是對「batch 中的實例」做 message passing，強化 row-level表示。 |
| columnwise | `process_batch_interaction(...)`：TabNet 的 attention mask 與特徵分裂階段；`gnn_stage=='columnwise'` hook 建立 `gnn_columnwise` 並在每個 decision-step 的 `feature_x` 上呼叫（目前在 batch 維度上建立全連接/knn graph）。 | 可視為對 decision-step 的特徵向量做跨-row GNN，或（更原生）把「columns 當作節點」在每個 row 內做 column-level GNN（目前實作是跨 rows 的 GNN，語意上與原 TabNet 的欄位 selection 不完全相同）。 |
| decoding | After `out = lin(out)` / 或 `gnn_decoding_eval(...)` | post-hoc row-level GNN 校正或替代 decoder：`gnn_decoding_eval` 會收集所有 row 的 embeddings (train/val/test)，建 KNN graph，並使用 supervised loss 在圖上訓練 GNN（早停、回報 val/test 指標）。 |

### 四、實務注意事項（重要）
- 資訊洩漏風險：
  - `gnn_after_start_fn` 與 `gnn_after_materialize_fn` 在程式中目前會把 train/val/test concat 後建圖並訓練 GNN（雖然函式內建立了 train/val/test mask，但目前的重構 loss 在 `gnn_after_start_fn` 與 `gnn_after_materialize_fn` 實作中使用的是完整的 MSE on all nodes —— 這會導致 leakage）。
  - 建議：在 precompute 階段只用 train 範例訓練 GNN（loss 只計算於 train_mask 範圍），再把 GNN 至少以 inductive 方式對 val/test embed（即不使用 val/test labels；或乾脆只在 train data 上做 representation learning 並將產生的 embedding 當作特徵注入）。

- encoding/columnwise 的 GNN 實作需注意語意不一致：
  - 目前 `encoding` hook 把 flattened encoding 當成 node features（node = row in batch），並在 batch 內建立全連接圖，這與「column-level GNN（columns as nodes）」不同。
  - 若你的目標是 column-level message passing（在每個 row 的 columns 之間傳遞訊息），需要在 encoding 階段把 encoder 的輸出 reshape 為 `[batch * num_cols, channels]`，對每個 row 建小圖（num_cols 節點）並合併 batch axis 執行 GNN（但計算成本會顯著增加）。

- 批次/圖的稀疏化建議：
  - 避免全連接 dense graph（在 code 中常見的雙重 for-loop 建 edge_index），改用 top-k 或 KNN 在 embedding 空間建立稀疏邊可以大幅降低記憶體與計算成本。可在 batch 內使用 `sklearn.neighbors.NearestNeighbors` 或 PyTorch 邊緣採樣。 

- 一致性/可比較性：比較不同插入點時請固定：random seed、data split、GNN 超參（k、hidden、layers）、訓練策略（joint vs precompute）與 early-stop 條件；並記錄訓練時間與 GNN precompute 時間。

### 五、具體改進建議（可直接 patch / 實作）
1. 在 `gnn_after_start_fn` / `gnn_after_materialize_fn` 中，只用 `train_mask` 上的節點計算 GNN loss（例如：loss = mse_loss(out[train_mask], x[train_mask])），把 val/test 僅用來推斷 embedding（no label access）。
2. 在 `encoding` / `columnwise` hook 裡用 top-k KNN 建圖：
   - 以 `x` 或 `feature_x` 的 L2 空間做 KNN（k 小於 batch size），只連接 top-k neighbor，避免雙 loop 建全連接表。
   - 範例：使用 `NearestNeighbors` 得到 indices，然後轉為 `edge_index`。對於 GPU-only 流程，可考慮 `faiss` 或自製 top-k sampling。 
3. 若要做真正的 column-level GNN（columns as nodes）：在 `encode_batch` 後保留 `(batch, num_cols, channels)`，reshape 為 `(batch * num_cols, channels)`，並為每個 row 建 graph（num_cols 節點）——要注意這會顯著增加計算，僅在 num_cols 小時實用。 
4. `gnn_decoding_eval` 的流程已較為健全（training loss 僅在 train_mask 上計算），但仍要確保不要在早期步驟泄漏 test label / embedding。若使用 decoding GNN 作為最終 ensemble，紀錄其獨立貢獻與 compute cost。 

### 六、快速小結
- 對應關係：
  - start → `start_fn` + `gnn_after_start_fn`
  - materialize → `materialize_fn` (+ optional `gnn_after_materialize_fn`)
  - encoding → `StypeWiseFeatureEncoder` + `encode_batch`
  - columnwise → `process_batch_interaction` (TabNet 的 attn + feat transformer loop)
  - decoding → `lin` output layer (+ `gnn_decoding_eval` 可作 post-hoc GNN decoder)

- 我可以幫你做的後續步驟：
  1. 自動把 `gnn_after_start_fn` / `gnn_after_materialize_fn` 改成只在 train 範例上計算 loss（避免 leakage）；
  2. 在 `tabnet_core_fn` 中把 `encoding` / `columnwise` hook 的全連接 edge 建置改為 top-k KNN 範例；
  3. 或者直接幫你產生一個實驗 runner，比較五個插入點（同 seed/split/GNN 架構），並輸出效果與計算成本。

檔案位置：`models/pytorch_frame/tabnet_pytorch_frame_mapping.md`


## TabNet — 模型 pipeline（上游 → 下游）範例

範例設定：電商點擊預測，20000 筆、40 欄，train/val/test = 16000/2000/2000。

1) Start
   - 載入 DataFrame，`gnn_after_start_fn` 可作離線特徵注入。

2) Materialize
   - 物化為 `TensorFrame` 與 DataLoader，建立 col_stats 與 metric_computer。

3) Encoding
   - `StypeWiseFeatureEncoder` 轉欄位為向量，並做 batch-level initial processing。

4) Column-wise interaction
   - TabNet 的 decision steps（AttentiveTransformer + FeatureTransformer）在此做欄位選擇與交互。

5) Decoding
   - 線性輸出層或用 decoding-stage GNN 做後續 supervised training。

PyTorch-Frame 對應：start / materialize / encoding / columnwise / decoding。
