# SCARF — PyTorch-Frame mapping与GNN插入建議

語言: 中文

本文檔把 `models/custom/scarf.py` 的實作對應到 PyTorch-Frame 的四個 stage（start / materialize / encoding / column-wise / decoding），並列出可插入 GNN 的位置、現有實作的注意事項（包含資料滲漏 risk）、以及低風險的修改建議。

## 簡短結論
- `scarf.py` 已經採用典型的五階段流程：`start_fn`, `materialize_fn`, `scarf_core_fn`（內含 encoder → columnwise → decoder）以及三個外部 GNN Hook：`gnn_after_start_fn`, `gnn_after_materialize_fn`, 和 decoding-stage GNN 在 `scarf_core_fn` 中實作。
- 實作中包含若干良好實踐（在 encoding/columnwise stage 使用 detach 建圖以避免梯度流出），但也存在潛在資料滲漏（precompute GNN 訓練時沒有限定只用 train 節點來計算有監督或自編碼損失）。

---

## 對應表（PyTorch-Frame stage -> `scarf.py`）

  - 函數: `start_fn(train_df, val_df, test_df)`
  - 作用: 初步檢查 / pass-through。`gnn_after_start_fn` 在該 repo 實作（位於同檔）作為可選的 precompute GNN hook。

  - 函數: `materialize_fn(train_df, val_df, test_df, dataset_results, config)`
  - 作用: 標準化、移除常數欄、產生 `SCARFDataset`、`DataLoader` 以及訓練參數（batch_size、epochs、device、task_type）。
  - Hook: `gnn_after_materialize_fn(material_outputs, dataset_results, config, task_type)` 會讀出 materialized loaders/datasets、把所有特徵合併為矩陣、以 kNN 建圖並訓練一個自編碼式 GCN（SimpleGCN），訓練完後把 GNN embeddings 轉回 DataFrame 並替換 `material_outputs` 中的 loader/dataset（也回傳 gnn_early_stop_epochs）。

  - 在 `scarf_core_fn`，encoder 實作為 `ScarfEncoder`。
  - 可插入 GNN: `gnn_encoding`（若 `gnn_stage == 'encoding'` 則建立 `SimpleGCN` 並在 forward 中套用）。
  - 實作細節: 每個 batch 中會用 `get_edge_index(z_detached)`（呼叫 `knn_graph`）基於 z（encoder 輸出）建立邊，並在 `z = gnn_encoding(z, gnn_encoding_edge_index)` 使用 GNN。注意：在建立圖時使用 `.detach()`，避免梯度穿越到圖構建步驟，這是良好做法。

  - 在 `scarf_core_fn` 中的 `columnwise = ScarfColumnwiseLayer(...)`，對 encoder 輸出做欄位式交互（paper 對應 PyTorch-Frame 的 column-wise stage）。
  - 可插入 GNN: `gnn_columnwise`（若 `gnn_stage == 'columnwise'` 則建立 `SimpleGCN` 並在 columnwise 之後套用）。同樣以 batch-level knn_graph 建邊並 detach 建圖。

  - `decoder = ScarfDecoder(...)` 並在 training/validation/test 的 forward path 中使用。
  - decoding-stage GNN: 當 `gnn_stage == 'decoding'` 時，`scarf_core_fn` 會在外部把 encoder/columnwise 的 embeddings（train/val/test）全部蒐集、對 embedding 建圖（kNN）、並以有監督方式訓練一個 SimpleGCN 作為最終預測器（supervised GNN over instance-graph）。該程式碼會建立 train/val/test mask 以便計算度量。

```

## SCARF — 模型 pipeline（上游 → 下游）範例

範例設定：信用評分資料，3000 筆、18 欄，train/val/test = 2100/450/450。

1) Start
   - 讀入 DataFrame；`gnn_after_start_fn` 可做整表 GNN 自編碼 precompute 並回寫 embedding。

2) Materialize
   - `materialize_fn` 做標準化、移除常數欄，建立 `SCARFDataset` 與 DataLoader。

3) Encoding
   - `ScarfEncoder` 取得 latent 表示；若 `gnn_stage=='encoding'` 則在 latent 上加入 GNN（batch-level knn）。

4) Column-wise interaction
   - `ScarfColumnwiseLayer` 做欄位層級交互；可在此插入 `gnn_columnwise` 增強表示。

5) Decoding
   - `ScarfDecoder` 輸出預測；若 `gnn_stage=='decoding'`，則會收集 embeddings 並在 row-graph 上訓練 supervised GNN。

PyTorch-Frame 對應：start → materialize → encoding → columnwise → decoding。

```

## 重要實作觀察與風險（現有 code points）

1. knn_graph helper
   - 檔中已有 `knn_graph` helper（會 handle N<=1 與 k 的上限），並用 sklearn.NearestNeighbors 在 CPU 上建圖，這是穩健且對 GPU memory 友善的選擇。建議把此 helper 抽到 shared utils 供其它 models 重用。

2. precompute GNNs（`gnn_after_start_fn` / `gnn_after_materialize_fn`）
   - 目前兩個 hook 都把 train/val/test concat 成一個 all_X，對整個 all_X 做自編碼式 MSE 訓練（loss = mse(out, x)），然後把 embedding 分回 train/val/test 並回填到資料集或回傳 emb DataFrames。
   - 風險: 訓練時沒有把損失限定為 train 節點（例如只對 train_mask 的節點計算損失），這會導致資料滲漏（train 階段看到 test/val 資料的特徵分布 / 重建誤差），使下游評估不可信。必須改成只在 train_mask 節點上監督或使用無標籤的自監督（並在評估時小心處理）。

3. decoding-stage GNN
   - 在 decoding-stage 實作中：作者已生成 `train_mask` / `val_mask` / `test_mask` 並建立 y_tensor。要確認在訓練 supervised GNN 時（loss/backward）僅計算 train 節點的損失；驗證時計算 val/test 的度量（且不要用 val/test 改動模型權重，除非是在早停或監督式 validation 的情況並只用 val 做模型選擇）。

4. Encoding/Columnwise 的 batch-level knn_graph
   - 目前針對每個 batch 都重建 edge_index，並在建圖時使用 `.detach()`，然後送入 GNN；這避免了圖構建導致額外的 autograd 開銷，與可能的梯度通過 knn 的問題，是正確做法。但要注意：對小 batch 或高維 embedding，knn 的計算成本可能變高；可考慮按 epoch/若干步驟重建或使用 approximate kNN library（FAISS）以提速/節省 memory。

---

## 建議改動（低風險、易於回滾）

1) 防止 precompute GNN 資料滲漏（高優先）
   - 在 `gnn_after_start_fn` 和 `gnn_after_materialize_fn` 的訓練迴圈中，當計算自編碼 MSE loss 時，僅使用 train_mask 節點參與 loss：
     - out = gnn(all_X, edge_index)
     - loss = mse(out[train_mask], all_X[train_mask])
   - 這會保證 GNN 學到的表示只由 train 節點的重建誤差驅動（但仍可把所有節點一起 forward 以保留 graph 結構）。

2) decoding-stage supervised GNN：只在 train 節點計算 supervised loss
   - 在 supervised GNN training loop 中，當計算 logits = gnn(all_emb, edge_index) 後，計算 loss 只用 logits[train_mask] 與 y_tensor[train_mask]。
   - 在驗證/測試時計算 val/test 指標但不要回傳梯度。

3) 新增 config 選項
   - `gnn_mode`: 'forward-only' or 'joint-train'。
     - 'forward-only'：GNN 在 forward 時使用，但其參數不加入 optimizer（預訓練或凍結）。
     - 'joint-train'：GNN 參數加入 optimizer，與 encoder/columnwise 一起訓練。
   - `gnn_pretrain_train_only`: bool，當 True 時 `gnn_after_*` 的 pretrain 只計算 train 節點損失。

4) 保留/抽出 knn_graph helper
   - 檔案已有穩健 `knn_graph` 實作，建議把它搬到 `models/pytorch_frame/utils.py` 或 repo 公共 utils，所有 models 共用，避免重複。

5) Sparsification / top-k 保護
   - knn_graph 已做 top-k，若要進一步安全，建議在 `knn_graph` 中加入 `max_edges` 或閥值來拒絕過多邊；或支援 approximate knn（FAISS）以加速大型資料集。

6) logging / instrumentation
   - 在 precompute GNN 訓練中回傳 `gnn_early_stop_epochs`（已有），並將 `gnn_mode` 與 `gnn_*` 超參數寫入 `material_outputs['info']` 以利實驗紀錄和重現。

---

## 小型契約（Contract）
- Inputs: pandas DataFrame (train_df/val_df/test_df) 與 `config`（包含 gnn_knn, gnn_hidden, gnn_epochs, gnn_mode 等）和 dataset_results（包含 task_type）。
- Outputs: 與現有 `main()` 介面相同—返回訓練/驗證/測試度量字典，或在 `gnn_after_*` 時替換 material_outputs 的 loader/dataset（若 precompute embedding 被使用）。
- Error modes: kNN on tiny N (N<=1) handled by knn_graph; OOM risk for large N when building graphs—應用 top-k/approx knn。

---

## 可能的邊界情況（Edge cases）
1. 小資料集 (N<=1): knn_graph 已處理並回傳空圖。
2. 非數值或缺失值: `materialize_fn` 已用 StandardScaler，但需確保缺失值被先處理。
3. 不同 task types: decoding-stage GNN 對於多類別/迴歸/二分類都有分支，請確保 y_tensor dtype 與 loss 一致。

---

## 建議的下一步（可直接實作）
1. 把 `knn_graph` 抽到 `models/pytorch_frame/utils.py` 並讓其它模型重用。
2. 在 `gnn_after_start_fn` 與 `gnn_after_materialize_fn` 的訓練迴圈把 loss 限定為 `train_mask`（高優先，避免滲漏）。
3. 在 decoding-stage supervised GNN training loop 中同樣只在 train_mask 上計算 loss。
4. 新增 `gnn_mode` config 並根據其值決定是否把 GNN parameters 加到 `optimizer`。
5. 若要做大規模實驗，考慮換用 approximate kNN (FAISS) 或在 epoch 間隔重建 knn_graph 以減少頻繁呼叫開銷。

---

## 參考位置（檔案 / 函數）
- `models/custom/scarf.py`:
  - `start_fn`, `gnn_after_start_fn`
  - `materialize_fn`, `gnn_after_materialize_fn`
  - `scarf_core_fn` (encoder/columnwise/decoder + decoding-stage GNN)

---

若需要，我可以：
- 直接在 `scarf.py` 中實作「只在 train_mask 上計算 precompute / decoding GNN loss」的小補丁並執行快速 smoke test；或
- 把 `knn_graph` 抽成共用 utils；或
- 實作 `gnn_mode` 與 `gnn_pretrain_train_only` config 並將其連接到 optimizer/訓練迴圈。請告訴我你想先做哪一項，我會立刻開始並更新 todo 狀態。
