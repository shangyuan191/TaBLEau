# TabM — PyTorch-Frame 映射與 GNN 插入建議

語言: 中文

本文檔對應 `models/custom/tabm.py` 到 PyTorch-Frame 的五個階段（start / materialize / encoding / columnwise / decoding），並說明可以插入 GNN 的位置、目前實作的注意事項（包含資料滲漏風險）、以及可立即採取的低風險修改建議與實驗提示。

## 簡短結論
- `tabm.py` 已完整實作五階段設計，並在多個位置提供 GNN hook：`gnn_after_start_fn`, `gnn_after_materialize_fn`, 在 core 中以 `gnn_stage` 控制 `encoding` / `columnwise` 的 joint-train，與 `tabm_decoding_with_gnn` 作為 decoding-stage 的 supervised GNN decoder。
- 優點：knn_graph 有防護（N<=1 與 k 上限），encoding/columnwise 的 GNN 可以作 joint-train（參數加入 optimizer），decoding-stage 的 GNN decoder 也只在 train 範圍上計算 loss（good）。
- 需注意：precompute hooks (`gnn_after_start_fn`, `gnn_after_materialize_fn`) 目前以對全體節點計算 MSE loss（loss = F.mse_loss(out, x)），沒有限定僅計算 train 節點，這可能導致資料滲漏（high priority 修正）。

---

## Stage -> 檔案函數對應

- Start
  - `start_fn(train_df, val_df, test_df)` — dummy pass-through
  - Hook: `gnn_after_start_fn(train_df, val_df, test_df, config, task_type)` — 合併三個 DataFrame，建圖並以 SimpleGCN 做自編碼式重構訓練，最後以 embedding 替換原始特徵（回傳新 df 與 gnn_early_stop_epochs）

- Materialize
  - `materialize_fn(train_df, val_df, test_df, dataset_results, config)` — 數據預處理（QuantileTransformer）、建立 tensor、feature embeddings（PiecewiseLinearEmbeddings）與 metadata 回傳
  - Hook: `gnn_after_materialize_fn(X_train, y_train, X_val, y_val, X_test, y_test, config, task_type)` — 合併 tensors、建圖並以 SimpleGCN 訓練自編碼器，最後回傳 GNN 轉換後的 tensors

- Encoding
  - 在 `tabm_core_fn` 中建立 backbone，並把 backbone 分割成 `encoding_blocks` / `columnwise_blocks`。
  - 若 `gnn_stage == 'encoding'`，會建立 `gnn = SimpleGCN(d_block, gnn_hidden, d_block)` 並在 forward 裡基於 batch 的平均表示用 `knn_graph` 計算 edge_index，然後把 gnn 的輸出以 expand 回原來的形狀以供後續 block 使用（joint-train 可將 gnn 參數加入 optimizer）。

- Column-wise
  - columnwise blocks 在 core 裡作為 `columnwise_blocks`，若 `gnn_stage == 'columnwise'`，在 columnwise block 前後會以相似方式插入 GNN forward。

- Decoding
  - 當 `gnn_stage == 'decoding'`，`tabm_decoding_with_gnn` 會先訓練 encoder+columnwise（使用臨時 output 層），然後把所有資料通過 encoder 取得 embeddings，建造 kNN 圖，並用 supervised GNN decoder（SimpleGCN）在 train 範圍上計算 loss 並驗證/測試（該函數在訓練時正確使用 train 範圍計算 loss）。

---

## 重要實作觀察

1) knn_graph
   - 已包含防護（若 N<=1 返回空圖；k 上限為 N-1），並使用 sklearn 以避免在 GPU 上轉換大量資料。建議把該 helper 抽到共用 util。

2) precompute GNN（start / materialize）
   - 都使用 MSE loss over all nodes (loss = F.mse_loss(out, x) 或 F.mse_loss(out, X_all))，沒有用 train_mask 限定損失計算範圍，這會導致滲漏。

3) encoding/columnwise GNN
   - 在 forward 中以 batch-level 建圖：常見做法是對 batch 計算 mean 向量、呼叫 knn_graph 建圖，再用 GNN。注意 batch 小時需防護 k 值（程式已用 min(5, batch_size-1)）。

4) decoding-stage GNN
   - `tabm_decoding_with_gnn` 對 supervised GNN decoder 使用 `out[:n_train]` 作 loss，並在驗證時計算 val/test metric，這種 train-only loss 實作是正確的。

5) joint-train（gnn in all_params）
   - 如果 `gnn` 存在，其參數會被加入 `all_params`，因此在 encoding/columnwise 階段 GNN 將與 backbone 一起被訓練（end-to-end）。

---

## 風險與建議修補（優先級）

1) 高優先：修正 precompute GNN 的訓練損失以避免資料滲漏
   - 在 `gnn_after_start_fn` 與 `gnn_after_materialize_fn` 的訓練迴圈中，將 loss 改為只在 train 節點計算，例如：
     - 在 start hook：建立 train_mask（基於 n_train, n_val, n_test），計算 loss = F.mse_loss(out[train_mask], x[train_mask])
     - 在 materialize hook：相同邏輯，確保回傳 embedding 時不含 val/test 的監督影響
   - 這樣即使對整個圖 forward，也只用 train 節點的 supervision 更新參數，防止泄露驗證/測試資訊。

2) 中優先：新增 `gnn_mode` 與 `gnn_pretrain_train_only` config
   - `gnn_mode`: 'forward-only' 或 'joint-train'，控制是否把 GNN 參數加入 optimizer
   - `gnn_pretrain_train_only`: bool，若 True 則 precompute GNN 只在 train 節點計算 loss

3) 中優先：抽出 `knn_graph` 至共用 utils
   - 把 knn_graph 抽到 `models/pytorch_frame/utils.py`，並在該處提供 `max_edges`、`approximate`（FAISS）等選項以便大資料集使用

4) 低優先：記錄更多實驗資訊
   - 把 `gnn_early_stop_epochs`、gnn 超參數與 `gnn_mode` 記錄到 `material_outputs` 或 `results` 以支援實驗追蹤

5) 效能建議
   - 對大型資料集使用 approximate knn 或把 precompute GNN 改為 mini-batch neighbor sampling（例如 PyG 的 neighbor sampling）以降低記憶體與時間花費

---

## 小契約 (Contract)
- Inputs: `train_df/val_df/test_df` 或 materialized tensors (`X_train`, `y_train`, ...)，`config`（gnn_* 參數）與 `dataset_results['info']['task_type']`。
- Outputs: 與 `main()` 介面一致；precompute hook 可回傳替換後的 tensors 或 DataFrame；decoding hook 回傳 best_val_metric / best_test_metric 與 gnn_early_stop_epochs。

---

## 建議的下一步（我可以代為執行）
1. 立即在 `tabm.py` 中實作「precompute GNN 訓練僅以 train 節點計算 loss」的修補並執行快速 smoke test（建議，低風險）。
2. 把 `knn_graph` 抽取到 `models/pytorch_frame/utils.py`，讓所有模型共享（中風險，需小心改引用）。
3. 新增 `gnn_mode` 與 `gnn_pretrain_train_only` 配置選項，並在 core 的 optimizer 建構處與 precompute hook 中使用（中風險）。

請告訴我你想先做哪一項，我會把對應 todo 標為 in-progress、套用變更並執行最小驗證，然後回報 diff 與測試結果。

---

參考位置：
- `models/custom/tabm.py`: `start_fn`, `gnn_after_start_fn`, `gnn_after_materialize_fn`, `materialize_fn`, `tabm_core_fn`, `tabm_decoding_with_gnn`, `knn_graph`, `SimpleGCN`

```

## TabM — 模型 pipeline（上游 → 下游）範例

範例設定：分類任務，5000 筆、30 欄，train/val/test = 3500/750/750。

1) Start
   - 讀入 DataFrame；`gnn_after_start_fn` 可訓練全表 GNN 並回寫 embedding。

2) Materialize
   - `materialize_fn` 做 QuantileTransformer 與 PiecewiseLinearEmbeddings，回傳 tensors 與 embeddings module。

3) Encoding
   - 透過 EnsembleView 與 backbone 的 encoding_blocks 做前半段處理；可在此插入 encoding-stage GNN（batch-level）。

4) Column-wise interaction
   - backbone 的 columnwise_blocks 做後半段處理；若 `gnn_stage=='columnwise'` 可於此插入 GNN。

5) Decoding
   - 若 `gnn_stage=='decoding'`，會先訓練 encoder（使用暫時的 output）再提取 embeddings 並以 GNN 作為 decoder 在 row-graph 上訓練 supervised loss。

PyTorch-Frame 對應：start → materialize → encoding → columnwise → decoding。

```
