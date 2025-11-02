# SubTab — PyTorch-Frame 對應與 GNN 插入建議

語言: 中文

這份文件把 `models/custom/subtab.py` 對應到 PyTorch-Frame（start → materialize → encoding → column-wise → decoding）各階段，並總結了該實作中 GNN 的插入點、潛在風險（例如資料滲漏）、以及可立刻採取的低風險修補與實驗建議。

## 快速摘要
- `subtab.py` 已明確實作五階段流程，並提供三個外部 hook：`gnn_after_start_fn`, `gnn_after_materialize_fn`, `gnn_decoding_eval`。在 `subtab_core_fn` 中，`gnn_stage` 可在 `encoding` 或 `columnwise` 時以 joint-train 的方式加入 GNN（GNN 參數會加入 optimizer）。
- 優點：knn_graph 使用 sklearn、knn_graph 內使用 `detach()`（可避免 autograd 透過 knn 建圖），decoding 階段的 supervised GNN 已正確只在 train_mask 上計算損失。
- 需要修正：precompute / materialize 階段的 GNN 訓練（`gnn_after_start_fn` 和 `gnn_after_materialize_fn`）目前以對全體節點做 MSE loss 的自編碼訓練，未限定只用 train 節點來計算 loss，會導致資料滲漏。

---

## 檔案 -> Stage 對應

- Start
  - 函數: `start_fn(train_df, val_df, test_df)` — 直接回傳 (pass-through)
  - Hook: `gnn_after_start_fn(train_df, val_df, test_df, config, task_type)` — 對合併資料做 kNN 與 SimpleGCN 自編碼訓練，最後把 embeddings 分回並回傳新的 DataFrame

- Materialize
  - 函數: `materialize_fn(train_df, val_df, test_df, dataset_results, config)` — 把 DataFrame 包為 `TabularDataset` 與 `TorchDataLoader`
  - Hook: `gnn_after_materialize_fn(train_loader, val_loader, test_loader, config, task_type)` — 從 DataLoader 收集所有 X，合併並建圖、訓練 SimpleGCN、以 embeddings 創建新的 DataLoader 並回傳

- Encoding
  - 在 `subtab_core_fn` 中的 AEWrapper encoder (子集輸入) 負責 encoding
  - 若 `gnn_stage == 'encoding'`，會建立 `gnn_encoding = SimpleGCN(latent_dim, gnn_hidden, latent_dim)`，並在每個 batch 中基於 `latent_all` 呼叫 `knn_graph`（knn_graph 會 `.detach()` 以避免 gradient flow），然後把 `latent_all = gnn_encoding(latent_all, edge_index)` 再 decode

- Column-wise
  - SubTab 透過 `subset_generator` 生成子集並做 contrastive / reconstruction，`columnwise` 行為整合於 core 函數中（非獨立 module）
  - 若 `gnn_stage == 'columnwise'`，則以類似方式插入 `gnn_columnwise` 並把其參數加入 optimizer

- Decoding
  - `subtab_core_fn` 在訓練/驗證/測試結束後會提取 `z_train/z_val/z_test` 與標籤，若 `gnn_stage == 'decoding'`，`gnn_decoding_eval` 會合併 embeddings、建圖、並以 supervised GNN (SimpleGCN) 進行訓練和驗證（該函數在計算 loss 時已限定為 train_mask）。

---

## 重要觀察（程式碼層面）

1) knn_graph
   - `knn_graph` 使用 sklearn.NearestNeighbors，並在一開始將 tensor detach 到 CPU numpy：`x_np = x.detach().cpu().numpy()`。這是穩健且避免 GPU OOM 的做法。

2) precompute GNN（start / materialize hooks）
   - 兩個 hook 都會把 train/val/test 合併，建立全局圖，並以 MSE 自編碼重構 loss 訓練 GNN：loss = F.mse_loss(out, x)
   - 風險：這會讓 GNN 在訓練時看到 val/test 範例，應改為只在 train 節點上計算 loss（或僅 forward 全體節點、但 loss 只計算 train 節點）。

3) encoding/columnwise 的 joint-train
   - 若 `gnn_stage` 為 `encoding` 或 `columnwise`，GNN 會被加入 optimizer（即 joint-train）並在訓練中 update，這有利於 end-to-end 訓練但需注意比較基線時應保持一致配置。

4) decoding-stage GNN
   - `gnn_decoding_eval` 的實作已在 loss 和 metric 計算階段使用 `train_mask/val_mask/test_mask` 正確地分離訓練/驗證/測試（good）。

5) batch-level graph 建構
   - 在 encoding 的 batch 內使用 `knn_graph(latent_all, k=...)` 並依賴 `knn_graph` 的 detach 行為；這避免了 knn 建構與 autograd 混為一談，但如果 batch 很小（或 latent_all 的行數 < k+1）要注意 k 的選擇（code 中已用 `min(5, latent_all.shape[0]-1)` 作防護）。

---

## 建議修補（優先級排序）

1) （高）修正 precompute GNN 的訓練損失以避免資料滲漏
   - 在 `gnn_after_start_fn` 與 `gnn_after_materialize_fn` 的訓練迴圈把 loss 改成只在 train 節點計算，例如：
     - 在 `gnn_after_start_fn`：建立 train_mask = torch.zeros(N, dtype=torch.bool); train_mask[:n_train]=True；然後 loss = F.mse_loss(out[train_mask], x[train_mask])
     - 在 `gnn_after_materialize_fn`：同理使用 n_train, n_val, n_test 分割點並只用 train 區段計算 loss
   - 為何：保證 precompute GNN 的學習只由 train 集驅動，避免模型在隱性上利用 val/test 的分布資訊。

2) （中）新增 `gnn_mode` 與 `gnn_pretrain_train_only` config
   - `gnn_mode` ∈ {`'forward-only'`, `'joint-train'`}：決定是否把 GNN 參數加入 optimizer
   - `gnn_pretrain_train_only`: bool，若 True 則 precompute GNN 的 loss 只計算 train 節點
   - 這讓實驗更容易重現與配置比對。

3) （中）把 `knn_graph` 抽成共用 util
   - 建議把 `knn_graph` 抽到 `models/pytorch_frame/utils.py`（或 repo 公共 utils），讓其他 model mapping 重用，並可在該處加入選項：`max_edges`、`approximate`（FAISS）或 `random_seeded` 等。

4) （低）把 precompute GNN 訓練的 early-stopping 與 metrics 更詳細記錄到 `material_outputs`（例如 `material_outputs['gnn_info'] = {...}`），利於實驗追蹤。

5) （低）如果 batch-size 很小，考慮在 epoch 間隔（例如每 5 steps）才重新建圖，或使用 approximate knn，避免頻繁呼叫 sklearn 導致開銷。

---

## 小契約（Contract）
- Inputs: `train_df/val_df/test_df` 或 materialized `train_loader/val_loader/test_loader`，`config` 與 `dataset_results['info']['task_type']`。
- Outputs: 與現有 `main()` interface 相同；precompute hook 可回傳替換後的 DataFrame 或 DataLoader；decoding hook 回傳 best_val_metric, best_test_metric, gnn_early_stop_epochs。

---

## SubTab — 模型 pipeline（上游 → 下游）範例

範例設定：中型資料集，1500 筆、24 欄，train/val/test = 1050/225/225。

1) Start
   - 讀入 DataFrame；`gnn_after_start_fn` 可作離線特徵注入。

2) Materialize
   - `materialize_fn` 轉為 `TabularDataset` / DataLoader，做標準化與常數欄移除。

3) Encoding
   - `AEWrapper` 對子集輸入做 encoder，`subset_generator` 產生子集供 encoder 使用。

4) Column-wise interaction
   - `subset_generator` + `columnwise` 邏輯實現 contrastive / reconstruction 與子集聚合；若 `gnn_stage=='columnwise'`，會在 latent 或聚合表示上套用 GNN。

5) Decoding
   - 從聚合 latent 提取 embeddings，訓練下游分類器（Logistic/Linear），或使用 decoding GNN 做後處理。

PyTorch-Frame 對應：start → materialize → encoding → columnwise → decoding。

