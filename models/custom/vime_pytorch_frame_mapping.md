## VIME 與 PyTorch-Frame 五個 stage 對照說明

本檔案依據 `models/custom/vime.py` 的實作，將 VIME 的 pipeline 對應到 PyTorch-Frame 的五個 stage（start、materialize、encoding、columnwise、decoding），並說明程式中現有的 GNN hook、語意、風險與具體改進建議。

### 一、精要
- VIME 是一個 self-supervised pretext-style 模型：
  - Materialization：`materialize_fn` 將 raw DataFrame 轉為標準化的 numpy arrays（X_train/X_val/X_test）並傳回相關標準化參數；
  - Encoding：`vime_core_fn` 中使用 `VIMEEncoder`（MLP）產生表示 h、mask_logits 與 reconstruction，並以 reconstruction + mask 預訓練 encoder；
  - Columnwise interaction：VIME 沒有傳統的 column-conv；但在 `vime_core_fn` 的訓練迴圈內提供了在 batch-level 上對 encoder embeddings 做 forward-only GNN（`gnn_stage in ['encoding','columnwise']`）；
  - Decoding：`gnn_decoding_eval` 可在 encoder embeddings (Z_train/val/test) 上訓練 supervised GNN 作為 downstream evaluator；若不使用 decoding GNN，則使用 baseline 線性/Logistic/Ridge 評估。 

### 二、VIME 的 pipeline（程式如何處理 input）
1. start
   - `start_fn` 為 identity；若使用 `gnn_stage == 'start'`，會呼叫 `gnn_after_start_fn` 進行 row-level precompute（see below）。

2. materialize
   - `materialize_fn(train_df, val_df, test_df, ...)`：
     - 以 numpy arrays 回傳標準化後的 `X_train/X_val/X_test` 與 `y_*`、`mean/std` 等；
     - 這裡不會建立 Torch Dataset/Loader；若需要在 materialize 階段加入 GNN，使用 `gnn_after_materialize_arrays`（接受 arrays 並回傳同樣結構的 mat）。

3. encoding
   - `vime_core_fn(mat, config, task_type, gnn_stage)`：建立 `VIMEEncoder`（MLP）與訓練迴圈 `run_epoch`。
   - 在每個 mini-batch 裡，encoder 的 forward 回傳 `h, mask_logits, recon`，其中 `h` 為 batch-level embedding（shape [B, hidden])。
   - 如果 `gnn_stage in ['encoding', 'columnwise']` 且安裝了 PyG，程式會：
     - 用 `knn_graph(h, k)` 建批次內的 KNN edge_index（sklearn 實作，CPU）;
     - 動態建立一個 `SimpleGCN`（輸入為 h.size(1)）並套用於 h（h = gnn(h, edge_index)）；
     - 注意：這個 GNN 是 per-batch ephemeral（每 batch 都會新建 model），且僅在 forward 上使用（不以圖為監督訓練）。

4. columnwise
   - VIME 原本沒有 columnwise convs；`columnwise_fn` 在檔案中實作為 placeholder，實際上會把 materialize 後的 arrays concat 並用 `knn_graph` 建圖，再套用 `SimpleGCN`（離線 injection），等價於把 GNN 當作一個 materialize 之後的 transform。

5. decoding
   - `gnn_decoding_eval(Z_train, y_train, Z_val, y_val, Z_test, y_test, ...)`：
     - 將 Z 合併、建 KNN 圖、用 supervised loss 在圖上訓練 GNN（支援二元、多類或回歸），使用 train_mask 計算 loss 並在 val 上 early stop；
     - 這是最清晰的 post-hoc decoding GNN，用以量化 GNN 在 embeddings 空間對下游任務的貢獻。

### 三、程式中 GNN hook 與語意（簡表）

| PyTorch-Frame stage | 程式對應 (file / fn / 變數) | 語意 |
|---|---:|---|
| start | `gnn_after_start_fn(...)` | row-level precompute：concat train/val/test，建 KNN（sklearn）並訓練 GNN（loss 限制於 train_mask），輸出 embedding 回寫為數值欄位並回傳三個 DataFrame。此函式已使用 train/val mask 來避免 leakage（建議仍確認）。 |
| materialize | `materialize_fn(...)` + `gnn_after_materialize_arrays(...)` | materialize 回傳 arrays；`gnn_after_materialize_arrays` 在 arrays 上合併建圖並用 train_mask 訓練 GNN，最後把 mat 中的 X_* 置換為 GNN output（offline feature injection）。 |
| encoding | `vime_core_fn` 的 `run_epoch`（mini-batch）中的 forward-only GNN | batch-level row-message-passing：將 batch 內的 sample embeddings (`h`) 視為 nodes，建 KNN（knn_graph）並套用 GNN（ephemeral, forward-only）。通常這不使用 labels，而是作為模型訓練中的一個非監督前處理/feature transform。 |
| columnwise | `columnwise_fn` placeholder / `vime_core_fn` supports `gnn_stage=='columnwise'` | VIME 沒有內建 columnwise conv；所謂 columnwise hook 在 VIME context 多為「對整個表格（row embeddings 或 arrays）做圖操作」的別名，而非每 row 的 columns-as-nodes。 |
| decoding | `gnn_decoding_eval(...)` | post-hoc supervised GNN：在 encoder embeddings 上訓練（loss 只在 train_mask），早停於 val，並回傳 val/test 指標。這是評估 GNN downstream 貢獻的安全方式。 |

### 四、已觀察到的良好實作與風險
- 已觀察到的正面：
  - `knn_graph` 使用 sklearn 的 CPU KNN（避免 GPU 上 O(N^2) 的 torch.cdist OOM）；
  - `gnn_after_start_fn` 與 `gnn_after_materialize_arrays` 會使用 train/val mask 進行監督式早停，loss 也限定於 train_mask（減少 leakage 風險）。

- 風險與可改進之處：
  1) ephemeral per-batch GNN in `run_epoch`:
     - 每個 mini-batch 都會建立新的 `SimpleGCN` 與呼叫 `knn_graph(h,k)`；這會導致大量的模型建構/GC 與額外開銷。建議：移動 GNN 建構到 `vime_core_fn` 外層，並在訓練/評估階段重複使用（除非你刻意想做 per-batch 隨機化）。
  2) KNN / edges cost:
     - knn_graph 用 sklearn 已較安全，但仍要注意當 batch 大時的 CPU 成本；可增加 `config` 選項限制 `max_batch_for_gnn` 或使用 sampling（subsample batch nodes）或 top-k sparse edges。 
  3) 如果希望真正做 column-level GNN（columns-as-nodes），VIME 的現行設計主要是 row-level embedding；要做 column-level GNN 需在 materialize/encoding 上保留 per-column representation，並針對每 row 建小圖（num_cols 節點）——通常只有在 num_cols 很小時才可行。 

### 五、具體改進建議（可直接 patch）
1. Reuse GNN instance: 把 per-batch `gnn = SimpleGCN(...)` 的建立移到 `vime_core_fn` 外層（根據 `gnn_stage` 決定是否創建），避免每 batch 重建模型與 optimizer。如果你 want to fine-tune GNN jointly, add its params to optimizer; if forward-only, keep it fixed and call `.eval()`。 
2. Add `max_batch_for_gnn` or `gnn_subsample` config: 在 `run_epoch` 中當 batch size 過大時只抽樣 S ≤ B 的 nodes 來建 KNN 與 GNN。 
3. Make ephemeral vs joint-train explicit: 提供 `gnn_mode` config ∈ {"forward-only", "joint-train"}，當為 forward-only 時不要把 GNN 的參數加入 encoder optimizer；當為 joint-train 時則需要長期存在的 GNN module 並加入 optimizer。 
4. Document and log compute cost: 在每 epoch 記錄 GNN-precompute time、KNN time 與 total training time，方便比較插入點 trade-offs。 

### 六、快速小結
- VIME 的 GNN 支援已經包含合理的 safeguards（sklearn KNN、train/val mask 在 precompute training 中），但有幾個低成本改進能讓實驗更穩定與高效：
  - 將 per-batch ephemeral GNN 改為可選的常駐 GNN（或明確標示 forward-only）；
  - 在 batch-level GNN 中採用 subsampling 或 top-k 限制以控管 CPU 負載；
  - 如果要做 columns-as-nodes 的 column-level GNN，需要不同的資料形狀與成本估算（非現行預設）。

檔案位置：`models/pytorch_frame/vime_pytorch_frame_mapping.md`
## VIME — 模型 pipeline（上游 → 下游）範例

範例設定：半監督任務，2000 筆、20 欄，train/val/test = 1400/300/300。

1) Start
   - 讀入 DataFrame；`gnn_after_start_fn` 可在此對整表做 precompute GNN（注意 train-only loss）。

2) Materialize
   - `materialize_fn` 轉為 normalized numpy arrays（X_train/X_val/X_test）並回傳標準化參數。

3) Encoding (pretext)
   - `vime_core_fn` 中的 `VIMEEncoder` 執行 self-supervised pretraining（mask/reconstruction），產生 batch-level embedding `h`。

4) Column-wise / GNN integration
   - VIME 本身沒有 columnwise conv；若 `gnn_stage in ['encoding','columnwise']`，程式選擇對 batch-level embedding `h` 做 forward-only GNN（ephemeral 或長期 GNN 視設定而定）。

5) Decoding
   - `gnn_decoding_eval` 或 baseline classifier/regressor 在 embeddings 上評估下游表現；decoding GNN 在 train_mask 上訓練並 early-stop。

PyTorch-Frame 對應：start → materialize → encoding → columnwise (optional) → decoding。