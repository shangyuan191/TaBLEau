# T2G-Former ↔ PyTorch-Frame stage 對照報告

## 一、摘要結論（要點）

- 結論：T2G-Former 的內建圖機制（FR-Graph）屬於「encoding / backbone 內部的 columns-as-nodes」類型（等價於 PyTorch-Frame 的 encoding/columnwise 中把 columns 當作節點、並在 encoder/backbone 內 joint-train 的情況）。
- 換句話說：T2G-Former 並非 precompute（`start`/`materialize`）或 post-hoc decoder（`decoding`），而是把圖結合進每一層的注意力機制，於 encoder 層內即進行關係/拓樸學習與訊息聚合。

## 二、快速說明（行為對照）

- PyTorch-Frame 五個 stage（簡述）：
  - `start`：資料載入前/最前面，可做離線 row-level GNN（`gnn_after_start_fn`）。
  - `materialize`：`TensorFrame` 建立後，可在這做離線 row-level GNN（`gnn_after_materialize_fn`）。
  - `encoding`：encoder / backbone 的輸出或中間表示，可在這做 joint-train GNN（分 row-level 或 column-as-nodes）。
  - `columnwise`：針對 column-level 的交互或 decision-step，可插入 columnwise GNN（語意上可為 per-row columns-as-nodes）。
  - `decoding`：在輸出後做 post-hoc GNN（`gnn_decoding_eval`）。

- T2G-Former 的定位：`encoding` / `columnwise`（更精確是 encoder/backbone 內部的 column-as-nodes attention）。

## 三、技術證據（程式對應與關鍵片段）

- T2G-Former 關鍵實作檔案：
  - `/home/shangyuan/ModelComparison/t2g-former/bin/t2g_former.py`
    - `class Tokenizer`：把 numerical 與 categorical 欄位映成 token；有 `n_tokens` 屬性，代表 token 數（包含 CLS/readout token）。
    - `class MultiheadGEAttention`：FR-Graph integrated attention。重點：
      - 使用 `self.n_cols`（token/columns 數）作為圖上節點數量（columns-as-nodes）。
      - 計算 topology 分數 `top_score` 並用 `_get_topology` 產生 adjacency（`adj`），再轉成 `adj_mask` 當成 attention mask。
      - 最終以 `fr_graph = comp_func(weight_score + adj_mask, dim=-1)` 得到 graph-weighted attention，並用於 value 的聚合（在 forward 回傳 `x, fr_graph`）。
    - `class T2GFormer` 的 `forward`：
      - 每個 layer 的迴圈中呼叫 `layer['attention'](...)`（即 `MultiheadGEAttention`），並收集每層的 `fr_graphs`，顯示圖操作在每一層的 encoder 內被執行（joint-train）。

- 對照說明檔（PyTorch-Frame mapping）：
  - `/home/shangyuan/ModelComparison/TaBLEau/models/pytorch_frame/fttransformer_pytorch_frame_mapping.md`
  - 相關 mapping 檔（如 `tabnet_pytorch_frame_mapping.md`, `tabtransformer_pytorch_frame_mapping.md`）明確說明了 `start`/`materialize`/`encoding`/`columnwise`/`decoding` 的語意與程式對應，方便比對。

## 四、為何判定為 encoding / columns-as-nodes（要點）

- Token 層級：T2G-Former 的 `Tokenizer` 把每一個欄位變成 token（包含 CLS token），因此 attention 的「節點」自然對應欄位（columns-as-nodes），不是 row-level 的整體 embedding graph。
- Graph 結合方式：圖（`adj`）是 attention 運算的一部分（作為 attention mask/權重），且在每層 attention 都會重新計算或使用（joint-train）。這與在 `encoding` 內對 columns 做 per-row message passing 的語意一致。
- 與 PyTorch-Frame 的差別：
  - `start`/`materialize` 的 GNN 實作是離線/預計算（會把 embedding 回寫為新 column），而 T2G-Former 是線上 joint-train（forward 內的 attention）。
  - `decoding`-stage 的 GNN 是 post-hoc supervised training（收集所有 row embedding 再訓練 GNN），而 T2G-Former 並非此類。

## 五、實務影響與建議

- 若你要在 TaBLEau / PyTorch-Frame 中復現 T2G-Former 的行為：
  - 建議在 `encoding` 階段（也就是 backbone/transformer 層內）實作一個 attention-layer-level 的 GNN/graph-attention 模組（例如把 `MultiheadGEAttention` 移植為一個可插拔的 attention/gnn 模組），並在 forward 中把它加入到每層的 attention path，確保與 backbone joint-train（同一 optimizer）。
  - 若希望更精準模仿：在 `encoding` 的 layer 裡，tokenize 成 `(batch, n_tokens, d)` 後，以 `n_tokens`（等於欄位數 + CLS）為圖節點數，計算 topology + edge weights，並把所得 adjacency 當作 attention mask 使用，這與 T2G-Former 的 `MultiheadGEAttention` 機制一致。

- 注意事項：
  - column-as-nodes 的 per-row GNN（即每 row 建小圖）計算成本會很高，T2G-Former 的做法是把 columns 視為 token 並在 batch 中一起計算 attention，節省成本且能共享參數。
  - 若在 PyTorch-Frame 的 `encoding` 中採用 batch 層級的 GNN（node = row in batch），語意會變為 row-message-passing（不同於 T2G-Former 的 columns-as-nodes）；要避免混淆實作語意。

## 六、建議的下一步（可選，擇一）

- A. 幫你把 T2G-Former 的 `MultiheadGEAttention` 封裝成一個可插拔模組，然後在 TaBLEau 的某個 `pytorch_frame` model（例如 `fttransformer.py` 或 `tabtransformer.py`）中加入 `gnn_stage='encoding_internal'` 的實作範例（含 minimal code patch），以便直接比較與實驗。
- B. 幫你把 TaBLEau 中的 `encoding` hook 改為支援「columns-as-nodes」的 per-row graph（包含 safe 的 reshape/批次化範例），並給出 compute cost 與簡單 benchmark script。
- C. 只產出一個短的 patch 範例跟說明（README style），說明如何在 `fttransformer` 裡把 attention 換成 FR-Graph attention（不改動其它 train pipeline）。

## 七、引用（關鍵檔案）

- T2G-Former 原始實作：
  - `/home/shangyuan/ModelComparison/t2g-former/bin/t2g_former.py`（Tokenizer, MultiheadGEAttention, T2GFormer.forward）
- TaBLEau 的 PyTorch-Frame mapping 文件（stage 與 hook 說明）：
  - `/home/shangyuan/ModelComparison/TaBLEau/models/pytorch_frame/fttransformer_pytorch_frame_mapping.md`
  - 其他參考：`tabnet_pytorch_frame_mapping.md`, `tabtransformer_pytorch_frame_mapping.md`, 以及對應的 model 實作檔（例如 `fttransformer.py`, `tabtransformer.py`, `tabnet.py`）

## 八、簡短收尾

- 我已把主要的判定與程式證據列出，並提出三個可執行的下一步選項（A/B/C）。
- 若要我繼續，我可以直接在 repository 中套用你選擇的方案，並跑快速檢查或產生 patch（我會在每次做變動前再次用 todo-list 標註並報告進度）。
