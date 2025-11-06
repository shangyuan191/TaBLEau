# TabGNN ↔ PyTorch-Frame stage 對照報告

## 一、摘要結論（要點）

- 結論：TabGNN 是以圖為核心的模型，GNN 是模型的主體，會在 forward 流程中對已建構的圖執行 message-passing 與 readout；以 PyTorch-Frame 的五個 stage 類比，最接近的是「encoding（即在模型 forward 中 joint-train 的 GNN）」。
- 補充說明：TabGNN 不只是把 GNN 當作一個額外 hook（如 precompute 的 `gnn_after_start_fn` 或 post-hoc 的 `gnn_decoding_eval`），而是以圖為資料結構的主要推理流程（node initializers -> gnn_forward -> readout -> fcout），因此語意和實作都屬於在 encoding/backbone 階段直接運行 GNN 的類型。

## 二、快速說明（行為對照）

- PyTorch-Frame 五個 stage（簡述）：
  - `start`：資料載入前/最前面，可做離線 row-level GNN（`gnn_after_start_fn`）。
  - `materialize`：`TensorFrame` 建立後，可在這做離線 row-level GNN（`gnn_after_materialize_fn`）。
  - `encoding`：encoder / backbone 的輸出或中間表示，可在這做 joint-train GNN（分 row-level 或 column-as-nodes）。
  - `columnwise`：針對 column-level 的交互或 decision-step，可插入 columnwise GNN（語意上可為 per-row columns-as-nodes）。
  - `decoding`：在輸出後做 post-hoc GNN（`gnn_decoding_eval`）。

- TabGNN 的定位：`encoding`（更精確：模型的核心 GNN 在 forward 時針對批次化的圖執行 message-passing，並以 readout 產生預測，屬 joint-train 的 GNN）。

## 三、技術證據（程式對應與關鍵片段）

- 主要程式位置：
  - `/home/shangyuan/ModelComparison/TabGNN/models/GNN/GNNModelBase.py`
    - `class GNNModelBase`：
      - `__init__` 建立 `node_initializers`（使用 tabular model 對每種 node_type 做 feature encoding）與 `readout`、`fcout`。
      - `init_batch(self, bdgl, b_features)`：在每個 batch 開始時把原始 features 經過 `node_initializers` 編碼成節點初始向量（hidden_dim），並放入圖 `bdgl.ndata['h']`。
      - `forward(self, input)`：呼叫 `self.init_batch(...)` 後，接著呼叫 `self.gnn_forward(g, fz_embedding, main_node_ids)`，表明 GNN 的 message-passing 與 readout 就在模型的 forward 期間執行（joint-train）。
    - 檔案中還定義了 readout 與後處理 MLP（`fcout`），這是標準的 GNN pipeline（init encoders -> GNN -> readout -> head）。
  - `/home/shangyuan/ModelComparison/TabGNN/models/GNN/GCN.py`, `/models/GNN/GAT.py`, `/models/GNN/HAN.py` 等：
    - 這些檔案包含不同的 GNN layer 實作（GCN, GAT, HAN, ERGAT 等），會在 `gnn_forward` 中被呼叫（或被具體 model subclass 使用）。
  - `/home/shangyuan/ModelComparison/TabGNN/data/DatabaseDataset.py`
    - 此資料集負責把資料（database / relational tabular）轉成 preprocessed 的 datapoint 與 batched graph（lmdb 存取），表示模型的輸入本身已是圖（nodes + edges），非單純 row-by-row 的 TensorFrame。這與 PyTorch-Frame 的 row-level pipeline 不同：TabGNN 是圖為中心的整體模型。

- 實作語意說明：
  - `node_initializers` 的作用相當於把每個 node 的原始 tabular features 編碼成向量（這一步相當於 PyTorch-Frame 的 materialize/encoding 中的 feature encoding），但它是 per-node 的編碼並被直接放入圖中。接著 `gnn_forward` 在該圖上做 message-passing 並產出 node-level或圖級嵌入，最後做 readout -> fcout 得到預測。
  - 因為 GNN 是在 forward 中直接運作，且與 readout、fcout、loss joint-train（loss 在 GNN 輸出上計算），最相近的是 PyTorch-Frame 的 `encoding`（joint-train GNN）而非 precompute 或 post-hoc。

## 四、為何判定為 encoding（要點）

- 執行時機：GNN 在模型 forward 裡被執行（`forward -> init_batch -> gnn_forward`），是 online/joint-train 而非離線預計算。
- 輸入語意：輸入是已建構的圖（nodes/edges），節點的 feature 是由 tabular 初始器產生，這代表 GNN 是模型的核心而非附加器。
- 搭配訓練策略：GNN 與 readout/fcout 等一起接受 optimizer 更新與 loss，這符合 `encoding` 階段的 joint-train GNN 特性。

## 五、與 PyTorch-Frame hook 的差異（注意）

- PyTorch-Frame 的 `encoding` hook 多數示例是為了在現有的 transformer/ML backbone 裡加一個 GNN（例如把 encoder 的中間表示拿來做 batch-level GNN）；而 TabGNN 從設計上就是一個圖模型，輸入本身是圖，GNN 是主體。兩者語意相近（都在 forward 內 joint-train），但 TabGNN 更接近「graph-native」的架構。
- PyTorch-Frame 的 `start` / `materialize` 多為離線或預處理型 GNN（把 embedding 回寫為 columns），而 TabGNN 不做這種操作（沒有回寫 embedding 到 DataFrame），因此不屬於 `start`/`materialize`。
- PyTorch-Frame 的 `decoding` GNN 則是 post-hoc supervised GNN（以 encoder 的 pooled embeddings 建圖再訓練），而 TabGNN 的 GNN 則是 primary supervised模型的一部分（不是 post-hoc）。

## 六、實務建議（若要在 TaBLEau 中比較/整合 TabGNN）

- 如果你要在 TaBLEau 的比較框架中把 TabGNN 與其它 PyTorch-Frame 實作做公平比較：
  - 把 TabGNN 當成一個完整的 model（非 hook）放入比較清單，設計好資料 pipeline讓 TabGNN 的輸入格式（graph）能從相同的原始資料產生（例如建圖策略、knn 還是關聯表）。
  - 在比較 GNN 插入點時，將 TabGNN 的行為歸類為 `encoding` 類（joint-train），並與 PyTorch-Frame 中在 `encoding` stage 插入的 GNN（例如 `gnn_stage=='encoding'`）做對比（同 seed、相同訓練步驟、相近參數量）。
- 若你想把 TabGNN 的 node-initializer（tabular encoder）替換為 TaBLEau 的某個 `encoder(tf)` 實作，請確認輸出維度與資料結構（node types mapping）一致，然後在 TabGNN 的 `init_batch` 中替換初始化器即可。

## 七、關鍵檔案清單（參考）

- `/home/shangyuan/ModelComparison/TabGNN/models/GNN/GNNModelBase.py`（核心 pipeline：init_batch -> gnn_forward）
- `/home/shangyuan/ModelComparison/TabGNN/models/GNN/GCN.py`, `/GAT.py`, `/HAN.py`（各種 GNN layer 實作）
- `/home/shangyuan/ModelComparison/TabGNN/data/DatabaseDataset.py`（圖資料的讀取與 preprocessed datapoint）

## 八、下一步（擇一）

- A. 幫你在 TaBLEau 的某個 `pytorch_frame` model 中新增一個 `gnn_stage=='encoding_graph_native'` 的 adapter，能把 TabGNN 的圖式 pipeline 包裝為一個可比較的 model（含資料建圖/loader 範例）。
- B. 幫你把 TabGNN 的 `node_initializers` 換成 TaBLEau 的 encoder（範例 patch），使得不同模型共享相同的 feature encoding。 
- C. 產生簡短的實驗設計文件（README）說明如何在 TaBLEau 中公平比較 TabGNN 與其它 models（包含建圖策略、超參對齊、訓練/early-stop 規則）。

如果你想要我把這份報告寫入檔案或要我執行上述某個下一步，告訴我選項（A/B/C）或要我直接 commit/PR。