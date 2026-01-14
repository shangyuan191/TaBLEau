# 自帶 GNN（Self-contained）Baseline 全面比較：TabGNN / T2G-Former / DGM / LAN-GNN / IDGL-GNN / GLCN / LDS-GNN

## 摘要（Abstract）

本文檔以 SAGE 的研究問題「Where to Graph-ify tabular deep learning？」為主軸，將 TaBLEau 中 7 個 **self-contained（自帶 GNN / 圖式交互）** baseline：TabGNN、T2G-Former、DGM、LAN-GNN、IDGL-GNN、GLCN、LDS-GNN，統一到同一個分析框架，回答三件事：
1) 這些模型的「圖」到底是什麼（row graph vs feature/token graph）？
2) 從 DataFrame 到訓練/評估的資料管線如何落地（transductive vs inductive；mask/indices 怎麼用）？
3) 它們內部的「圖化/消息傳遞/交互」可類比到 PyTorch-Frame 五階段中的哪個 stage，從而與 SAGE 的 GNN 注入實驗形成可比對照。

## 本文檔的貢獻（What this document gives you）

- 提供跨模型一致的座標系：PyTorch-Frame 五階段（start/materialize/encoding/columnwise/decoding）。
- 提供一致的符號化描述：把「節點是 row 還是 feature」明確化，避免比較時偷換概念。
- 提供工程視角的可比性備註：哪些模型在 TaBLEau 內已完整落地、哪些路徑是 dense/sparse fallback（以及大 N 時的降級策略）。
- 產出可直接寫進論文的內容：taxononomy、比較表、複雜度與威脅效度（Threats to Validity）、以及可落地設計指引（Actionable Guidelines）。

延伸閱讀（更細節的單模型剖析）：
- DGM stage 分析：[DGM_GNN_Stage_Analysis.md](DGM_GNN_Stage_Analysis.md)
- LAN-GNN 架構分析：[LAN_GNN_Architecture_Analysis.md](LAN_GNN_Architecture_Analysis.md)
- IDGL-GNN 架構分析：[IDGL_GNN_Architecture_Analysis.md](IDGL_GNN_Architecture_Analysis.md)

---

## 符號與問題設定（Problem Setup & Notation）

令表格資料在一次實驗中形成三個 split：$\mathcal{D}_{tr},\mathcal{D}_{va},\mathcal{D}_{te}$。

### Row graph（樣本圖）設定
- 節點：每筆樣本（row）是一個節點 $v_i$，特徵向量為 $x_i\in\mathbb{R}^D$。
- 圖：$G=(V,E)$，$|V|=N$。
- 核心操作：在圖上做消息傳遞 $H^{(\ell+1)}=\mathrm{MP}^{(\ell)}(H^{(\ell)},E)$。
- 訓練型態：常見是 transductive（把三個 split 合併成一張大圖，梯度只對 train mask 回傳）。

### Feature/token graph（特徵圖）設定
- 節點：每個欄位（或 tokenizer 後的 token）是節點（常再加一個 readout/CLS token）。
- 圖：描述欄位/特徵之間的關係，核心是 feature interaction。
- 訓練型態：多為 inductive（mini-batch），因為圖大小與欄位數相關而非樣本數。

> 重要：SAGE 的「Where to Graph-ify」在 row-graph 與 feature-graph 的語意不同。
> - row-graph：問的是「在哪個 stage 做 row-level message passing/graph learning 最有效？」
> - feature-graph：問的是「是否以圖化方式重寫/約束 columnwise interaction（類 attention）？」

## 0. 共同前提與名詞釐清

### 0.1 什麼叫 self-contained baseline？
在 TaBLEau 內，`models/comparison/*.py` 這類模型被視為 **comparison**：
- 不參與 PyTorch-Frame 五階段的 `gnn_stage` 注入實驗
- 模型本身內部自帶（或主幹依賴）圖結構/消息傳遞/圖式交互

### 0.2 兩種「圖化」其實是不同物種
很多混淆來自於：大家都叫 Graph，但圖的「節點」可以完全不同。

- **Row graph（樣本圖）**：節點=資料的每一筆樣本（row）。
  - 代表模型：TabGNN / DGM / LAN-GNN / IDGL-GNN
  - 關係建模：樣本之間的相似度、鄰接、消息傳遞

- **Feature graph（特徵圖 / Token graph）**：節點=特徵欄位（或其 token），圖刻畫「欄位之間」的關係。
  - 代表模型：T2G-Former（FR-Graph）
  - 關係建模：特徵交互（類似 attention，但用 learnable topology 來約束/強化）

> 因此，當你在 SAGE 談「GNN 插入 stage」時，必須先說清楚：你插入的是 **row-level message passing**，還是 **feature-level interaction**。

---

## 1. PyTorch-Frame 五階段（用來做對照的共同座標系）

| Stage | 名稱 | 典型操作（抽象） |
|---|---|---|
| start | Start | 缺失值處理、切分策略、隨機種子 |
| materialize | Materialize | DataFrame → tensor/np array |
| encoding | Encoding | 數值標準化、類別編碼/embedding |
| columnwise | Columnwise | 列間/欄間交互（Attention / GNN message passing） |
| decoding | Decoding | MLP readout / logits / regression head |

---

## 2. 一眼看懂的比較總表（重點維度）

下表刻意把「圖的 domain」拉出來，避免把 row-graph 與 feature-graph 直接放在同一個天平上比較（那會導致錯誤結論）。

| 模型 | Graph domain | 圖如何來（概念） | 圖是否可學習 | Interaction / MP 核心算子 | 訓練型態 | 實作狀態（TaBLEau） | 主要風險 |
|---|---|---|---|---|---|---|---|
| TabGNN（本專案 wrapper） | row graph | 固定 kNN（sklearn） | 否 | GCNConv（PyG）或 Linear fallback | transductive | 已落地 | 這是「kNN+GCN baseline」，不等於所有 TabGNN 文獻版本 |
| T2G-Former（本專案 wrapper） | feature/token graph | FR-Graph：learnable topology + threshold | 是（topology+weights） | FR-Graph attention（帶 topology mask 的 softmax） | inductive（mini-batch） | 已落地 | 與 row-graph 完全不同；比較時需先定義可比問題 |
| DGM | row graph | 每層動態 kNN（Gumbel/KeOps） | 是（可微分採樣） | (learn edges) + GCN/GAT/EdgeConv | transductive | 已落地（含降級路徑） | 大 N 計算重；KeOps 依賴；動態邊穩定性 |
| LAN-GNN | row graph | 每層 learnable adjacency（DGG/top-k） | 是（straight-through/top-k） | Dense GCN（$\tilde{A}XW$）或 PyG fallback | transductive | 已落地 | 小 N dense adjacency；大 N 需走稀疏 fallback |
| IDGL-GNN | row graph | GraphLearner attention/topk（小 N）；大 N 固定 kNN | 小 N 是 / 大 N 否 | DenseGCN 或 PyG GCNConv | transductive | 已落地 | dense adjacency 記憶體；IDGL 依賴可用性 |
| GLCN | row graph | 可學習稀疏圖（structure learning + graph regularization） | 是（learned adjacency） | GCN + 可學習鄰接（TF1） | transductive | 已落地（TF1 subprocess） | 大 N 容易踩到 dense $N\times N$；需確保 loss/regularizer 為 sparse-safe |
| LDS-GNN | row graph | 邊/結構視為超參數（離散結構學習 + 超梯度） | 是（結構作為超參數） | GCN + FAR-HO/implicit gradient（TF1） | transductive | 已落地（TF1 subprocess） | 訓練非常慢；依賴鏈複雜；超參數/收斂敏感 |

### 計算複雜度與記憶體（paper-style 粗估）

令 $N$ 為樣本數、$D$ 為輸入維度、$H$ 為 hidden dim、$k$ 為近鄰數、$L$ 為 message passing 層數、$F$ 為特徵 token 數。

| 類型 | 主要瓶頸 | 時間複雜度（粗估） | 記憶體（粗估） |
|---|---|---|---|
| Row graph + dense adjacency | 形成/存取 $N\times N$ adjacency | $\mathcal{O}(N^2H)$（常見） | $\mathcal{O}(N^2)$ |
| Row graph + sparse kNN | 近鄰搜尋與稀疏 MP | 建圖約 $\mathcal{O}(N\log N)$ 或 $\mathcal{O}(NkD)$（依實作）；MP 約 $\mathcal{O}(L\cdot NkH)$ | $\mathcal{O}(Nk)$ |
| Feature/token graph | token 交互（類 attention） | 常見 $\mathcal{O}(L\cdot F^2 d)$ 或變體 | $\mathcal{O}(F^2)$（但 F 通常遠小於 N） |

---

## 3. Stage mapping：把 self-contained 的內部流程映射回五階段

> 注意：這是「類比」，不是說它們真的跑 PyTorch-Frame。

| 模型 | start/materialize | encoding | columnwise（關係建模的主戰場） | decoding |
|---|---|---|---|---|
| TabGNN | 合併 split + 建索引 | one-hot（get_dummies）、缺失填補 | kNN row graph + GCN message passing | linear head 输出 |
| T2G-Former | 數值/類別前處理 + batch loader | Tokenizer（CLS + num/emb） | FR-Graph attention（feature graph）+ Transformer blocks | CLS readout + head |
| DGM | 轉導式 whole-graph | embed_f / pre_fc | 動態圖構建 + message passing（每層） | MLP head |
| LAN-GNN | 轉導式 whole-graph + scaler/label | 投影層（Linear→ReLU） | learnable adjacency + dense GCN（每層） | MLP head |
| IDGL-GNN | 轉導式 whole-graph + scaler/label | StandardScaler + label one-hot | GraphLearner adjacency + DenseGCN（或固定 kNN + SparseGCN） | logits out_dim |
| GLCN | 轉導式 whole-graph + mask | 特徵標準化/投影（依 wrapper） | learned adjacency + GCN + graph regularization | logits/regression head |
| LDS-GNN | 轉導式 whole-graph + mask | StandardScaler / one-hot label（依 wrapper） | 結構學習（離散/超參數）+ GCN | logits/regression head |

### Stage mapping 的方法論（Methodology for mapping）

本研究使用以下準則把 self-contained baseline 映射回五階段：
1. **encoding**：只要是「把原始欄位值轉成可供模型交互的表徵」都算（例如 StandardScaler、one-hot、tokenizer embedding）。
2. **columnwise**：只要是「顯式建模關係/交互」都算（attention、message passing、learned adjacency、dynamic kNN）。
3. **decoding**：最後把表徵轉成 logits/回歸值的 readout/head。
4. **start/materialize**：以 TaBLEau wrapper 的資料處理語義為準（split 合併、mask、DataFrame→np/tensor）。

> 特別提醒：T2G-Former 的 columnwise 是 **feature interaction**；而其他 row-graph baselines（TabGNN、DGM、LAN-GNN、IDGL-GNN、GLCN、LDS-GNN）的 columnwise 是 **row-level message passing**。兩者共享 stage 名字，但不是同一個 object。

**在本專案的 7 個 self-contained baselines 中，分類如下**：
- **Row-graph baselines（row-level message passing / graph learning）**：TabGNN、DGM、LAN-GNN、IDGL-GNN、GLCN、LDS-GNN
- **Feature/token-graph baseline（feature interaction）**：T2G-Former

因此，當你在文件或論文中寫「columnwise 階段最有效」時，請先交代你指的是：
- row-level 的 message passing（樣本圖），還是
- feature interaction（特徵/欄位圖）。

---

## 4. 模型逐一深挖（架構 + 管線 + GNN 位置）

### 4.1 TabGNN（TaBLEau 內的 `tabgnn.py` wrapper）

**程式位置**：`models/comparison/tabgnn.py`

#### 架構（就本 wrapper 而言）
- 先把 `train/val/test` 做一致的 one-hot（`pd.get_dummies`）
- 把所有樣本堆起來建立 kNN row graph（`NearestNeighbors`）
- 用 `SimpleGCN`：
  - `NodeInitializer: Linear(D→H)+ReLU+Dropout`
  - `n_layers` 層 `GCNConv(H→H)`（若 PyG 不可用就退化成 Linear）
  - `head: Linear(H→out)`

#### 資料管線
1. `_preprocess_tables`：
   - `target` 欄（或最後一欄）當 label
   - `get_dummies(dummy_na=True)`，避免 split 類別不一致
   - 缺失填 0
2. `build_knn_graph(X_all, k)`：建立 `edge_index`
3. `train_node_classification`：以 train_idx 回傳 loss，val 做 early stopping

#### stage 類比
- encoding：one-hot + 缺失處理
- columnwise：kNN graph + GCNConv
- decoding：linear head

> 備註：此 wrapper 更像「kNN+GCN baseline」。如果你要寫論文時主張“TabGNN 文獻模型”，建議在文中標註：**此處採用簡化版 GCN row-graph baseline**，避免誤解。

---

### 4.2 T2G-Former（本專案 `t2gformer.py` wrapper）

**上游位置**：`ModelComparison/t2g-former/`
- 核心模型：`bin/t2g_former.py`（AAAI 2023 oral）

#### 這個模型的「圖」是什麼？
- 節點不是 row，而是 **feature tokens（欄位 token）**。
- T2G-Former 的核心是 FR-Graph（Feature Relation Graph）：
  - **topology（A）**：由「column embeddings」經內積得到 topology score，再經 `sigmoid + threshold` 轉成 0/1 邊（straight-through）
  - **edge weights（Gw）**：由 token embedding 經多頭投影 + relation embedding（對角）計算
  - **組裝**：把 topology 轉成 attention mask 加到 weight score 上，再 softmax 得到 `fr_graph`

#### 架構（高度濃縮）
```
(x_num, x_cat)
  ↓ Tokenizer
tokens: [B, n_tokens, d_token]   (包含 [CLS] token)
  ↓ (n_layers 次)
FR-Graph Attention + FFN + Residual
  ↓ 只保留 CLS
head: Linear(d_token→d_out)
```

T2G-Former 的 MultiheadGEAttention（FR-Graph integrated attention）在每層會產生 `fr_graph`：
- 形狀大致是 `[B, heads, n_head_nodes, n_cols]`
- 可用來視覺化 feature relation 與 readout collection

#### stage 類比（對 PyTorch-Frame）
- encoding：Tokenizer（數值投影 + 類別 embedding + CLS）
- columnwise：FR-Graph attention（本質是「欄位之間交互」）
- decoding：CLS readout + head

#### 與 row-graph baseline 的關鍵差異
- row-graph：在樣本之間傳遞訊息，偏「半監督/少樣本」常見的 transductive 設定
- feature-graph：在欄位之間建模交互，偏 tabular transformer 的 feature interaction

#### 重要現況（對 TaBLEau）
- TaBLEau 的 `models/comparison/t2gformer.py` 已接上 split 訓練/評估流程（可在 116 datasets 的兩種切分下產生 summary_results）。
- 由於其 graph domain 為 feature/token graph，與 row-graph baselines 的 inductive/transductive 假設不同，寫論文時建議在比較段落先明確聲明「可比的任務定義」。

---

### 4.3 DGM（Differentiable Graph Module）

詳見：[DGM_GNN_Stage_Analysis.md](DGM_GNN_Stage_Analysis.md)

#### 核心一句話
- 每層：`embed_f`（特徵編碼）→ `Gumbel/KeOps kNN`（動態建圖）→ `GNN message passing`
- 並在多層迭代中反覆「重建圖」

#### stage 類比
- encoding：`pre_fc` / `embed_f`
- columnwise：動態建圖 + message passing
- decoding：final MLP

---

### 4.4 LAN-GNN（Learnable Adaptive Neighborhood）

詳見：[LAN_GNN_Architecture_Analysis.md](LAN_GNN_Architecture_Analysis.md)

#### 核心一句話
- 每層：先投影到 hidden，再用（DGG 或 top-k）產生 dense adjacency，做 dense GCN message passing。
- 大 N 時避免 dense adjacency，改走 sklearn kNN + PyG。

#### stage 類比
- encoding：投影層（Linear→ReLU）
- columnwise：learned adjacency + dense GCN
- decoding：MLP head

---

### 4.5 IDGL-GNN（GraphLearner adjacency + GCN）

詳見：[IDGL_GNN_Architecture_Analysis.md](IDGL_GNN_Architecture_Analysis.md)

#### 核心一句話
- 小 N：GraphLearner（multi-perspective attention similarity）→ top-k adjacency → DenseGCN。
- 大 N：固定 kNN（sklearn）→ SparseGCN（PyG）。

#### stage 類比
- encoding：StandardScaler / label encoding
- columnwise：learned adjacency + GCN（或 sparse kNN + GCN）

---

### 4.6 GLCN（Graph Learning Convolutional Network）

**程式位置**：`models/comparison/glcn.py`

#### 核心一句話
- 透過可學習的鄰接（sparse structure learning）與圖正則項，把「建圖 + message passing」合成一個 end-to-end 的 row-graph baseline（TF1）。

#### stage 類比
- encoding：特徵前處理/投影
- columnwise：learned adjacency + GCN + graph regularization
- decoding：head 輸出

---

### 4.7 LDS-GNN（Learning Discrete Structures）

**程式位置**：`models/comparison/lds_gnn.py`

#### 核心一句話
- 把圖結構（離散的鄰接/選邊）當成可優化目標的一部分，透過超梯度/implicit gradient 做結構學習的 row-graph baseline（TF1）。

#### stage 類比
- encoding：特徵前處理
- columnwise：結構學習（離散/超參數）+ GCN
- decoding：head 輸出

---

## 5. 討論：對 SAGE 的可落地洞察（Discussion & Actionable Guidelines）

### 6.1 你其實在比較兩個問題，而不是一個

在 SAGE 的框架下，"Where to Graph-ify" 至少有兩個互相纏繞但不相同的研究問題：
1. **Row-level Graph-ify（樣本圖）**：把樣本變成節點，靠相似度/近鄰形成圖，再做 message passing。
2. **Column-level Graph-ify（特徵圖）**：把欄位/特徵 token 變成節點，在 feature interaction 層面引入 topology/graph inductive bias。

它們在計算型態上天生不同：前者複雜度主要受 $N$ 影響；後者主要受 $F$ 影響。

### 6.2 實驗設計指引（可直接寫進論文）

若你的目標是「用 self-contained baseline 幫助解釋注入式結果」：
- 建議把 row-graph baseline（TabGNN/DGM/LAN/IDGL）作為同一組對照，用來回答：注入式 GNN 的最佳 stage 是否能逼近/超過從頭圖化（row graph）的效果。
- 將 T2G-Former 視為另一組（feature-graph）對照，用來回答：若把 columnwise interaction 重寫成 graph-constrained attention，是否與注入式 columnwise GNN 的結論一致。

若你的目標是「可落地建議」：
- $N$ 大（大型資料集）時，**先避免 dense adjacency**；以 sparse kNN（或近似）為預設，再探索 learned adjacency。
- few-shot 情境中，row-graph 方法常常更「語意對齊」：因為 message passing 本質上是 label smoothing/structure propagation 的一種歸納偏置（但需避免 leakage 的誤解，見下節）。

## 7. Threats to Validity（威脅效度與注意事項）

### 7.1 Transductive 設定的解讀風險
Row-graph baseline 多採用 transductive（train/val/test 合併成一張圖），這會引發常見質疑：
- 這是否等同於 test leakage？

需要在論文中清楚說明：
- 特徵 $X$ 在 transductive 中是可見的；但 loss/梯度只對 train mask 回傳。
- 這對應於經典半監督圖學習的設定，與「用 test label 訓練」不同。

### 7.2 Baseline 的「同名不同物」
例如 TabGNN：文獻/實作版本非常多。TaBLEau 內的 `tabgnn.py` wrapper 更接近 "kNN+GCN"。
若論文想主張引用某篇 TabGNN 論文，需要在方法章節說清楚你用的是哪個版本、差異在哪。

### 7.3 T2G-Former 的可比性
T2G-Former 是 feature graph；若你把它與 row-graph baseline 的表現放在同一張排名表，需要明確交代：
- 你在比較的任務是「tabular prediction」的同一套 split/metric
- 但 inductive bias 與計算條件不同（$F$ vs $N$），因此結論應聚焦在「graph-ify 的位置/語意」而非純粹宣稱哪個比較好。

