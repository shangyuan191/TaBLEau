# IDGL-GNN：可學習圖構建的表格預測 Baseline（TaBLEau comparison）

> 本文檔是 **IDGL-GNN**（`models/comparison/idgl_gnn.py`）在 TaBLEau/SAGE 研究脈絡下的「架構 + 資料管線 + PyTorch-Frame stage 類比」說明。
>
> 參考寫法與對齊方式：見 [DGM_GNN_Stage_Analysis.md](DGM_GNN_Stage_Analysis.md)。

---

## 目錄
1. [模型概述](#模型概述)
2. [整體設計：兩種運行模式](#整體設計兩種運行模式)
3. [資料管線（TaBLEau wrapper）](#資料管線tableau-wrapper)
4. [模型架構細節](#模型架構細節)
5. [訓練與評估流程（含早停）](#訓練與評估流程含早停)
6. [PyTorch-Frame 五階段對應（Stage mapping）](#pytorch-frame-五階段對應stage-mapping)
7. [與 DGM / LAN-GNN 的定位比較](#與-dgm--lan-gnn-的定位比較)
8. [工程與穩定性細節（大 N fallback）](#工程與穩定性細節大-n-fallback)

---

## 模型概述

### 基本資訊
- **名稱（本專案）**：IDGL-GNN
- **定位**：self-contained（自帶 GNN）comparison baseline（不走外部 `gnn_stage` 注入）
- **實作位置**：[models/comparison/idgl_gnn.py](../models/comparison/idgl_gnn.py)
- **靈感來源**：IDGL（Iterative Deep Graph Learning）「可學習圖構建」的思想
  - 本 repo 內部使用的 GraphLearner 來自：`ModelComparison/IDGL/src/core/layers/graphlearn.py`

### 核心想法（在表格資料上的改寫）
對表格資料的每一筆樣本（row）視為一個節點，建立 **row-graph**（樣本圖）：
1. 先把所有 split（train/val/test）合併成一張圖（**transductive**）。
2. 在小樣本數 $N$ 的情況下，使用 GraphLearner 直接輸出 **dense 的可學習鄰接**（近似 learned kNN）。
3. 在大樣本數 $N$ 的情況下，避免 $N\times N$ 記憶體，改用 sklearn 在 CPU 建 **固定 kNN edge_index**（稀疏圖），再用 PyG 的 `GCNConv` 做消息傳遞。

這個 baseline 的研究用途是：把「圖構建」與「GNN 消息傳遞」整合成一個自洽模型，作為 SAGE 的 **外部對照組**，用來回答：
- 「如果直接用圖模型從頭做表格預測（而非插入式 GNN），表現會如何？」
- 「可學習圖構建」是否在某些資料型態（few-shot / 特徵類型 / 資料量）下更有利？

---

## 整體設計：兩種運行模式

IDGL-GNN 在 TaBLEau wrapper 中有 **一個重要的自動分支**：

### Mode A：Learned dense adjacency（小 N）
- 觸發條件：
  - IDGL GraphLearner 可 import（`_IDGL_AVAILABLE=True`）
  - 估計 dense adjacency 記憶體 $N^2\times 4\text{ bytes}$ 未超過閾值（預設 `idgl_dense_memory_gb=10.0`）
- 核心計算：
  - GraphLearner 產生 `raw_adj`（形狀 `[N, N]`）
  - 做 row-wise softmax 得到 `adj`（row-stochastic）
  - 做 GCN 常用的對稱正規化 $\tilde{A}=D^{-1/2}(A+I)D^{-1/2}$
  - 用 DenseGCN（矩陣乘法版本）做 message passing

### Mode B：Fixed sparse kNN fallback（大 N）
- 觸發條件：
  - Mode A 不成立（通常是 $N$ 太大，或 IDGL 依賴不可用）
- 核心計算：
  - sklearn `NearestNeighbors` 在 CPU 建 kNN（`edge_index`，形狀 `[2, E]`）
  - PyG `GCNConv` 堆疊做 message passing

> 重要：不論哪種模式，**資料都是 transductive**（train/val/test 合併成同一張圖），梯度只透過 train mask 回傳。

---

## 資料管線（TaBLEau wrapper）

入口函式：`main(train_df, val_df, test_df, dataset_results, config, gnn_stage=None)`。

### Step 0：合併 split（Transductive）
- `all_df = concat([train_df, val_df, test_df])`
- `target_col` 優先 `target`，其次 `label`，否則取最後一欄

### Step 1：特徵與標籤抽取
- `X = all_df.drop(target_col).values` → 形狀 `[N, D]`
- `y_raw = all_df[target_col].values`

### Step 2：特徵標準化（StandardScaler）
- `X = StandardScaler().fit_transform(X)`

### Step 3：任務處理（classification / regression）
- 判斷規則：`task_type in ['binclass','multiclass']` → classification；否則 regression
- Classification：
  - `LabelEncoder` → `y_enc`（整數類別）
  - one-hot → `y`（形狀 `[N, C]`）
- Regression：
  - `y = y_raw.reshape(-1,1)`（形狀 `[N, 1]`）

### Step 4：建立 train/val/test mask
- 依 concat 的順序切 mask：
  - `train_mask[:n_train]=True`
  - `val_mask[n_train:n_train+n_val]=True`
  - `test_mask[...] = True`

### Step 5：決定是否走 learned adjacency
- 預估 dense adjacency 記憶體：
  - `estimated_memory_gb = (N^2 * 4) / 1024^3`
- 閾值：`idgl_dense_memory_gb`（預設 10GB）

### Step 6：建圖（Learned 或 Fixed）
- learned：模型 forward 時由 GraphLearner 即時產生 dense adjacency
- fixed：先建立 `edge_index = build_knn_edge_index_sklearn(X, k)`（CPU，`n_jobs=1`）

---

## 模型架構細節

### 1) GraphLearner（IDGL 上游）如何輸出 adjacency
檔案：`ModelComparison/IDGL/src/core/layers/graphlearn.py`。

在 `metric_type='attention'`（本 baseline 預設）下：
- 有 `num_pers=16` 個線性投影 `Linear(input_size→hidden_size)`，每個視為一個「視角」
- 每個視角：
  - `context_fc = relu(W_p context)`
  - `score_p = context_fc @ context_fc^T`
- 最終 `attention = mean_p(score_p)`
- 若設 `topk=k`：保留每列 top-k 的分數（其餘設為 `markoff_value=-INF`）

> 本 baseline 呼叫方式是 `raw_adj = graph_learner(x)`，其中 `x` 是節點特徵 `[N, D]`。

### 2) Dense adjacency 路徑（Mode A）
IDGLTabularNet.forward（use_learned_adj=True）：

```
X [N,D]
  ↓ GraphLearner
raw_adj [N,N] (帶 -INF mask 的加權鄰接)
  ↓ row-wise softmax
adj [N,N] (row-stochastic)
  ↓ symmetric normalize + add self-loop
adj_norm [N,N]
  ↓ DenseGCN (L layers)
logits [N,out_dim]
```

DenseGCN 是「矩陣乘法版」GCN：每層做 `adj_norm @ (H @ W)`。

### 3) Sparse kNN 路徑（Mode B）
IDGLTabularNet.forward（use_learned_adj=False）：

```
X [N,D]
  ↓ sklearn kNN (CPU)
edge_index [2,E]
  ↓ SparseGCN (PyG GCNConv 堆疊)
logits [N,out_dim]
```

> 這個 fallback 的本質是：**把 IDGL 的「learned graph」退化成固定 kNN**，只保留 GNN message passing。

---

## 訓練與評估流程（含早停）

### Loss
- classification：`cross_entropy(logits[train_mask], y_onehot.argmax(-1))`
- regression：`mse_loss(logits[train_mask], y[train_mask])`

### Metric（與 TaBLEau 其他 comparison baseline 對齊）
- binclass：AUC
- multiclass：Accuracy
- regression：RMSE

### Early stopping
- 以 **val metric** 做 early stop：
  - classification：越大越好
  - regression：越小越好
- `patience = config['patience']`（預設 10）
- 回傳：
  - `early_stop_epochs`：停止 epoch
  - `patience_counter`：最後累積的 counter

---

## PyTorch-Frame 五階段對應（Stage mapping）

> 這裡的「stage 類比」是為了把 self-contained baseline 的內部流程，映射回 SAGE 的五階段語言，方便做對照分析。

### Stage 定義（沿用 PyTorch-Frame）
| Stage | 名稱 | 說明 |
|---|---|---|
| start | Start | 資料清理/前處理（缺失值、基本處理） |
| materialize | Materialize | DataFrame → tensor/np array |
| encoding | Encoding | 數值/類別編碼、尺度變換 |
| columnwise | Columnwise | 交互/關係建模（Attention 或 GNN message passing） |
| decoding | Decoding | 讀出/預測頭 |

### IDGL-GNN 元件對應表
| IDGL-GNN 元件 | 程式位置（概念上） | 對應 Stage | 功能 |
|---|---|---|---|
| concat splits + mask | `main()` | start / materialize | 轉導式設定與 mask 定義 |
| `StandardScaler` | `main()` | encoding | 數值特徵尺度對齊 |
| `LabelEncoder` + one-hot | `main()` | encoding | 目標轉換（分類） |
| GraphLearner（attention/topk） | `IDGLTabularNet.graph_learner` | columnwise | **可學習圖構建**（row graph） |
| adjacency softmax + normalization | `forward()` | columnwise | 把 score 轉為可用的 GCN adjacency |
| DenseGCN / SparseGCN | `DenseGCN` / `SparseGCN` | columnwise | **GNN 消息傳遞** |
| logits out_dim | Dense/Sparse 最後一層 | decoding（讀出） | 產生預測 logits |

**一句話總結**：IDGL-GNN 的「圖 + GNN」主要對應 PyTorch-Frame 的 **columnwise**。

---

## 與 DGM / LAN-GNN 的定位比較

### 與 DGM（[DGM_GNN_Stage_Analysis.md](DGM_GNN_Stage_Analysis.md)）
- 相同點：
  - 都是 row-graph、transductive、圖構建與 message passing 在模型內部完成
- 不同點：
  - DGM：每層用 Gumbel-kNN 動態採樣（可微分離散採樣），甚至帶強化學習式 logprobs
  - IDGL：用 GraphLearner 產生加權相似度，再 top-k 截斷（更像「可學習的相似度函數 + kNN」）
  - DGM 的「圖重建」更強耦合層級迭代；IDGL baseline 目前以 **輸入特徵 X** 直接學圖（未做層間反覆重建）

### 與 LAN-GNN（[LAN_GNN_Architecture_Analysis.md](LAN_GNN_Architecture_Analysis.md)）
- 相同點：
  - 都可視作「learnable adjacency + GCN」的路線
  - 都有大 N 時的 sparse kNN fallback
- 不同點：
  - LAN：使用 DGG_StraightThrough / LocalTopKAdj 產生 adjacency（含 straight-through hard top-k）
  - IDGL：使用 GraphLearner 的 multi-perspective attention similarity（更偏 feature-projection + inner product）

---

## 工程與穩定性細節（大 N fallback）

1. **避免 dense adjacency OOM**
   - 以 `estimated_memory_gb = N^2*4/1024^3` 估算 float32 adjacency 的理論成本
   - 超過 `idgl_dense_memory_gb` 直接切到 sparse kNN

2. **kNN 建圖的 CPU/BLAS 穩定性**
   - `NearestNeighbors(..., n_jobs=1)`
   - 若 `threadpoolctl` 可用，會以 `threadpool_limits(limits=1)` 限制底層 BLAS thread，避免大資料時 OpenBLAS oversubscription

3. **依賴**
   - learned adjacency：需要 IDGL 的 `GraphLearner` 可 import
   - sparse fallback：需要 `torch_geometric`（PyG）提供 `GCNConv`

---
