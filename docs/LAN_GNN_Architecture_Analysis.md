# LAN-GNN: 學習自適應鄰接圖神經網絡

## 目錄
1. [模型概述](#模型概述)
2. [完整架構剖析](#完整架構剖析)
3. [核心組件詳解](#核心組件詳解)
4. [資料管線](#資料管線)
5. [PyTorch-Frame 階段對應](#pytorch-frame-階段對應)
6. [與 DGM 模型的比較](#與-dgm-模型的比較)
7. [實驗配置與運行](#實驗配置與運行)
8. [總結](#總結)

---

## 模型概述

### 基本信息
- **全名**: LAN-GNN (Learnable Adaptive Neighborhood GNN)
- **參考論文**: "Learning Adaptive Neighborhoods for Graph Convolutional Networks"
- **實現文件**: `/home/skyler/ModelComparison/TaBLEau/models/comparison/lan_gnn.py`
- **上游代碼**: `/home/skyler/ModelComparison/learning-adaptive-neighborhoods-for-gnns/`
- **集成方式**: 獨立 comparison 模型（不插入 PyTorch-Frame 五階段 GNN）

### 模型定位
LAN-GNN 是一個**自帶完整 GNN 的 baseline 模型**，與 DGM 的區別：

| 特性 | DGM | LAN-GNN |
|------|-----|---------|
| **集成方式** | 可插入五階段框架 | 獨立 baseline（無外部 GNN 注入） |
| **圖構建** | 動態 k-NN（每層重新計算） | 每層自適應鄰接（DGG 或局部 top-k） |
| **訓練方式** | 端到端聯合優化 | 單次前向傳播優化 |
| **資料規模** | 任意（可批量） | 轉導（全圖）+ 記憶體調節 |
| **應用場景** | 插入模型管線強化 | 獨立對標實驗 |

### 核心特點
1. **轉導學習 (Transductive)**
   - 將訓練、驗證、測試集合併為單一圖
   - 所有節點視為一個大圖的節點
   - 梯度只通過訓練 mask 反傳

2. **自適應鄰接生成**
   - 使用 DGG_StraightThrough（或局部 top-k 後備）學習每層的 k-NN 結構
   - 距離度量採 Euclidean（metric），避免 MLP 距離的數值不穩定

3. **魯棒的後備機制**
   - 若 DGG 生成器失敗 → 自動降級到 LocalTopKAdj
   - 小資料集若超記憶體閾值 → 禁用 DGG，使用固定 kNN

4. **度量穩定化**
   - 分類損失採 cross-entropy（而非 BCE）
   - 預測與目標經過 NaN/Inf 淨化再計算指標

---

## 完整架構剖析

### 整體管線

```
Input Features (X) [N, D]
    ↓
[Standardize 標準化] → StandardScaler
    ↓
[Encode Labels 編碼目標]
    ├─ Classification: one-hot encoding [N, C]
    └─ Regression: reshape to [N, 1]
    ↓
┌─────────────────────────────────────────┐
│  LANGNN: Multi-Layer Adaptive GNN      │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ LANLayer 1                       │  │
│  │  Input: [N, D]                   │  │
│  │                                  │  │
│  │  ┌─ proj: [N, D] → [N, H]       │  │
│  │  │  Linear(D, H) + ReLU         │  │
│  │  └─ Dropout(dropout=0.2)        │  │
│  │                                  │  │
│  │  ┌─ 自適應鄰接生成              │  │
│  │  │  Option 1: DGG_StraightThrough│  │
│  │  │   • 輸入: [1, N, H]  (加 batch dim)│
│  │  │   • 輸出: adj_soft [N, N]    │  │
│  │  │   • 距離: metric (Euclidean) │  │
│  │  │   • k-NN: hard top-k          │  │
│  │  │   • 噪音: False (確定性)      │  │
│  │  │                               │  │
│  │  │  Option 2 (後備): LocalTopKAdj│  │
│  │  │   • 相似度: cosine            │  │
│  │  │   • 採樣: Gumbel noise + top-k│  │
│  │  └─ normalize_adj_dense          │  │
│  │     D^{-1/2} A D^{-1/2}         │  │
│  │                                  │  │
│  │  ┌─ DenseGraphConvolution      │  │
│  │  │  A_norm @ (X @ W)            │  │
│  │  │  消息傳遞 + ReLU + Dropout   │  │
│  │  └─ 輸出: [N, H]                │  │
│  │                                  │  │
│  └──────────────────────────────────┘  │
│           ↓ (層疊)                      │
│  ┌──────────────────────────────────┐  │
│  │ LANLayer 2 (同上結構)             │  │
│  │ 輸入: [N, H] → 輸出: [N, H]       │  │
│  └──────────────────────────────────┘  │
│           ↓                              │
├─ 解碼頭 (MLP)                           │
│  Linear(H, H/2) + ReLU + Dropout       │
│  Linear(H/2, out_dim)                  │
│  out_dim = C (分類) or 1 (回歸)        │
│           ↓                              │
└─────────────────────────────────────────┘
    ↓
Output Logits [N, C] or [N, 1]
    ↓
[計算損失 & 指標]
    ├─ Train: Cross-entropy (分類) / MSE (回歸)
    ├─ Val/Test: AUC/Acc/RMSE + NaN 淨化
    └─ 早停: patience=10
```

---

## 核心組件詳解

### 1. DenseGraphConvolution (密集圖卷積)

```python
class DenseGraphConvolution(nn.Module):
    """Dense adjacency variant of GCN layer (no biases/norms)."""
    def __init__(self, in_features, out_features):
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
    
    def forward(self, x, adj_norm):
        # x: [N, Fin]
        # adj_norm: [N, N] (已標準化的鄰接矩陣)
        support = x @ self.weight  # [N, Fout]
        out = adj_norm @ support   # [N, N] @ [N, Fout] = [N, Fout]
        return out
```

**特點**:
- 無偏差項、無層標準化（輕量級）
- 直接使用稀疏鄰接矩陣進行矩陣乘法
- 權重初始化採 uniform 分佈 [-stdv, stdv]

**對應階段**: **Columnwise** (圖上消息傳遞)

---

### 2. normalize_adj_dense (鄰接矩陣標準化)

```python
def normalize_adj_dense(A: torch.Tensor) -> torch.Tensor:
    """計算 D^{-1/2} A D^{-1/2}"""
    N = A.size(0)
    A = A.clone()
    A[torch.arange(N), torch.arange(N)] = 0  # 移除自環
    A_hat = A + torch.eye(N, ...)            # 加回自環
    deg = A_hat.sum(-1)                      # 度數
    deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ A_hat @ D
```

**功能**:
1. 去除自環
2. 加入自環（保留節點信息）
3. 計算度數矩陣的平方根倒數
4. 標準化 GCN 式: $\tilde{A} = D^{-1/2} \tilde{A} D^{-1/2}$

---

### 3. LocalTopKAdj (局部 Top-K 鄰接生成)

```python
class LocalTopKAdj(nn.Module):
    """後備鄰接生成器: cosine 相似度 + Gumbel 噪音 + top-k"""
    def __init__(self, k: int = 10):
        self.k = k
    
    def forward(self, h: torch.Tensor, temperature: float = 0.5, noise: bool = True):
        # h: [N, F]
        sim = (h @ h.t()) / (||h|| * ||h.t()||)  # cosine [N, N] ∈ [-1, 1]
        sim = (sim + 1.0) / 2.0                   # 縮放到 [0, 1]
        
        if noise:
            g = -log(-log(Uniform)) # Gumbel(0,1)
            sim = sim + g
        
        probs = softmax(sim / temp)  # [N, N]
        adj = top_k_one_hot(probs, k)  # hard [N, N]
        adj[self_loop_positions] = 0    # 移除自環
        return adj
```

**特點**:
- Cosine 相似度（特徵標準化後）
- 可選 Gumbel 噪音（訓練時添加）
- Hard top-k: 每行恰好 k 個非零元素
- 自環遮罩

**何時觸發**: DGG 生成器失敗、超記憶體閾值、或層級 forward 異常

---

### 4. LANLayer (自適應鄰接層)

```python
class LANLayer(nn.Module):
    def __init__(self, dim, hidden, k=10, temperature=0.5, dropout=0.2):
        self.proj = nn.Linear(dim, hidden)
        self.gen = DGG_StraightThrough(...)  # 主要生成器
        self.dense_conv = DenseGraphConvolution(hidden, hidden)
        self.local_gen = LocalTopKAdj(k=k)   # 後備生成器
    
    def forward(self, x):
        # x: [N, Fin]
        h = ReLU(self.proj(x))              # 投影到隱層 [N, H]
        h = Dropout(h)
        
        try:
            h_b = h.unsqueeze(0)            # 加 batch dim: [1, N, H]
            adj_soft = self.gen(h_b, temp=..., noise=False)[0]  # [N, N]
        except Exception as e:
            logger.warning(f"DGG failed, using LocalTopKAdj: {e}")
            adj_soft = self.local_gen(h, temperature=..., noise=...)
        
        adj_norm = normalize_adj_dense(adj_soft)
        out = self.dense_conv(h, adj_norm)
        out = ReLU(out) + Dropout(out)
        return out  # [N, H]
```

**層級流程**:
1. **特徵投影**: Linear + ReLU → 從 dim 變換到 hidden
2. **自適應鄰接**: DGG (或後備) 生成圖結構
3. **標準化**: 鄰接矩陣 Symmetric Normalization
4. **消息傳遞**: Dense GCN 卷積
5. **激活 & 正則**: ReLU + Dropout

**對應階段**: 
- **Encoding**: 投影層
- **Columnwise**: 鄰接生成 + 圖卷積

---

### 5. LANGNN (完整模型)

```python
class LANGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2, k=10, layers=2, dropout=0.2):
        self.layers = nn.ModuleList([
            LANLayer(dim=in_dim, hidden=hidden, k=k, ...),
            LANLayer(dim=hidden, hidden=hidden, k=k, ...),
            # ... (layers 個 LANLayer)
        ])
        self.head = MLP(hidden → hidden/2 → out_dim)
    
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        logits = self.head(h)  # [N, out_dim]
        return logits
```

**結構**:
- **輸入**: [N, D] 原始特徵
- **隱層堆疊**: N 個 LANLayer，每層都學習自己的鄰接
- **輸出頭**: 簡單 MLP (2 層)
- **輸出**: [N, C] (分類) 或 [N, 1] (回歸)

**對應階段**:
- **Encoding**: 所有 LANLayer 的投影與後續變換
- **Columnwise**: LANLayer 內的鄰接生成與圖卷積
- **Decoding**: MLP 解碼頭

---

## 資料管線

### 轉導式資料準備

```python
def main(train_df, val_df, test_df, dataset_results, config, gnn_stage=None):
    # === 步驟 1: 合併資料 ===
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # [n_train + n_val + n_test, features + target]
    
    # === 步驟 2: 特徵與目標分離 ===
    X = all_df.drop(columns=['target']).values  # [N, D]
    y_raw = all_df['target'].values              # [N]
    
    # === 步驟 3: 特徵標準化 ===
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # StandardScaler 已自動標準化到 [0, 1] 附近
    
    # === 步驟 4: 目標編碼 ===
    is_classification = task_type in ['binclass', 'multiclass']
    if is_classification:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)
        num_classes = len(np.unique(y_enc))
        y = np.eye(num_classes)[y_enc]  # One-hot [N, C]
    else:
        y = y_raw.reshape(-1, 1)  # [N, 1]
    
    # === 步驟 5: 掩碼生成 ===
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    
    # === 步驟 6: 記憶體檢查與適應 ===
    estimated_memory_gb = (X.shape[0]**2 * 4) / (1024**3)
    memory_threshold_gb = 10.0
    use_learnable_adj = DGG_AVAILABLE and (estimated_memory_gb <= memory_threshold_gb)
    
    if not use_learnable_adj:
        logger.warning(f"N={X.shape[0]} needs {estimated_memory_gb:.2f}GB, using fixed kNN")
    
    # === 步驟 7: 資料集與加載器 ===
    dataset = TabularGraphDataset(X, y, train_mask | val_mask | test_mask)
    dataloader = DataLoader(dataset, batch_size=1)  # 轉導: 一次加載整個圖
```

### 掩碼式訓練

```python
for epoch in range(epochs):
    # === 訓練 ===
    model.train()
    logits = model(X_t)  # [N, C]
    
    # 只在訓練掩碼上計算損失
    train_logits = logits[train_mask_t]
    train_y = y_t[train_mask_t]
    loss = F.cross_entropy(train_logits, train_y.argmax(-1))
    loss.backward()
    optimizer.step()
    
    # === 評估 ===
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        
        # 驗證集指標
        val_logits = logits[val_mask_t].detach().cpu().numpy()
        val_y = y_t[val_mask_t].detach().cpu().numpy()
        val_metric = _compute_metric(val_y, val_logits, task, num_classes)
        
        # 測試集指標
        test_logits = logits[test_mask_t].detach().cpu().numpy()
        test_y = y_t[test_mask_t].detach().cpu().numpy()
        test_metric = _compute_metric(test_y, test_logits, task, num_classes)
    
    # === 早停 ===
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        stopped_epoch = epoch + 1
        break
```

### 轉導式優勢與限制

| 面向 | 優勢 | 限制 |
|------|------|------|
| **圖結構** | 利用全量資料，圖更完整 | 記憶體 O(N²) |
| **梯度** | 完整的端到端可微分 | 無法小批量訓練（需調整） |
| **早停** | 驗證集在圖內，無分佈漂移 | 訓練集大時速度慢 |
| **測試** | 測試節點可利用鄰近節點特徵 | 無法處理完全新節點 |

---

## PyTorch-Frame 階段對應

### LAN-GNN 作為獨立 Baseline

由於 LAN-GNN 已內含完整 GNN，它作為 comparison 模型**不參與**五階段框架的 GNN 注入。

| 階段 | 是否有 GNN 注入 | 解釋 |
|------|----------------|------|
| **start** | ✗ | 無，LAN-GNN 自行準備資料 |
| **materialize** | ✗ | 無，LAN-GNN 自行張量化 |
| **encoding** | ✓ (內置) | LANLayer 的投影層 |
| **columnwise** | ✓ (內置) | LANLayer 的鄰接生成 + 圖卷積 |
| **decoding** | ✓ (內置) | MLP 解碼頭 |

### 對應表

| LAN-GNN 組件 | PyTorch-Frame Stage | 功能 |
|------------|---------------------|------|
| Standardize + LabelEncode | Start | 資料預處理 |
| X_t = FloatTensor(X) | Materialize | 張量化 |
| LANLayer.proj (Linear) | Encoding | 特徵投影 |
| DGG/LocalTopKAdj | Columnwise | 圖結構學習 |
| DenseGraphConvolution | Columnwise | 消息傳遞 |
| LANGNN.head (MLP) | Decoding | 最終預測 |

---

## 與 DGM 模型的比較

### 核心異同

| 維度 | DGM | LAN-GNN |
|-----|-----|---------|
| **圖動態性** | 每層重新計算邊 | 每層重新計算鄰接（依配置） |
| **編碼方式** | embed_f 參與圖構建 | proj 投影到隱層，後從隱層生成圖 |
| **距離度量** | 可配（euclidean/cosine） | 固定 metric（Euclidean），後備 cosine |
| **k-NN 生成** | Gumbel-Softmax (可微) | DGG_StraightThrough (可微) + 後備 LocalTopKAdj |
| **資料規模** | 支援小批量迭代 | 全圖轉導（記憶體限制） |
| **訓練策略** | 端到端 + 圖結構損失 | 監督損失 + 早停 |
| **集成方式** | 可插入五階段框架 | 獨立 baseline (comparison) |

### 實驗對比

```bash
# DGM 作為 encoding 階段 GNN 注入
python main.py --dataset house --models resnet \
  --gnn_stages encoding --epochs 100

# LAN-GNN 作為獨立 baseline
python main.py --dataset house --models lan_gnn --epochs 100

# SCARF + DGM (多階段)
python main.py --dataset house --models scarf \
  --gnn_stages all --epochs 100
```

---

## 實驗配置與運行

### 推薦參數

```python
# 分類任務 (小資料)
config = {
    'epochs': 100,
    'patience': 10,
    'lr': 1e-3,
    'gnn_hidden_dim': 128,
    'gnn_layers': 2,
    'gnn_dropout': 0.2,
    'lan_k': 10,
    'batch_size': 256,  # 實際無用（轉導），保持 1
}

# 回歸任務 (大資料，打開記憶體自適應)
config = {
    'epochs': 100,
    'patience': 15,
    'lr': 5e-4,
    'gnn_hidden_dim': 64,
    'gnn_layers': 2,
    'gnn_dropout': 0.3,
    'lan_k': 15,
}
```

### 快速驗證

```bash
# 小資料分類（快速健檢）
python main.py --dataset kaggle_Audit_Data --models lan_gnn \
  --epochs 20 --patience 5

# 回歸（確認 RMSE 路徑）
python main.py --dataset house --models lan_gnn \
  --epochs 100 --patience 10

# 大資料（測記憶體自適應）
python main.py --dataset helena --models lan_gnn \
  --epochs 50 --patience 10
```

### 關鍵日誌標記

```
[正常運行]
LAN-GNN: Running LAN-GNN (learnable adaptive kNN) model...
Epoch 10/100 - Train: 0.9971, Val: 0.9715, Test: 0.9736
Early stopping at epoch 11 (patience=10)
Best Test AUC: 0.9753

[後備觸發 - LocalTopKAdj]
LAN-GNN: DGG generator failed: ... Falling back to local top-k generator.

[記憶體自適應]
LAN-GNN: N=50000 would need 9.31GB. Using fixed kNN fallback.

[指標淨化]
_compute_metric: Sanitizing NaN/Inf values before AUC computation
```

---

## 總結

### LAN-GNN 的定位
- **自帶 GNN 的 Baseline 模型**，無需外部 GNN 注入
- **轉導式全圖學習**，充分利用未標記資料
- **多層自適應鄰接**，每層動態調整圖結構
- **魯棒後備機制**，處理不穩定場景

### 核心優勢
1. **完整端到端**: 自動化整個 encoding → columnwise → decoding 流程
2. **適應能力**: 記憶體自適應、DGG 失敗自動降級、預測淨化
3. **對標性強**: 作為獨立 baseline，可與其他模型直接比較
4. **穩定性**: 確定性鄰接 + 交叉熵損失 + 指標淨化

---

## 附錄：用超簡單例子理解「LAN-GNN 的 GNN 在哪個階段」

這一節的目標是把 **LAN-GNN 這種自帶 GNN 的 baseline**，用最直覺方式類比到：
1) PyTorch-Frame 的五階段（start/materialize/encoding/columnwise/decoding）
2) 你在 ALIGN 研究裡的 `gnn_stages`（start/materialize/encoding/columnwise/decoding/none）

### 一個玩具例子：每一列都是一個節點

假設表格是二分類，每列是一個客戶：

| row | age | income | label |
|---:|---:|---:|---:|
| 0 | 25 | 30k | 0 |
| 1 | 27 | 35k | 0 |
| 2 | 45 | 90k | 1 |
| 3 | 43 | 85k | 1 |

直覺上 row0↔row1 很像、row2↔row3 很像。LAN-GNN 的核心就是：
1) 先把每個 row 的特徵投影到隱向量（讓距離更有意義）
2) 依據隱向量學一個「誰該連誰」的鄰接（k-NN）
3) 在這個圖上做消息傳遞（把相似節點的資訊聚合）
4) 用 MLP 輸出 label

### 對應到 PyTorch-Frame 五階段（「類比」而非外部注入）

LAN-GNN 在程式中是 comparison baseline，因此不吃外部 `gnn_stage` 注入；但你可以把它的內部運算**類比**如下：

| LAN-GNN 內部步驟 | 代碼概念 | PyTorch-Frame stage 類比 | 為什麼像 |
|---|---|---|---|
| 特徵標準化、目標編碼 | `StandardScaler` / `LabelEncoder` | start | 資料預處理（縮放/編碼） |
| DataFrame→Tensor | `X_t`, `y_t` | materialize | 轉成張量進模型 |
| row 特徵投影 | `LANLayer.proj` | encoding | 把 raw features 投影成 hidden 表徵 |
| 依 hidden 建圖 | `DGG_StraightThrough` 或 `LocalTopKAdj` | columnwise | 做 row-to-row 互動的「結構」 |
| 在圖上聚合 | `DenseGraphConvolution` | columnwise | message passing（互動本身） |
| 最終輸出 | `LANGNN.head` | decoding | 把表徵映射到預測 |

### 對應到 ALIGN 的 `gnn_stages`

在 ALIGN 實驗中，`gnn_stages` 的含義是「把外部 GNN 插進某個 stage」。
但 LAN-GNN 是自帶 GNN，因此在實驗設計上：

1. **跑 LAN-GNN 時**：它等價於 `gnn_stages=none`（因為不插外部 GNN），但模型內部仍做了 GNN。
2. **如果要找最像的注入點來做概念對照**：LAN-GNN 的主要增益最像 `gnn_stage=columnwise`，因為它的核心就是「row-to-row 結構 + message passing」。

一句話總結：LAN-GNN 的 GNN 是「內建的 encoding+columnwise(+decoding) 組合」，如果硬要對照 ALIGN 插入點，最像 **columnwise**。

### 論文式總結表：LAN-GNN vs DGM vs ALIGN 注入

> 讀法：前兩欄是「自帶 GNN 的 baseline 內建機制」，第三欄是「ALIGN 實驗中外部注入的定義」。

| 對照面向 | LAN-GNN（自帶 GNN baseline） | DGM（自帶 GNN baseline） | ALIGN 外部注入（`gnn_stages`） |
|---|---|---|---|
| 模型型態 | comparison baseline；不吃外部注入 | comparison baseline / 原生 GNN；不吃外部注入 | 在既有 tabular 模型管線某一 stage 額外插入 GNN |
| 節點定義 | 每列資料 = 節點（row-graph） | 每列資料 = 節點（row-graph） | 依 stage 而定（可能是 row graph、或特徵互動後的表徵圖） |
| 圖是怎麼來的 | 每層從 hidden 表徵生成 kNN 鄰接（DGG；失敗則 LocalTopKAdj / 固定 kNN） | 每層動態採樣 kNN 邊（Gumbel / 可微採樣）；每層重建 | 多數情況是固定圖或由外部流程產生；通常不會每層重建 |
| message passing 發生在哪 | `DenseGraphConvolution`（密集鄰接上做聚合） | GCN/GAT/EdgeConv 等在動態邊上做聚合 | 插入的那一個 stage：encoding/columnwise/decoding 等 |
| 最接近的 ALIGN 對照點 | **columnwise**（row-to-row 互動是核心） | **columnwise**（建圖+傳遞緊耦合且可學） | `gnn_stage=columnwise` 表示在 columnwise 做外部 GNN 互動 |
| 跟 encoding/decoding 的關係 | 內建 `proj`（類比 encoding）+ `head`（類比 decoding） | 內建 pre-fc/embed_f（類比 encoding）+ 最終 FC（類比 decoding） | 可把外部 GNN 插在任意 stage；與模型內部結構無關 |
| 計算/記憶體特性 | 常見為轉導全圖；鄰接為 [N,N]，記憶體 O(N²)（有閾值降級） | 可轉導或批量/稀疏邊；重建圖成本高，但較可用稀疏邊控記憶體 | 取決於你插入的 GNN 形式與圖大小 |
| 實驗詮釋建議 | 視為 `gnn_stages=none` 的 baseline（但內部本來就做了 GNN） | 同左；同時強調「圖也可學」是其差異點 | 用來測「把 GNN 插在某一 stage」對既有模型的增益 |

### 關鍵實驗建議
- 與 DGM（作為框架注入）對比，驗證端到端 vs. 混合的性能差異
- 測試不同資料規模下的記憶體行為與自動降級效果
- 在分類 vs. 回歸任務上驗證通用性

---

## 附錄：代碼結構樹

```
models/comparison/lan_gnn.py
├── 導入與依賴配置
│   ├── DGG_StraightThrough (from learning-adaptive-neighborhoods-for-gnns)
│   └── GCNConv (from torch_geometric, 可選)
│
├── DenseGraphConvolution (密集 GCN 卷積)
├── normalize_adj_dense (鄰接矩陣標準化)
├── LocalTopKAdj (後備鄰接生成器)
│
├── LANLayer (自適應鄰接層)
│   ├── self.proj: Linear(dim → hidden)
│   ├── self.gen: DGG_StraightThrough (主)
│   ├── self.dense_conv: DenseGraphConvolution
│   └── self.local_gen: LocalTopKAdj (後備)
│
├── LANGNN (完整模型)
│   ├── self.layers: ModuleList[LANLayer]
│   └── self.head: MLP(hidden → out_dim)
│
├── TabularGraphDataset (轉導資料集)
├── _compute_metric (指標計算 + NaN 淨化)
│
└── main(...) 主函式
    ├── 資料合併與標準化
    ├── 特徵與目標編碼
    ├── 掩碼生成
    ├── 記憶體檢查
    ├── 模型初始化
    ├── 訓練迴圈（掩碼式）
    ├── 驗證與早停
    └── 結果回傳 (best_val_metric, best_test_metric, early_stop_epochs)
```
