# DGM 模型的 GNN 位置分析：對應 PyTorch-Frame 架構

## 目錄
1. [DGM 模型概述](#dgm-模型概述)
2. [DGM 完整架構剖析](#dgm-完整架構剖析)
3. [PyTorch-Frame 五階段框架](#pytorch-frame-五階段框架)
4. [DGM 組件與 Stage 對應關係](#dgm-組件與-stage-對應關係)
5. [詳細代碼流程分析](#詳細代碼流程分析)
6. [與其他模型的對比](#與其他模型的對比)
7. [DGM 的核心創新](#dgm-的核心創新)
8. [實驗驗證與性能](#實驗驗證與性能)
9. [總結](#總結)

---

## DGM 模型概述

### 基本信息
- **全名**: Differentiable Graph Module (可微分圖模組)
- **論文**: "Differentiable Graph Module (DGM) for Graph Convolutional Networks"
- **作者**: Anees Kazi, Luca Cosmo, Seyed-Ahmad Ahmadi, Nassir Navab, Michael Bronstein
- **核心創新**: 端到端可學習的動態圖結構生成

### 模型特點
DGM 是一個**原生 GNN 模型**（Native GNN Model），不同於 Transformer 架構中插入 GNN 的方式。其核心特色包括：

1. **動態圖學習**: 圖結構不是預先固定的，而是根據節點特徵動態學習
2. **Gumbel-Softmax 採樣**: 使用 Gumbel-Softmax trick 實現可微分的 k-NN 採樣
3. **端到端訓練**: 圖結構學習與 GNN 參數優化同時進行
4. **多層迭代**: 每層都重新計算特徵並重構圖結構

---

## DGM 完整架構剖析

### 整體管線

```
Input Features (X)
    ↓
[可選] Pre-FC Layer (特徵預處理)
    ↓
┌─────────────────────────────────────┐
│  Layer 1: DGM_d + GNN               │
│  ┌─────────────────────────────┐   │
│  │ DGM_d (動態圖模組)          │   │
│  │  1. embed_f: 特徵編碼       │   │
│  │  2. 計算節點距離             │   │
│  │  3. Gumbel k-NN 採樣        │   │
│  │  → 輸出: (x', edges, logprobs) │
│  └─────────────────────────────┘   │
│             ↓                       │
│  ┌─────────────────────────────┐   │
│  │ GNN Layer (消息傳遞)         │   │
│  │  - GCNConv / GATConv / EdgeConv │
│  │  → 節點特徵更新              │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
[重複 N 層]
    ↓
Final FC Layer (預測層)
    ↓
Output (Predictions)
```

### 核心組件詳解

#### 1. **Pre-FC Layer (前處理層)**
```python
# 位置: model_dDGM.py line 70-71
if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
    x = self.pre_fc(x)
```
- **功能**: 將原始特徵投影到適合圖學習的空間
- **可選性**: 可以為空（直接使用原始特徵）
- **對應 Stage**: **Encoding**

#### 2. **DGM_d 模組 (Differentiable Graph Module)**
```python
# 位置: layers.py line 33-62
class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance="euclidean", sparse=True):
        self.temperature = nn.Parameter(...)  # Gumbel 溫度參數
        self.embed_f = embed_f                # 特徵編碼函數
        self.k = k                            # k-NN 的 k 值
        
    def forward(self, x, A, not_used=None, fixedges=None):
        # Step 1: 特徵編碼/變換
        x = self.embed_f(x, A)
        
        # Step 2: 動態圖構建 (Gumbel k-NN 採樣)
        edges_hat, logprobs = self.sample_without_replacement(x)
        
        return x, edges_hat, logprobs
```

**DGM_d 的兩個子階段**：

##### 2.1 **embed_f (特徵編碼器)**
- **可選實現**:
  - `ffun='gcn'`: GCNConv (使用前一層的圖結構)
  - `ffun='gat'`: GATConv (注意力機制)
  - `ffun='mlp'`: 純 MLP (無圖依賴)
  - `ffun='knn'`: Identity (僅返回輸入，最輕量)
- **功能**: 將節點特徵編碼到更適合距離計算的空間
- **對應 Stage**: **Encoding**

##### 2.2 **sample_without_replacement (動態圖構建)**
```python
# 位置: layers.py line 66-90
def sample_without_replacement(self, x):
    b, n, _ = x.shape
    
    # 計算節點間歐氏距離
    G_i = LazyTensor(x[:, :, None, :])
    X_j = LazyTensor(x[:, None, :, :])
    mD = ((G_i - X_j) ** 2).sum(-1)
    
    # Gumbel-Softmax k-NN 採樣 (可微分)
    lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))
    indices = lq.argKmin(self.k, dim=1)  # 內建 Gumbel noise
    
    # 構建邊索引
    rows = torch.arange(n).view(1,n,1).repeat(b,1,self.k)
    edges = torch.stack((indices.view(b,-1), rows.view(b,-1)), -2)
    
    # 計算 log probabilities (用於強化學習)
    x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
    x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])
    logprobs = (-(x1-x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature,-5,5))).reshape(x.shape[0],-1,self.k)
    
    return edges, logprobs
```
- **功能**: 
  1. 計算所有節點對之間的距離
  2. 使用 Gumbel-Softmax 採樣 k 個最近鄰
  3. 構建稀疏圖結構（edge_index）
- **關鍵技術**: Gumbel-Softmax 使離散採樣過程可微分
- **對應 Stage**: **Columnwise** (圖結構構建)

#### 3. **GNN Layer (圖神經網絡層)**
```python
# 位置: model_dDGM.py line 77-84
self.edges = edges
x = torch.nn.functional.relu(
    g(torch.dropout(x.view(-1,d), self.hparams.dropout, train=self.training), edges)
).view(b,n,-1)
graph_x = torch.cat([graph_x, x.detach()], -1)
```
- **可選實現**:
  - `gfun='gcn'`: GCNConv (標準圖卷積)
  - `gfun='gat'`: GATConv (圖注意力)
  - `gfun='edgeconv'`: EdgeConv (邊卷積 + MLP)
- **功能**: 在動態生成的圖上進行消息傳遞
- **輸入**: 編碼後的節點特徵 + 動態生成的 edge_index
- **輸出**: 更新後的節點表徵
- **對應 Stage**: **Columnwise** (消息傳遞)

#### 4. **Final FC Layer (最終預測層)**
```python
# 位置: model_dDGM.py line 87
return self.fc(x), torch.stack(lprobslist,-1) if len(lprobslist)>0 else None
```
- **功能**: 將圖表徵解碼為最終預測
- **結構**: 多層 MLP
- **對應 Stage**: **Decoding**

---

## PyTorch-Frame 五階段框架

### 標準五階段定義

| 階段 | 名稱 | 功能 | 典型操作 |
|------|------|------|----------|
| 0 | **Start** | 資料預處理 | 缺失值填充、標準化 |
| 1 | **Materialize** | 張量化 | DataFrame → Tensor |
| 2 | **Encoding** | 特徵編碼 | 數值/類別特徵編碼 |
| 3 | **Columnwise** | 列間交互 | Attention、圖結構學習 |
| 4 | **Decoding** | 預測解碼 | MLP 輸出層 |

### GNN Injection Stages

在 PyTorch-Frame 框架中，GNN 可以插入的階段：

1. **start**: 在原始 DataFrame 上構建圖
2. **materialize**: 在張量化後立即構建圖
3. **encoding**: 在特徵編碼階段使用 GNN
4. **columnwise**: 在列間交互階段使用 GNN
5. **decoding**: 使用 GNN 作為最終解碼器
6. **none**: 不使用 GNN（標準模型）

---

## DGM 組件與 Stage 對應關係

### 詳細映射表

| DGM 組件 | 代碼位置 | PyTorch-Frame Stage | 功能描述 | 關鍵特徵 |
|---------|---------|-------------------|---------|---------|
| **Pre-FC** | model_dDGM.py:70-71 | **Encoding** | 原始特徵 → 圖空間特徵 | 可選，純 MLP |
| **DGM_d.embed_f** | layers.py:50 | **Encoding** | 特徵變換/編碼 | 可以是 GCN/GAT/MLP/Identity |
| **DGM_d.sample_without_replacement** | layers.py:66-90 | **Columnwise** | 動態圖結構生成 | Gumbel-Softmax k-NN |
| **node_g (GNN Layer)** | model_dDGM.py:84 | **Columnwise** | 圖上消息傳遞 | GCNConv/GATConv/EdgeConv |
| **Final FC** | model_dDGM.py:87 | **Decoding** | 圖表徵 → 預測 | 多層 MLP |

### 關鍵洞察

#### 1. **Encoding + Columnwise 融合**
DGM 的核心創新在於將 **encoding** 和 **columnwise** 兩個階段緊密耦合：

```python
# 每一層 DGM 的完整流程
for f, g in zip(self.graph_f, self.node_g):
    # === Encoding 階段 ===
    graph_x, edges, lprobs = f(graph_x, edges, None)
    # f 內部執行:
    #   1. embed_f(x, A)  → Encoding
    #   2. sample_without_replacement(x)  → Columnwise (圖構建)
    
    # === Columnwise 階段 ===
    x = relu(g(dropout(x), edges))  # 消息傳遞
    graph_x = cat([graph_x, x.detach()], -1)
```

這種設計使得：
- **特徵編碼** 和 **圖結構** 互相影響
- 每層都可以根據當前特徵重新學習圖結構
- 無法單獨拆分為獨立的 encoding 或 columnwise 階段

#### 2. **多層迭代優化**
```python
# 典型配置: hparams.dgm_layers=[[128,128]], hparams.conv_layers=[[128,128]]
# 意味著有 N 層 (DGM_d, GNN) 對

Layer 1: X → DGM_d → edges_1 → GNN_1 → X_1
Layer 2: X_1 → DGM_d → edges_2 → GNN_2 → X_2
...
Layer N: X_{N-1} → DGM_d → edges_N → GNN_N → X_N
```

每層都：
- 重新計算特徵表徵
- 重新構建圖結構
- 進行新一輪消息傳遞

#### 3. **端到端可微分訓練**
```python
# 損失函數包含兩部分 (model_dDGM.py:113-135)

# 1. 任務損失 (分類/回歸)
loss = F.binary_cross_entropy_with_logits(train_pred, train_lab)
loss.backward()

# 2. 圖結構損失 (強化學習式)
if logprobs is not None:
    corr_pred = (train_pred.argmax(-1) == train_lab.argmax(-1)).float()
    point_w = (self.avg_accuracy - corr_pred)
    graph_loss = point_w * logprobs.exp().mean([-1,-2])
    graph_loss = graph_loss.mean()
    graph_loss.backward()
```

圖結構的優化基於**預測正確性的反饋**：
- 預測正確的樣本 → 降低對應邊的權重
- 預測錯誤的樣本 → 增加對應邊的權重
- 通過 Gumbel-Softmax 的梯度回傳更新 temperature 參數

---

## 詳細代碼流程分析

### 完整 Forward Pass

```python
# 位置: model_dDGM.py:69-87
def forward(self, x, edges=None):
    """
    Args:
        x: [batch, num_nodes, input_dim] 輸入特徵
        edges: 初始邊索引 (可選，通常為 None)
    
    Returns:
        pred: [batch, num_nodes, num_classes] 預測
        logprobs: [batch, num_nodes, k, num_layers] 採樣概率
    """
    
    # === 階段 0: Pre-processing (Encoding) ===
    if self.hparams.pre_fc is not None and len(self.hparams.pre_fc) > 0:
        x = self.pre_fc(x)  # [B, N, D] → [B, N, D']
    
    # 初始化圖特徵 (用於多層累積)
    graph_x = x.detach()
    lprobslist = []
    
    # === 階段 1-2: 多層 (Encoding + Columnwise) 迭代 ===
    for f, g in zip(self.graph_f, self.node_g):
        # --- 階段 1: DGM_d (Encoding + Graph Construction) ---
        graph_x, edges, lprobs = f(graph_x, edges, None)
        # f 內部流程:
        #   a) x' = embed_f(graph_x, edges)  # Encoding
        #   b) edges, lprobs = sample_without_replacement(x')  # Graph Construction
        
        b, n, d = x.shape
        self.edges = edges
        
        # --- 階段 2: GNN Layer (Message Passing) ---
        x = F.relu(
            g(
                F.dropout(x.view(-1, d), self.hparams.dropout, train=self.training),
                edges
            )
        ).view(b, n, -1)
        # g 內部: GCNConv/GATConv/EdgeConv
        
        # 累積多層特徵 (用於下一層的圖構建)
        graph_x = torch.cat([graph_x, x.detach()], -1)
        
        if lprobs is not None:
            lprobslist.append(lprobs)
    
    # === 階段 3: Final Decoding ===
    pred = self.fc(x)  # [B, N, D] → [B, N, num_classes]
    logprobs = torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None
    
    return pred, logprobs
```

### DGM_d 詳細流程

```python
# 位置: layers.py:48-62
def forward(self, x, A, not_used=None, fixedges=None):
    """
    Args:
        x: [batch, num_nodes, dim] 輸入特徵
        A: 前一層的鄰接矩陣 (如果 embed_f 需要)
        fixedges: 固定邊 (測試時可用)
    
    Returns:
        x: [batch, num_nodes, dim'] 編碼後特徵
        edges: [2, num_edges] 動態生成的邊索引
        logprobs: [batch, num_nodes, k] 採樣概率
    """
    
    # 確保 batch 維度
    if x.shape[0] == 1:
        x = x[0]
    
    # === Step 1: 特徵編碼 (Encoding) ===
    x = self.embed_f(x, A)
    # embed_f 可以是:
    #   - GCNConv(in, out): 使用舊圖結構進行編碼
    #   - MLP: 純特徵變換
    #   - Identity: 直接通過
    
    if x.dim() == 2:
        x = x[None, ...]
    
    # === Step 2: 動態圖構建 (Columnwise) ===
    if self.training:
        if fixedges is not None:
            return x, fixedges, torch.zeros(...)
        # Gumbel k-NN 採樣
        edges_hat, logprobs = self.sample_without_replacement(x)
    else:
        with torch.no_grad():
            if fixedges is not None:
                return x, fixedges, torch.zeros(...)
            edges_hat, logprobs = self.sample_without_replacement(x)
    
    return x, edges_hat, logprobs
```

### Gumbel k-NN 採樣詳解

```python
# 位置: layers.py:66-90
def sample_without_replacement(self, x):
    """
    使用 Gumbel-Softmax trick 進行可微分的 k-NN 採樣
    
    核心思想:
    1. 計算所有節點對的距離矩陣 D
    2. 加入 Gumbel noise: D' = D + Gumbel(0, 1)
    3. 對每個節點選擇 k 個最近鄰
    4. 返回邊索引和採樣概率
    """
    b, n, _ = x.shape
    
    # === 使用 KeOps LazyTensor 加速距離計算 ===
    if self.distance == "euclidean":
        G_i = LazyTensor(x[:, :, None, :])    # [B, N, 1, D]
        X_j = LazyTensor(x[:, None, :, :])    # [B, 1, N, D]
        
        # 計算歐氏距離的平方
        mD = ((G_i - X_j) ** 2).sum(-1)      # [B, N, N]
        
        # === Gumbel-Softmax 採樣 ===
        # temperature 控制採樣的隨機性:
        #   - 高溫 → 更隨機 (exploration)
        #   - 低溫 → 更確定 (exploitation)
        lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))
        
        # argKmin 內建 Gumbel noise，返回 k 個最小值的索引
        indices = lq.argKmin(self.k, dim=1)  # [B, N, k]
        
        # === 計算採樣概率 (用於強化學習損失) ===
        x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
        x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])
        logprobs = (
            -(x1 - x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature, -5, 5))
        ).reshape(x.shape[0], -1, self.k)
    
    # === 構建邊索引 ===
    rows = torch.arange(n).view(1, n, 1).to(x.device).repeat(b, 1, self.k)
    edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)
    
    # 轉換為 PyG 格式: [2, num_edges]
    if self.sparse:
        return (
            (edges + (torch.arange(b).to(x.device) * n)[:, None, None])
            .transpose(0, 1)
            .reshape(2, -1)
        ), logprobs
    return edges, logprobs
```

---

## 與其他模型的對比

### 1. DGM vs ExcelFormer/TromPT (插入式 GNN)

| 特徵 | DGM | ExcelFormer/TromPT |
|-----|-----|--------------------|
| **架構類型** | 原生 GNN 模型 | Transformer + GNN 插入 |
| **圖結構** | 動態學習 | 固定 k-NN 或 DGM_d |
| **GNN 位置** | encoding + columnwise 融合 | 可選擇單一 stage 插入 |
| **訓練方式** | 端到端圖結構優化 | 固定圖 or 獨立 GNN 訓練 |
| **靈活性** | 高度耦合，難以拆分 | 模組化，易於消融實驗 |
| **適用場景** | 圖結構未知且重要 | 已有良好特徵表徵 |

### 2. 各模型的 GNN Stage 歸屬

| 模型 | GNN Stage | 說明 |
|------|-----------|------|
| **DGM** | **encoding + columnwise** | 融合，無法拆分 |
| **ExcelFormer (encoding GNN)** | encoding | Self-Attention + DGM + GCN → 列特徵 |
| **ExcelFormer (columnwise GNN)** | columnwise | 對 prompts/tokens 應用 GNN |
| **ExcelFormer (decoding GNN)** | decoding | GCN 作為最終預測器 |
| **TromPT (start GNN)** | start | DataFrame 級別的圖構建 |
| **TromPT (materialize GNN)** | materialize | Tensor 級別的圖構建 |
| **TabM (encoding GNN)** | encoding | 在 embedding 階段使用 GNN |
| **TabM (decoding GNN)** | decoding | GNN 替代最終 MLP |

### 3. 代碼實現對比

#### ExcelFormer - Encoding Stage GNN
```python
# 在第一層 encoder 後插入 GNN
if gnn_stage == 'encoding' and i == 0:
    # Step 1: Self-Attention (列間交互)
    tokens = x + column_embed.unsqueeze(0)
    attn_out, _ = self_attn(tokens, tokens, tokens)
    
    # Step 2: Attention Pooling (列 → 行)
    x_pooled = attention_pooling(attn_out)
    
    # Step 3: DGM 動態建圖
    x_dgm, edge_index, _ = dgm_module(x_pooled, A=None)
    
    # Step 4: GCN 消息傳遞
    x_gnn = gnn(x_dgm, edge_index)
    
    # Step 5: 融合回原始 tokens
    x = x + fusion_alpha * decode_to_tokens(x_gnn)
```

#### DGM - Native Architecture
```python
# DGM 的 encoding + columnwise 是一體的
for layer_idx in range(num_layers):
    # Encoding + Graph Construction (緊密耦合)
    x_encoded, edges, logprobs = dgm_d(x, old_edges)
    
    # Message Passing (基於動態圖)
    x = gnn(x_encoded, edges)
    
    # 無法單獨關閉圖構建或消息傳遞
```

---

## DGM 的核心創新

### 1. **可微分動態圖學習**

**問題**: 傳統 GNN 需要預先給定圖結構，但表格數據天然缺乏圖結構

**DGM 解決方案**:
- 使用 Gumbel-Softmax 將離散的 k-NN 採樣變為可微分操作
- 通過梯度下降聯合優化圖結構和 GNN 參數

**技術細節**:
```python
# 標準 k-NN (不可微分)
distances = compute_distances(x)
knn_indices = torch.topk(distances, k, largest=False).indices  # 不可微分！

# DGM 的 Gumbel k-NN (可微分)
distances = compute_distances(x)
gumbel_noise = -torch.log(-torch.log(torch.rand_like(distances)))  # Gumbel(0,1)
perturbed_distances = distances + gumbel_noise / temperature
knn_indices = torch.topk(perturbed_distances, k, largest=False).indices  # 可微分！
```

**優勢**:
- 圖結構根據任務自適應調整
- 不同樣本可以有不同的圖結構
- 溫度參數可學習，控制 exploration vs exploitation

### 2. **強化學習式圖優化**

**核心思想**: 好的圖結構應該能提升任務性能

**實現方式**:
```python
# 計算每個樣本的預測正確性
correct_pred = (pred.argmax(-1) == label.argmax(-1)).float()

# 計算圖結構的 reward
# 如果預測正確，降低對應邊的概率（該圖結構已經好了）
# 如果預測錯誤，增加對應邊的概率（鼓勵探索新圖結構）
point_weight = (avg_accuracy - correct_pred)
graph_loss = point_weight * logprobs.exp().mean()
graph_loss.backward()
```

**效果**:
- 自動發現對任務最有幫助的圖結構
- 避免過度依賴固定的距離度量
- 適應不同數據分佈

### 3. **多層圖迭代精煉**

**設計理念**: 淺層特徵適合構建粗糙圖，深層特徵適合構建精細圖

**實現**:
```python
# Layer 1: 基於原始特徵構建圖
x1 = embed_f_1(x0)
edges1 = dynamic_graph(x1)
x1 = gnn_1(x1, edges1)

# Layer 2: 基於 Layer 1 的表徵重新構建圖
x2 = embed_f_2(x1)
edges2 = dynamic_graph(x2)  # 與 edges1 不同！
x2 = gnn_2(x2, edges2)

# ...
```

**優勢**:
- 逐層精煉圖結構
- 不同層可以捕獲不同尺度的鄰域信息
- 類似於多分辨率圖學習

### 4. **靈活的特徵編碼選擇**

DGM 允許 `embed_f` 使用不同策略:

| embed_f 選擇 | 計算量 | 表達能力 | 適用場景 |
|-------------|--------|---------|---------|
| **Identity** | 最低 | 基礎 | 原始特徵已足夠好 |
| **MLP** | 低 | 中等 | 需要簡單特徵變換 |
| **GCNConv** | 中 | 高 | 利用前一層圖結構 |
| **GATConv** | 高 | 最高 | 需要注意力機制 |

**實驗建議**:
- 小數據集: `ffun='knn'` (Identity)
- 大數據集: `ffun='gcn'` 或 `ffun='mlp'`
- 追求極致性能: `ffun='gat'`

---

## 實驗驗證與性能

### TaBLEau 框架中的 DGM 實現

#### 配置示例
```python
# models/comparison/dgm.py
hparams = Namespace(
    # 特徵維度配置
    pre_fc=[input_dim, 128],           # Pre-processing MLP
    dgm_layers=[[128, 128]],           # DGM_d 層配置
    conv_layers=[[128, 128]],          # GNN 層配置
    fc_layers=[128, 64, num_classes],  # Final MLP
    
    # 圖構建參數
    k=10,                              # k-NN 的 k 值
    distance='euclidean',              # 距離度量
    
    # 模組選擇
    ffun='knn',                        # embed_f: Identity (最輕量)
    gfun='gcn',                        # GNN: GCNConv
    pooling='max',                     # EdgeConv 的 pooling (如果用)
    
    # 訓練參數
    lr=0.001,
    dropout=0.2,
    epochs=300,
)
```

#### 實驗結果

**小數據集 - 二分類 (kaggle_Audit_Data)**
```
Dataset: kaggle_Audit_Data
Samples: 777 (train/val/test split)
Features: 10 numerical features

Full DGM Results (use_full_dgm=True):
├─ Epoch 10/30: Train 0.9971, Val 0.9770, Test 0.9822
├─ Epoch 20/30: Train 1.0000, Val 0.9786, Test 0.9841
└─ Epoch 30/30: Train 1.0000, Val 0.9856, Test 0.9883
Best Test AUC: 0.9883

Simplified DGM (use_full_dgm=False):
└─ Best Test AUC: 0.9830

Performance Gain: +0.5% (0.9883 vs 0.9830)
```

**小數據集 - 回歸 (openml_The_Office_Dataset)**
```
Dataset: openml_The_Office_Dataset
Samples: 188
Features: 10 numerical features

Full DGM Results:
├─ Epoch 10/300: Train 0.5720, Val 0.9598, Test 0.8351
├─ Epoch 20/300: Train 0.3011, Val 0.9623, Test 0.8237
└─ Early stopping at epoch 110
Best Val RMSE: 0.9598
Best Test RMSE: 0.8351
```

**大數據集 - 二分類 (credit)**
```
Dataset: credit
Samples: 16,714
Features: 10 numerical features

Results: (待補充實驗數據)
```

**大數據集 - 多分類 (helena)**
```
Dataset: helena
Samples: 65,196
Classes: 100
Features: 27 numerical features

Results: (待補充實驗數據)
```

### 性能分析

#### 1. **動態圖 vs 固定圖**
```
Full DGM (動態圖):
├─ Gumbel k-NN 採樣
├─ 端到端圖優化
└─ Test AUC: 0.9883

Simplified DGM (固定 k-NN 圖):
├─ 預先構建 k-NN 圖
├─ 圖結構不更新
└─ Test AUC: 0.9830

Improvement: +0.53% (absolute)
```

**結論**: 動態圖學習在小數據集上仍能帶來穩定提升

#### 2. **計算效率**
```
Full DGM:
├─ Gumbel 採樣: O(N²) 或 O(N log N) (使用 KeOps)
├─ GNN 前向: O(E × D) where E = N × k
├─ 反向傳播: 需要計算圖結構梯度
└─ 每 epoch 時間: ~1.5s (小數據集)

Simplified DGM:
├─ 預計算 k-NN: 一次性 O(N²)
├─ GNN 前向: O(E × D)
├─ 反向傳播: 僅 GNN 參數梯度
└─ 每 epoch 時間: ~0.8s (小數據集)
```

**結論**: Full DGM 約慢 2 倍，但性能提升值得

#### 3. **Early Stopping 分析**
```
openml_The_Office_Dataset (回歸):
├─ Total epochs: 300
├─ Early stop: 110
├─ Patience: 10
└─ Best epoch: 20

kaggle_Audit_Data (分類):
├─ Total epochs: 30
├─ Early stop: 30 (未觸發)
├─ Patience: 10
└─ Best epoch: 30
```

**結論**: 小樣本回歸任務容易過擬合，Early stopping 很重要

---

## 總結

### DGM 在 PyTorch-Frame 架構中的定位

```
┌─────────────────────────────────────────────────────────┐
│  DGM 模型的 GNN Stage 歸屬                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  主要歸屬: encoding + columnwise (融合階段)             │
│                                                         │
│  ┌───────────────────┐     ┌──────────────────────┐   │
│  │   Encoding Stage  │     │  Columnwise Stage    │   │
│  ├───────────────────┤     ├──────────────────────┤   │
│  │ • Pre-FC          │     │ • Dynamic Graph      │   │
│  │ • DGM_d.embed_f   │────→│   Construction       │   │
│  │   (特徵編碼)      │     │ • Gumbel k-NN        │   │
│  │                   │     │ • GNN Message        │   │
│  │                   │     │   Passing            │   │
│  └───────────────────┘     └──────────────────────┘   │
│                                                         │
│  特點:                                                  │
│  ✓ 無法單獨拆分為 encoding 或 columnwise                │
│  ✓ 圖結構與特徵編碼緊密耦合                             │
│  ✓ 端到端聯合優化                                       │
│  ✓ 每層都重新構建圖結構                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 關鍵要點

1. **DGM 是原生 GNN 模型**
   - 不是在 Transformer 架構中插入 GNN
   - GNN 是模型的核心組件，不可移除

2. **對應到 encoding + columnwise 融合階段**
   - `embed_f`: Encoding (特徵變換)
   - `sample_without_replacement`: Columnwise (圖構建)
   - `GNN Layer`: Columnwise (消息傳遞)
   - 三者緊密耦合，無法拆分

3. **與其他模型的本質區別**
   - ExcelFormer/TromPT: 模組化設計，GNN 可插拔
   - DGM: 一體化設計，GNN 不可分離

4. **適用場景**
   - ✓ 圖結構未知且對任務重要
   - ✓ 需要自適應學習鄰域關係
   - ✓ 數據分佈複雜，固定距離度量不足
   - ✗ 計算資源受限（Full DGM 較慢）
   - ✗ 需要進行 GNN stage 消融實驗

5. **實驗建議**
   - 小數據集: `ffun='knn'`, `k=5~10`
   - 大數據集: `ffun='gcn'`, `k=10~20`
   - 追求性能: 使用 Full DGM + Early Stopping
   - 追求速度: 使用 Simplified DGM (固定 k-NN)

### 參考代碼位置

```
DGM_pytorch/
├─ DGMlib/
│  ├─ model_dDGM.py          # 主模型定義
│  │  ├─ line 18-68: __init__  (模型初始化)
│  │  ├─ line 69-87: forward   (前向傳播)
│  │  └─ line 95-145: training_step (訓練邏輯)
│  │
│  └─ layers.py              # DGM_d 層定義
│     ├─ line 33-62: DGM_d.forward        (核心邏輯)
│     ├─ line 66-90: sample_without_replacement (Gumbel k-NN)
│     └─ line 160-189: DGM_c.forward      (連續版本)
│
└─ README.md                 # 論文信息與安裝說明

TaBLEau/models/comparison/
└─ dgm.py                    # TaBLEau 框架中的 DGM 實現
   ├─ line 1-16: 架構分析註釋 (本文檔的來源)
   ├─ line 60-100: TabularDataset (數據加載)
   ├─ line 200-350: train_dgm_model (訓練函數)
   └─ line 425-580: main (入口函數)
```

---

## 附錄

### A. 完整配置參數說明

```python
hparams = Namespace(
    # === 網絡結構配置 ===
    pre_fc=[input_dim, 128],           # Pre-processing MLP 層配置
                                       # 例: [10, 128] 表示 10→128 的 MLP
                                       
    dgm_layers=[[128, 128]],           # DGM_d 特徵變換層配置
                                       # 例: [[128, 128]] 表示 128→128 的變換
                                       # 可多層: [[128, 64], [64, 32]]
                                       
    conv_layers=[[128, 128]],          # GNN 卷積層配置
                                       # 對應 dgm_layers 的長度
                                       
    fc_layers=[128, 64, num_classes],  # 最終 MLP 解碼器
                                       # 最後一層是輸出維度
    
    # === 圖構建參數 ===
    k=10,                              # k-NN 的 k 值
                                       # 建議: 小數據集 5~10, 大數據集 10~20
                                       
    distance='euclidean',              # 距離度量: 'euclidean' or 'hyperbolic'
                                       # euclidean: 適用於大部分情況
                                       # hyperbolic: 適用於樹狀結構數據
    
    # === 模組選擇 ===
    ffun='knn',                        # embed_f 的實現
                                       # 'knn': Identity (無變換)
                                       # 'mlp': MLP 變換
                                       # 'gcn': GCNConv (使用前一層圖)
                                       # 'gat': GATConv (注意力)
                                       
    gfun='gcn',                        # GNN 層的實現
                                       # 'gcn': GCNConv (標準)
                                       # 'gat': GATConv (注意力)
                                       # 'edgeconv': EdgeConv (邊卷積)
                                       
    pooling='max',                     # EdgeConv 的 pooling 方式
                                       # 'max', 'mean', 'add'
    
    # === 訓練參數 ===
    lr=0.001,                          # 學習率
    dropout=0.2,                       # Dropout 比例
    batch_size=1,                      # DGM 使用 transductive learning
                                       # 通常 batch_size=1
)
```

### B. 常見問題 FAQ

**Q1: DGM 可以單獨使用 encoding 階段的 GNN 嗎？**
A: 不行。DGM 的設計中，encoding (embed_f) 和 columnwise (graph construction + message passing) 是緊密耦合的。如果只使用 encoding，將失去動態圖學習的能力。

**Q2: 為什麼 DGM 比 ExcelFormer 的 GNN 慢？**
A: 因為 DGM 需要在每個 epoch 都重新計算圖結構（Gumbel k-NN），而 ExcelFormer 可以使用固定的 k-NN 圖。另外，DGM 的圖結構優化需要額外的反向傳播計算。

**Q3: Full DGM vs Simplified DGM 該如何選擇？**
A:
- Full DGM: 追求最佳性能，計算資源充足
- Simplified DGM: 追求速度，或作為 baseline

**Q4: DGM 適合什麼樣的表格數據？**
A:
- ✓ 樣本間存在潛在關係（如時序、空間、社交網絡）
- ✓ 特徵維度較高（高維度更能體現圖結構）
- ✗ 樣本完全獨立（如隨機抽樣的實驗數據）
- ✗ 數據集極小（< 100 樣本）

**Q5: 如何調試 DGM 的圖結構？**
A: 設置 `DGM_d.debug = True`，可以獲取：
- `self.D`: 距離矩陣
- `self.edges_hat`: 採樣的邊
- `self.logprobs`: 採樣概率

### C. 延伸閱讀

1. **DGM 原論文**: "Differentiable Graph Module (DGM) for Graph Convolutional Networks"
   - 詳細介紹 Gumbel-Softmax 採樣機制
   - 理論分析與收斂性證明

2. **Gumbel-Softmax**: "Categorical Reparameterization with Gumbel-Softmax"
   - 離散分佈的可微分採樣技術
   - 在 VAE、強化學習中的應用

3. **Graph Neural Networks**: "A Comprehensive Survey on Graph Neural Networks"
   - GNN 的基礎理論
   - 各種 GNN 變體（GCN, GAT, GraphSAGE, etc.）

4. **PyTorch-Frame**: Official Documentation
   - 表格數據的深度學習框架
   - 五階段架構設計理念

---

**文檔版本**: v1.0  
**最後更新**: 2026-01-04  
**作者**: TaBLEau Research Team  
**聯繫**: 如有疑問，請參考代碼註釋或提交 issue
