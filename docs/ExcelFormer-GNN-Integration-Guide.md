# ExcelFormer + GNN + DGM_d 整合改寫「超詳細」教學（含原始/改動程式碼對照）

本教學提供「原始版本」與「改動後版本」的關鍵程式片段對照，讓 agent 能依照範例，將任一未改裝模型改寫為：
- 自注意力（列間交互）→ 注意力池化（行向量）→ DGM_d 動態圖 → GCN → 自注意力重建（回到列級特徵）
- 支援在 `start`、`materialize`、`encoding`、`columnwise`、`decoding` 階段插入 GNN/DGM
- 在 `decoding` 階段以 GNN 直接作為 decoder（取代 `ExcelFormerDecoder`）

參考實作：請對照目前範例 [models/pytorch_frame/excelformer.py](models/pytorch_frame/excelformer.py)。

---

## 目標與適用範圍
- 讓 agent 在讀到「未改裝」模型時，能套用此教學，完整加上 Self-Attention + DGM_d + GCN 管線。
- 適用於 Torch Frame 風格的表格模型（如 `ExcelFormer`, `FTTransformer`）。

## 前置準備
- 安裝 DGMlib，可用本地路徑引用：
  ```python
  import sys
  sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
  from DGMlib.layers import DGM_d
  ```
- 依任務支援 `binclass`/`multiclass`/`regression`。
- 具備 PyTorch、PyG（`GCNConv`）、Torch Frame。

---

## A. Imports 與裝置選擇（原始 vs 改動）

**原始（示例）**：
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# ...（無 DGMlib 匯入、無通用裝置選擇函式）
```

**改動後**（見 [models/pytorch_frame/excelformer.py](models/pytorch_frame/excelformer.py)）：
```python
import math
import numpy as np
import sys
sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
from DGMlib.layers import DGM_d

def resolve_device(config: dict) -> torch.device:
  gpu_id = config.get('gpu', None)
  if torch.cuda.is_available():
    if gpu_id is not None:
      try:
        return torch.device(f'cuda:{int(gpu_id)}')
      except Exception:
        return torch.device('cuda')
    return torch.device('cuda')
  return torch.device('cpu')
```

重點：
- 加入 DGMlib 匯入與 `sys.path`
- 提供共用 `resolve_device(config)`，後續 materialize/訓練/測試一致使用

---

## B. 基礎模組與工具（原始 vs 改動）

### B.1 `SimpleGCN`
**原始**：
```python
class SimpleGCN(torch.nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super().__init__()
    self.conv1 = GCNConv(in_dim, hidden_dim)
    self.conv2 = GCNConv(hidden_dim, out_dim)

  def forward(self, x, edge_index):
    x = torch.relu(self.conv1(x, edge_index))
    x = self.conv2(x, edge_index)
    return x
```

**改動後**：
```python
class SimpleGCN(torch.nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
    super().__init__()
    self.layers = torch.nn.ModuleList()
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
      self.layers.append(GCNConv(dims[i], dims[i + 1]))

  def forward(self, x, edge_index):
    for i, layer in enumerate(self.layers):
      x = layer(x, edge_index)
      if i < len(self.layers) - 1:
        x = torch.relu(x)
    return x
```

重點：
- 支援可配置層數，易於在不同階段（encoding/columnwise/decoding）選擇輸出維度與深度。

### B.2 `knn_graph`
**原始**：
```python
def knn_graph(x, k):
  x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
  nbrs = NearestNeighbors(n_neighbors=k+1).fit(x_np)
  _, indices = nbrs.kneighbors(x_np)
  edge_index = []
  N = x_np.shape[0]
  for i in range(N):
    for j in indices[i][1:]:
      edge_index.append([i, j])
  edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
  return edge_index
```

**改動後**：
```python
def knn_graph(x, k, directed=False):
  x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
  nbrs = NearestNeighbors(n_neighbors=k+1).fit(x_np)
  _, indices = nbrs.kneighbors(x_np)
  edge_list = []
  N = x_np.shape[0]
  for i in range(N):
    for j in indices[i][1:]:
      edge_list.append([i, j])
      if not directed:
        edge_list.append([j, i])
  if len(edge_list) == 0:
    edge_list = [[i, i] for i in range(N)]
  edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
  return edge_index
```

重點：
- 防止梯度污染（`detach()`）
- 支援雙向邊與自迴路備援

### B.3 新增工具 `_standardize`、`_symmetrize_and_self_loop`
**改動後新增**：
```python
def _standardize(x, dim=0, eps=1e-6):
  mean = x.mean(dim=dim, keepdim=True)
  std = x.std(dim=dim, keepdim=True).clamp_min(eps)
  return (x - mean) / std

def _symmetrize_and_self_loop(edge_index, num_nodes):
  device = edge_index.device
  rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
  self_loops = torch.arange(num_nodes, device=device)
  self_edge = torch.stack([self_loops, self_loops], dim=0)
  ei = torch.cat([edge_index, rev, self_edge], dim=1)
  edge_ids = ei[0] * num_nodes + ei[1]
  unique_ids = torch.unique(edge_ids, sorted=False)
  ei0 = unique_ids // num_nodes
  ei1 = unique_ids % num_nodes
  return torch.stack([ei0, ei1], dim=0)
```

用途：
- 行向量標準化與邊集合穩定化，保證 GCN 圖連通性與一致性。

---

## C. Start 階段管線（原始 vs 改動）

### C.1 原始 `gnn_after_start_fn`（重點片段）
```python
def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  k = config.get('gnn_knn', 5)
  hidden_dim = config.get('gnn_hidden', 64)
  gnn_epochs = config.get('gnn_epochs', 200)
  all_df = pd.concat([train_df, val_df, test_df], axis=0)
  feature_cols = [c for c in all_df.columns if c != 'target']
  x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
  edge_index = knn_graph(x, k).to(device)

  gnn = SimpleGCN(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=x.shape[1]).to(device)
  optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
  for epoch in range(gnn_epochs):
    out = gnn(x, edge_index)
    loss = torch.nn.functional.mse_loss(out, x)
    loss.backward(); optimizer.step()
  final_emb = gnn(x, edge_index).cpu().numpy()
  # 切回 train/val/test 並命名為 N_feature_i
  # ...
  return train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs
```

特性：
- 使用 kNN + MSE 重建輸入特徵（非任務監督）
- 產出新欄位名 `N_feature_i`（與原欄位不同）

### C.2 改動後 `gnn_after_start_fn`（核心片段）
```python
def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
  device = resolve_device(config)
  dgm_k = int(config.get('dgm_k', 10))
  attn_dim = config.get('gnn_attn_dim', config.get('gnn_hidden', 64))
  attn_heads = config.get('gnn_num_heads', 4)
  # 1) Self-Attention（列間交互）+ 注意力池化 產生 row-level 向量
  attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
  attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
  input_proj = torch.nn.Linear(1, attn_dim).to(device)
  column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
  pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

  class DGMEmbedWrapper(torch.nn.Module):
    def forward(self, x, A=None):
      return x
  dgm_module = DGM_d(DGMEmbedWrapper(), k=min(dgm_k, n_train-1), distance=config.get('dgm_distance','euclidean')).to(device)
  gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim).to(device)
  gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
  out_proj = torch.nn.Linear(attn_dim, 1).to(device)

  # forward_pass: Self-Attn → Pool → DGM_d → GCN → Self-Attn 重建 → recon
  # 訓練用任務損失（分類/回歸），含 DGM -logprobs 正則 + 早停（val_loss）
  # 推論分別於 train/val/test 上重建，保留原欄位名與 target
```

改動重點：
- 任務監督（分類/回歸）取代 MSE 重建；更貼近最終目標
- 使用 DGM_d 學習動態圖（非固定 kNN），可與溫度共同訓練
- 自注意力重建回原欄位名（保持下游 transform 一致）
- 早停與最佳權重快照，提升小樣本穩定性

---

## D. Materialize 階段管線（原始 vs 改動）

### D.1 原始 `gnn_after_materialize_fn`（重點片段）
```python
def gnn_after_materialize_fn(train_tf, val_tf, test_tf, config, dataset_name, task_type):
  # TensorFrame → DataFrame → kNN + SimpleGCN(MSE) → DataFrame
  # 重新包裝為 Yandex → materialize → transforms → loaders
  # 返回 (train_loader, val_loader, test_loader, col_stats, mutual_info_sort, dataset, train_tf, val_tf, test_tf, gnn_early_stop_epochs)
```

### D.2 改動後 `gnn_after_materialize_fn`（核心片段）
```python
def gnn_after_materialize_fn(train_tf, val_tf, test_tf, config, dataset_name, task_type):
  device = resolve_device(config)
  # TensorFrame → DataFrame
  # Self-Attention → Pool → DGM_d → GCN → Self-Attention 重建
  # 重建的 DataFrame 保留原欄位名 → Yandex → materialize → transforms → loaders
  # 同樣使用任務監督訓練 + 早停（val_loss）+ 恢復最佳權重
```

改動重點：
- 與 Start 一致的動態圖與注意力重建；統一訓練信號與穩定策略

---

## E. Core 函式（forward）插入邏輯（原始 vs 改動）

### E.1 原始 `excelformer_core_fn`（重點片段）
```python
def excelformer_core_fn(material_outputs, config, task_type, gnn_stage=None):
  # 建立 encoder/convs/decoder
  class SimpleGCN_INTERNAL(nn.Module):
    def __init__(self, in_channels, out_channels):
      super().__init__()
      self.conv1 = GCNConv(in_channels, out_channels)
    def forward(self, x, edge_index):
      return self.conv1(x, edge_index)
  gnn = SimpleGCN_INTERNAL(channels, channels).to(device)

  def model_forward(tf, mixup_encoded=False):
    x, _ = encoder(tf)
    if gnn_stage == 'encoding':
      # 將 x reshape，建立完全圖 edge_index（每列相互連接），用 gnn 更新 x
    for conv in convs:
      x = conv(x)
    if gnn_stage == 'columnwise':
      # 同上在 convs 後做一次
    out = decoder(x)
    return out, y_mixedup
```

特性：
- encoding/columnwise 的 GNN 為簡單一層，且未包含 Self-Attention、DGM、FFN、殘差融合
- decoding 階段使用外部 `gnn_decoding_eval` 而非 forward 內直接作為 decoder

### E.2 改動後 `excelformer_core_fn`（核心片段）
```python
def excelformer_core_fn(material_outputs, config, task_type, gnn_stage=None):
  # 依 gnn_stage 構建：Self-Attention（含 PreNorm+殘差 + FFN）+ DGM_d + GCN + Self-Attention 解碼
  # encoding/columnwise：在對應階段插入完整管線，並以 fusion_alpha 殘差融合回 x
  # decoding：跳過 ExcelFormerDecoder，改用 Self-Attention + Pool + DGM_d + GCN 直接輸出 out

  def model_forward(tf, mixup_encoded=False):
    x, _ = encoder(tf)

    if gnn_stage == 'encoding':
      # Self-Attn → Pool → DGM_d → GCN → Self-Attn 解碼 → fusion 殘差回 x

    for conv in convs:
      x = conv(x)

    if gnn_stage == 'columnwise':
      # Self-Attn → Pool → DGM_d → GCN → Self-Attn 解碼 → fusion 殘差回 x

    if gnn_stage == 'decoding':
      # Self-Attn → Pool → DGM_d → GCN(as decoder) → out
    else:
      out = decoder(x)
    return out, y_mixedup
```

改動重點：
- 完整加入 Self-Attention + FFN + 殘差 + DGM_d + GCN + Self-Attention 解碼的雙注意力結構
- `decoding` 階段直接用 GNN 作為 decoder（forward 即出預測）
- Optimizer 包含所有新增參數（attention, FFN, DGM, fusion_alpha, pool_query 等）
- 訓練/測試以 `val_loss` 早停、快照最佳權重、並恢復（含 stage-specific 模組）

---

## F. 刪除/合併的原始輔助

**原始：**
- `get_all_embeddings_and_targets(...)`
- `gnn_decoding_eval(...)`

**改動後：**
- 直接於 `model_forward` 的 `decoding` 階段使用 GNN-as-decoder，不再需要外部解碼評估函式。

---

## G. Materialize 與 Main 差異（原始 vs 改動）

**Materialize 原始**：直接 `torch.device('cuda' if ...)`，較分散的裝置控制。

**Materialize 改動後**：
- 使用 `resolve_device(config)` 一致選擇裝置並記錄在 material 輸出，供 encoder/convs/gnn/decoder/metrics 共用。

**Main**：
- 邏輯相同（支援 `gnn_stage` 為 `start/materialize/...`）
- `gnn_early_stop_epochs` 由各注入函式返回並合併到結果中。

---

## H. 配置鍵與參數（建議）
- 裝置：`gpu`
- DGM：`dgm_k`、`dgm_distance`
- 注意力/GCN 維度：`channels`、`gnn_attn_dim`、`gnn_hidden`、`gnn_out_dim`、`gnn_num_heads`
- 優化與訓練：`lr`、`gamma`、`gnn_lr`、`epochs`、`patience`、`loss_threshold`、`gnn_patience`、`gnn_loss_threshold`、`gnn_dropout`
- Mixup：`mixup`、`beta`

注意：`dgm_k` 在每個 split/batch 需安全裁剪：`k = min(dgm_k, Ns-1)`。

---

## I. 移植 Checklist（逐項核對）
1. 增加 imports 與 `resolve_device(config)`。
2. 加入 `_standardize`、`_symmetrize_and_self_loop`、（可選）`compute_adaptive_dgm_k`。
3. 在 `start/materialize` 插入 `gnn_after_*` 管線（DataFrame vs TensorFrame 輸入差異）。
4. 改造 core forward：依 `gnn_stage` 插入 Self-Attn + DGM_d + GCN，並在 encoding/columnwise 做殘差融合；decoding 用 GNN-as-decoder。
5. Optimizer 納入 attention/FFN/DGM/GNN/decoder 等所有參數。
6. 訓練/測試：`test` 回傳 `(metric, avg_loss)`；以 `val_loss` 早停並恢復快照（含 stage-specific 模組）。
7. 保持輸出 schema 與 `target`，保障 transforms/mutual-info 排序一致性。

---

## J. 驗證與執行（示例）
```bash
python main.py --dataset eye --models excelformer --gnn_stages all --epochs 300
```
或測試單一階段：
```bash
python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages start --epochs 200
python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages encoding --epochs 200
python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages columnwise --epochs 200
python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages decoding --epochs 200
```

---

## K. 小結與建議
- 核心結構：Self-Attn + Pool + DGM_d + GCN + Self-Attn 重建。
- `decoding` 階段以 GNN 作為 decoder，避免雙重解碼路徑。
- 早停以 `val_loss` 為準，並恢復快照，顯著提升穩定性（特別是小樣本）。

如需我直接依本教學改裝指定模型（如 `models/pytorch_frame/fttransformer.py`），我可以生成 patch 並執行基本測試。