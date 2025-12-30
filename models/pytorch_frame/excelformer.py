from __future__ import annotations
"""Reported (reproduced) accuracy (for multi-classification task), auc
(for binary classification task) and rmse (for regression task)
based on Table 1 of the paper https://arxiv.org/abs/2301.02819.
ExcelFormer uses the same train-validation-test split as the Yandex paper.
The reproduced results are based on Z-score Normalization, and the
reported ones are based on :class:`QuantileTransformer` preprocessing
in the Sklearn Python package. The above preprocessing is applied
to numerical features.

california_housing: 0.4587 (0.4550) mixup: feature, num_layers: 3,
gamma: 1.00, epochs: 300
jannis : 72.51 (72.80) mixup: feature
covtype: 97.17 (97.02) mixup: hidden
helena: 38.20 (37.68) mixup: feature
higgs_small: 80.75 (79.27) mixup: hidden
"""
import argparse
import os.path as osp
import os
import math
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame.data.loader import DataLoader
from torch_frame.datasets.yandex import Yandex
from torch_frame.nn import ExcelFormer
from torch_frame.transforms import CatToNumTransform, MutualInformationSort
from tqdm import tqdm
import json
import os
import pandas as pd
import xlsxwriter
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList
import torch_frame
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder
from torch_frame.nn.encoder.stype_encoder import ExcelFormerEncoder
from torch_frame.nn.encoder.stypewise_encoder import (
    StypeEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.typing import NAStrategy

import torch
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import sys
import numpy as np
sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
from DGMlib.layers import DGM_d

# 統一的裝置選擇函式：從 config 選擇 GPU，否則回退到可用裝置
def resolve_device(config: dict) -> torch.device:
    gpu_id = None
    try:
        gpu_id = config.get('gpu', None)
    except Exception:
        gpu_id = None
    if torch.cuda.is_available():
        if gpu_id is not None:
            try:
                device = torch.device(f'cuda:{int(gpu_id)}')
                print(f"[DEVICE] resolve_device: Using cuda:{int(gpu_id)} (from config['gpu']={gpu_id})")
                return device
            except Exception:
                device = torch.device('cuda')
                print(f"[DEVICE] resolve_device: Using cuda (default, config['gpu']={gpu_id} invalid)")
                return device
        device = torch.device('cuda')
        print(f"[DEVICE] resolve_device: Using cuda (default, no config['gpu'] specified)")
        return device
    print(f"[DEVICE] resolve_device: Using cpu (CUDA not available)")
    return torch.device('cpu')

def compute_adaptive_dgm_k(num_samples, num_features, dataset_name='', use_config_override=True):
    """
    根據數據集規模自動計算最合適的 dgm_k（DGM 候選池大小）。
    
    核心原理：
    - DGM_k 定義了候選池大小（能從多少個鄰居中選擇）
    - 實際連接邊數 = k 個候選中溫度參數學習選出的邊
    - 為避免過度稀疏，需要平衡：
        * 候選池足夠大（學習空間充足）
        * 避免過度密集（計算成本、過度擬合）
    
    策略：
    1. 基礎規則：k ≈ sqrt(N)，介於 5-20 之間
    2. 特徵維度調整：高維數據需要更多候選以捕捉多重相似性
    3. 樣本密度調整：樣本少時適當增加相對鄰居數
    
    Args:
        num_samples: 數據集樣本總數
        num_features: 特徵維度
        dataset_name: 數據集名稱（用於特殊規則）
        use_config_override: 是否允許配置文件覆蓋自動計算
    
    Returns:
        adaptive_k (int): 推薦的 dgm_k 值
        
    Examples:
        compute_adaptive_dgm_k(776, 20)   # ~14 (平衡稀疏與連通性)
        compute_adaptive_dgm_k(10000, 50) # ~18 (大數據集)
        compute_adaptive_dgm_k(200, 30)   # ~16 (小數據集，多特徵)
    """
    
    # 基礎公式：k ≈ sqrt(N) 作為起點
    base_k = int(np.sqrt(num_samples))
    
    # 特徵維度修正：高維時需更多候選來探索不同相似性角度
    feature_factor = 1.0 + np.log1p(num_features) / 10  # log 增長，避免過度增加
    adjusted_k = int(base_k * feature_factor)
    
    # 樣本密度修正：樣本少時 relative connectivity 要高
    # N=200 時，比 N=5000 的相對鄰居比例應該更高
    density_factor = 1.0
    if num_samples < 500:
        # 小數據集：相對鄰居比應更高以維持連通性
        density_factor = 1.3  # +30%
    elif num_samples > 5000:
        # 大數據集：可適度降低相對比例（計算成本考量）
        density_factor = 0.9  # -10%
    
    adaptive_k = int(adjusted_k * density_factor)
    
    # 硬性邊界：防止過度稀疏或過度密集
    # - 下限 5：太小容易斷連
    # - 上限 30：大於 30 計算成本顯著增加，邊際效用降低
    # - 上限動態調整：sqrt(N) 的 3-4 倍作為合理上界
    upper_limit = min(30, max(15, int(4 * np.sqrt(num_samples))))
    if num_samples < 1000:
        upper_limit = min(20, int(3 * np.sqrt(num_samples)))
    adaptive_k = max(5, min(adaptive_k, upper_limit))
    
    print(f"[DGM-K] Adaptive Calculation:")
    print(f"  - Dataset: {dataset_name if dataset_name else 'unknown'} | N={num_samples}, D={num_features}")
    print(f"  - Base k (√N): {base_k}")
    print(f"  - Feature Factor: {feature_factor:.2f}x (features={num_features})")
    print(f"  - Density Factor: {density_factor:.2f}x (samples={num_samples})")
    print(f"  - Adaptive k: {adaptive_k} (valid range: [5, {upper_limit}])")
    
    return adaptive_k



class SimpleGCN(torch.nn.Module):
    """簡化的 GCN，用於聯合訓練"""
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

def knn_graph(x, k, directed=False):
    """
    構建 k-NN 圖
    Args:
        x: [N, feat_dim] 特徵矩陣（可以有梯度）
        k: 鄰居數
        directed: 是否單向邊（True）或雙向邊（False）
    Returns:
        edge_index: [2, E] 邊索引
    """
    # ✅ 修復：detach() 確保沒有梯度信息
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_list = []
    N = x_np.shape[0]
    for i in range(N):
        for j in indices[i][1:]:  # 跳過自己
            edge_list.append([i, j])
            if not directed:
                edge_list.append([j, i])  # 添加反向邊（雙向）
    if len(edge_list) == 0:
        edge_list = [[i, i] for i in range(N)]  # 備用：自迴圈
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index



    
# 統一的圖前處理與輔助函式（不依資料集調參）
def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    """沿指定維度做 z-score 標準化。"""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std
    
    
def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """對 edge_index 做對稱化並加入自迴路，移除重複邊。"""
    device = edge_index.device
    # 加入反向邊
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    # 加入自迴路
    self_loops = torch.arange(num_nodes, device=device)
    self_edge = torch.stack([self_loops, self_loops], dim=0)
    # 合併
    ei = torch.cat([edge_index, rev, self_edge], dim=1)
    # 去重：將 (i,j) 映射為唯一 id，並以唯一 id 重建邊
    edge_ids = ei[0] * num_nodes + ei[1]
    unique_ids = torch.unique(edge_ids, sorted=False)
    ei0 = unique_ids // num_nodes
    ei1 = unique_ids % num_nodes
    ei_unique = torch.stack([ei0, ei1], dim=0)
    return ei_unique


def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    """
    在 start_fn 和 materialize_fn 之間插入「自注意力 → DGM動態圖 → 自注意力」的管線。

    流程：
    1) 針對每筆樣本（row），先以 Multi-Head Self-Attention 在欄位維度做上下文建模，
       得到每列的 contextualized column tokens，並透過注意力池化得到 row-level 向量。
    2) 以 row-level 向量作為節點特徵，用DGM動態圖模組（DGM_d）學習最優的圖結構，
       並用GCN訓練（監督式 loss 與任務一致）。DGM的temperature參數和GCN權重一同反向傳播。
    3) 將 GCN 輸出的 row-level embedding 再注入第二個 self-attention，重建回
       [num_rows, num_cols] 形狀的特徵矩陣，回傳與原欄位數一致的 DataFrame。
    """

    # 根據 config 選擇 GPU 編號（如 gpu=0 或 gpu=1）；若不可用則回退到 CPU
    gpu_id = config.get('gpu', None)
    if torch.cuda.is_available() and gpu_id is not None:
        try:
            device = torch.device(f'cuda:{int(gpu_id)}')
        except Exception:
            device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[START-GNN-DGM] Using device: {device}")

    
    # 參數設定：統一固定 dgm_k（為公平比較），並在後續以樣本數做安全上限
    dgm_k = int(config.get('dgm_k', 10))
    print(f"[DGM-K] Using fixed dgm_k={dgm_k} (no dataset-dependent adapt)")
    
    dgm_distance = config.get('dgm_distance', 'euclidean')  # euclidean 或 hyperbolic
    gnn_epochs = config.get('epochs', 200)
    patience = config.get('gnn_patience', 10)
    loss_threshold = config.get('gnn_loss_threshold', 1e-4)
    attn_dim = config.get('gnn_attn_dim', config.get('gnn_hidden', 64))
    gnn_hidden = config.get('gnn_hidden', 64)
    gnn_out_dim = config.get('gnn_out_dim', attn_dim)
    attn_heads = config.get('gnn_num_heads', 4)
    lr = config.get('gnn_lr', 1e-3)

    # 合併三個df（僅用於列名與拼接便利）
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    feature_cols = [c for c in all_df.columns if c != 'target']
    num_cols = len(feature_cols)
    # 各 split 的張量（統一 train-only 訓練、inductive 推論）
    x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
    y_train_np = train_df['target'].values
    y_val_np = val_df['target'].values
    y_test_np = test_df['target'].values

    # 自動計算 num_classes（統一使用 train/val/test 連結）與 out_dim
    if task_type in ['binclass', 'multiclass']:
        y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
        num_classes = len(pd.unique(y_all_np))
        if task_type == 'binclass' and num_classes != 2:
            num_classes = 2
        out_dim = num_classes
    else:
        out_dim = 1

    # 尺寸
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    # 模組（固定規則 + DGM 自適應）：訓練以 train-only 節點建圖
    attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    input_proj = torch.nn.Linear(1, attn_dim).to(device)
    
    # ✅ 改動：使用DGM_d動態圖模組代替靜態kNN
    # DGM_d 內部可學習的 temperature 會影響圖的邊，並通過loss反向傳播更新
    # DGM_d 的 embed_f 需要接受 (x, A) 兩個參數
    class DGMEmbedWrapper(torch.nn.Module):
        def forward(self, x, A=None):
            # 直接返回輸入，不做額外處理
            return x
    
    dgm_embed_f = DGMEmbedWrapper()
    # k 以訓練節點數為上限；統一規則但避免非法
    dgm_k_train = int(min(dgm_k, max(1, n_train - 1)))
    dgm_module = DGM_d(dgm_embed_f, k=dgm_k_train, distance=dgm_distance).to(device)
    
    gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim).to(device)
    gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
    pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)
    out_proj = torch.nn.Linear(attn_dim, 1).to(device)

    # 可學習的欄位 embedding 與 pooling query
    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    params = list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters()) \
        + list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters()) \
        + list(out_proj.parameters()) + list(dgm_module.parameters()) + [column_embed, pool_query]

    optimizer = torch.optim.Adam(params, lr=lr)

    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    # 早停最佳權重快照
    best_states = None

    def forward_pass(x_tensor, dgm_module_inst):
        # x_tensor: [Ns, num_cols]，針對單一 split（train/val/test）
        Ns = x_tensor.shape[0]
        x_in = input_proj(x_tensor.unsqueeze(-1))  # [Ns, num_cols, attn_dim]
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        # 注意力池化 → row-level 向量
        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_weights = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)  # [Ns, attn_dim]

        # z-score 標準化（沿樣本維度）
        row_emb_std = _standardize(row_emb, dim=0)

        # DGM_d 動態圖（拆分批次格式）
        row_emb_batched = row_emb_std.unsqueeze(0)  # [1, Ns, attn_dim]
        # 針對當前 split 安全的 k（避免 Ns 太小）
        dgm_k_split = int(min(dgm_k, max(1, Ns - 1)))
        # 若 split 尺寸不同於訓練，臨時以相同設定建立子模組（權重來自 dgm_module_inst）
        # 注意：DGM_d 的可學習溫度在 module 內；此處沿用同一 module 實例即可
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)  # [Ns, attn_dim]

        # 邊對稱化 + 自迴路
        edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)
        
        gcn_out = gnn(row_emb_dgm, edge_index_dgm)  # [Ns, gnn_out_dim]
        logits = pred_head(gcn_out)

        # 將 row embedding 注入第二個 self-attention，再解碼回欄位尺度
        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)  # [Ns,1,attn_dim]
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_with_ctx, tokens_with_ctx, tokens_with_ctx)
        recon = out_proj(tokens_out).squeeze(-1)  # [Ns, num_cols]
        
        # 簡易圖統計日誌
        E = edge_index_dgm.shape[1]
        avg_deg = E / max(1, Ns)
        print(f"[START-GNN-DGM] split Ns={Ns}, edges={E}, avg_deg={avg_deg:.2f}")
        
        return logits, recon, logprobs_dgm

    # ✅ 訓練時設定DGM_d為training模式，以啟用採樣機制
    for epoch in range(gnn_epochs):
        # ========== 訓練階段 ==========
        dgm_module.train()
        gnn.train()
        attn_in.train(); attn_out.train(); input_proj.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()

        optimizer.zero_grad()
        logits, _, logprobs_dgm = forward_pass(x_train, dgm_module)

        if task_type in ['binclass', 'multiclass']:
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            train_loss = F.cross_entropy(logits, y_train)
        else:
            y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
            train_loss = F.mse_loss(logits.squeeze(), y_train)

        # ✅ 改動：可選地加入DGM的logprobs正則項，使邊更穩定
        dgm_reg = -logprobs_dgm.mean() * 0.01  # 負logprob越小越好，傾向確定的邊
        train_loss = train_loss + dgm_reg

        train_loss.backward()
        optimizer.step()

        # ========== 驗證階段（用於早停判定） ==========
        dgm_module.eval()
        gnn.eval()
        attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
        
        with torch.no_grad():
            logits_val, _, _ = forward_pass(x_val, dgm_module)
            if task_type in ['binclass', 'multiclass']:
                y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
                val_loss = F.cross_entropy(logits_val, y_val)
            else:
                y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
                val_loss = F.mse_loss(logits_val.squeeze(), y_val)

        train_loss_val = train_loss.item()
        val_loss_val = val_loss.item()
        
        # ✅ 改動：基於 validation loss 判定是否改進（避免過擬合）
        improved = val_loss_val < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss_val
            early_stop_counter = 0
            # 保存所有相關模組的最佳權重
            best_states = {
                'attn_in': attn_in.state_dict(),
                'attn_out': attn_out.state_dict(),
                'input_proj': input_proj.state_dict(),
                'gnn': gnn.state_dict(),
                'gcn_to_attn': gcn_to_attn.state_dict(),
                'pred_head': pred_head.state_dict(),
                'out_proj': out_proj.state_dict(),
                'dgm_module': dgm_module.state_dict(),
                'column_embed': column_embed.detach().clone(),
                'pool_query': pool_query.detach().clone(),
            }
        else:
            early_stop_counter += 1

        print(f"[START-GNN-DGM] Epoch {epoch+1}/{gnn_epochs}, Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}{' ↓ (improved)' if improved else ''}")
        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"[START-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})")
            break

    # 恢復早停最佳權重（若有）
    if best_states is not None:
        attn_in.load_state_dict(best_states['attn_in'])
        attn_out.load_state_dict(best_states['attn_out'])
        input_proj.load_state_dict(best_states['input_proj'])
        gnn.load_state_dict(best_states['gnn'])
        gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
        pred_head.load_state_dict(best_states['pred_head'])
        out_proj.load_state_dict(best_states['out_proj'])
        dgm_module.load_state_dict(best_states['dgm_module'])
        with torch.no_grad():
            column_embed.copy_(best_states['column_embed'])
            pool_query.copy_(best_states['pool_query'])

    # 推論：各 split 獨立建圖並重建（inductive）
    dgm_module.eval()
    gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
    with torch.no_grad():
        _, recon_train, _ = forward_pass(x_train, dgm_module)
        _, recon_val, _ = forward_pass(x_val, dgm_module)
        _, recon_test, _ = forward_pass(x_test, dgm_module)
        train_emb = recon_train.cpu().numpy()
        val_emb = recon_val.cpu().numpy()
        test_emb = recon_test.cpu().numpy()

    train_df_gnn = pd.DataFrame(train_emb, columns=feature_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(val_emb, columns=feature_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(test_emb, columns=feature_cols, index=test_df.index)

    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values

    return train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs







def gnn_after_materialize_fn(train_tensor_frame, val_tensor_frame, test_tensor_frame, config, dataset_name, task_type):
    """
    在 materialize_fn 之後插入「自注意力 → DGM動態圖 → GCN → 自注意力」的管線。
    
    與 gnn_after_start_fn 相似的架構：
    1) 從 TensorFrame 提取特徵，以 Multi-Head Self-Attention 在欄位維度做上下文建模
    2) 注意力池化得到 row-level 向量，用 DGM_d 學習動態圖結構
    3) GCN 處理圖並訓練（監督式 loss，基於 val_loss 早停並恢復最佳權重）
    4) 將 GCN 輸出注入第二個 self-attention，重建回 [num_rows, num_cols] 特徵矩陣
    5) 返回重建的 DataFrame，保持原始欄位數一致
    """
    
    device = resolve_device(config)
    print(f"[MATERIALIZE-GNN-DGM] Using device: {device}")
    
    # 從 TensorFrame 提取 DataFrame
    def tensor_frame_to_df(tensor_frame):
        col_names = tensor_frame.col_names_dict[stype.numerical]
        features = tensor_frame.feat_dict[stype.numerical].cpu().numpy()
        df = pd.DataFrame(features, columns=col_names)
        if hasattr(tensor_frame, 'y') and tensor_frame.y is not None:
            df['target'] = tensor_frame.y.cpu().numpy()
        return df
    
    train_df = tensor_frame_to_df(train_tensor_frame)
    val_df = tensor_frame_to_df(val_tensor_frame)
    test_df = tensor_frame_to_df(test_tensor_frame)
    
    # 合併僅為取得欄位資訊；統一 train-only 訓練，inductive 推論
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    feature_cols = [c for c in all_df.columns if c != 'target']
    num_cols = len(feature_cols)
    
    # 統一固定 dgm_k（可由 config 指定），並以訓練節點數為上限避免非法
    if 'dgm_k' in config:
        dgm_k = config['dgm_k']
        print(f"[DGM-K] Using user-specified dgm_k={dgm_k}")
    else:
        dgm_k = 10
    
    # 參數設定
    dgm_distance = config.get('dgm_distance', 'euclidean')
    gnn_epochs = config.get('epochs', 200)
    patience = config.get('gnn_patience', 10)
    loss_threshold = config.get('gnn_loss_threshold', 1e-4)
    attn_dim = config.get('gnn_attn_dim', config.get('gnn_hidden', 64))
    gnn_hidden = config.get('gnn_hidden', 64)
    gnn_out_dim = config.get('gnn_out_dim', attn_dim)
    attn_heads = config.get('gnn_num_heads', 4)
    lr = config.get('gnn_lr', 1e-3)
    
    # 準備各 split 特徵和標籤
    x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
    y_train_np = train_df['target'].values
    y_val_np = val_df['target'].values
    y_test_np = test_df['target'].values
    
    # 自動計算 num_classes（統一使用 train/val/test 連結）與 out_dim
    if task_type in ['binclass', 'multiclass']:
        y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
        num_classes = len(pd.unique(y_all_np))
        if task_type == 'binclass' and num_classes != 2:
            num_classes = 2
        print(f"[MATERIALIZE-GNN-DGM] Detected num_classes: {num_classes}")
        out_dim = num_classes
    else:
        out_dim = 1
    
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    
    # 模組
    attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    input_proj = torch.nn.Linear(1, attn_dim).to(device)
    
    # DGM_d 動態圖模組
    class DGMEmbedWrapper(torch.nn.Module):
        def forward(self, x, A=None):
            return x
    
    dgm_embed_f = DGMEmbedWrapper()
    dgm_k_train = int(min(dgm_k, max(1, n_train - 1)))
    dgm_module = DGM_d(dgm_embed_f, k=dgm_k_train, distance=dgm_distance).to(device)
    
    gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim).to(device)
    gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
    pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)
    out_proj = torch.nn.Linear(attn_dim, 1).to(device)
    
    # 可學習的欄位 embedding 與 pooling query
    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))
    
    params = list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters()) \
        + list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters()) \
        + list(out_proj.parameters()) + list(dgm_module.parameters()) + [column_embed, pool_query]
    
    optimizer = torch.optim.Adam(params, lr=lr)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_states = None
    
    def forward_pass(x_tensor, dgm_module_inst):
        # x_tensor: [Ns, num_cols]
        Ns = x_tensor.shape[0]
        x_in = input_proj(x_tensor.unsqueeze(-1))  # [Ns, num_cols, attn_dim]
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)
        
        # 注意力池化 → row-level 向量
        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_weights = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)  # [Ns, attn_dim]
        
        # 標準化
        row_emb_std = _standardize(row_emb, dim=0)
        
        # DGM_d 動態圖
        row_emb_batched = row_emb_std.unsqueeze(0)  # [1, Ns, attn_dim]
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)  # [Ns, attn_dim]
        
        # 邊對稱化 + 自迴路
        edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)
        
        gcn_out = gnn(row_emb_dgm, edge_index_dgm)  # [Ns, gnn_out_dim]
        logits = pred_head(gcn_out)
        
        # 將 row embedding 注入第二個 self-attention，再解碼回欄位尺度
        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)  # [Ns, 1, attn_dim]
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_with_ctx, tokens_with_ctx, tokens_with_ctx)
        recon = out_proj(tokens_out).squeeze(-1)  # [Ns, num_cols]
        
        # 圖統計
        E = edge_index_dgm.shape[1]
        avg_deg = E / max(1, Ns)
        print(f"[MATERIALIZE-GNN-DGM] split Ns={Ns}, edges={E}, avg_deg={avg_deg:.2f}")
        
        return logits, recon, logprobs_dgm
    
    # 訓練循環
    for epoch in range(gnn_epochs):
        # 訓練階段
        dgm_module.train()
        gnn.train()
        attn_in.train(); attn_out.train(); input_proj.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()
        
        optimizer.zero_grad()
        logits, _, logprobs_dgm = forward_pass(x_train, dgm_module)
        
        if task_type in ['binclass', 'multiclass']:
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            train_loss = F.cross_entropy(logits, y_train)
        else:
            y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
            train_loss = F.mse_loss(logits.squeeze(), y_train)
        
        # DGM 正則項
        dgm_reg = -logprobs_dgm.mean() * 0.01
        train_loss = train_loss + dgm_reg
        
        train_loss.backward()
        optimizer.step()
        
        # 驗證階段
        dgm_module.eval()
        gnn.eval()
        attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
        
        with torch.no_grad():
            logits_val, _, _ = forward_pass(x_val, dgm_module)
            
            if task_type in ['binclass', 'multiclass']:
                y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
                val_loss = F.cross_entropy(logits_val, y_val)
            else:
                y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
                val_loss = F.mse_loss(logits_val.squeeze(), y_val)
        
        train_loss_val = train_loss.item()
        val_loss_val = val_loss.item()
        
        # 早停判定（基於 val_loss）
        improved = val_loss_val < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss_val
            early_stop_counter = 0
            # 保存最佳權重
            best_states = {
                'attn_in': attn_in.state_dict(),
                'attn_out': attn_out.state_dict(),
                'input_proj': input_proj.state_dict(),
                'gnn': gnn.state_dict(),
                'gcn_to_attn': gcn_to_attn.state_dict(),
                'pred_head': pred_head.state_dict(),
                'out_proj': out_proj.state_dict(),
                'dgm_module': dgm_module.state_dict(),
                'column_embed': column_embed.detach().clone(),
                'pool_query': pool_query.detach().clone(),
            }
        else:
            early_stop_counter += 1
        
        print(f"[MATERIALIZE-GNN-DGM] Epoch {epoch+1}/{gnn_epochs}, Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}{' ↓ (improved)' if improved else ''}")
        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"[MATERIALIZE-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})")
            break
    
    # 恢復最佳權重
    if best_states is not None:
        attn_in.load_state_dict(best_states['attn_in'])
        attn_out.load_state_dict(best_states['attn_out'])
        input_proj.load_state_dict(best_states['input_proj'])
        gnn.load_state_dict(best_states['gnn'])
        gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
        pred_head.load_state_dict(best_states['pred_head'])
        out_proj.load_state_dict(best_states['out_proj'])
        dgm_module.load_state_dict(best_states['dgm_module'])
        with torch.no_grad():
            column_embed.copy_(best_states['column_embed'])
            pool_query.copy_(best_states['pool_query'])
    
    # 推論：各 split 獨立建圖並重建
    dgm_module.eval()
    gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
    with torch.no_grad():
        _, recon_train, _ = forward_pass(x_train, dgm_module)
        _, recon_val, _ = forward_pass(x_val, dgm_module)
        _, recon_test, _ = forward_pass(x_test, dgm_module)
        train_emb = recon_train.cpu().numpy()
        val_emb = recon_val.cpu().numpy()
        test_emb = recon_test.cpu().numpy()
    
    train_df_gnn = pd.DataFrame(train_emb, columns=feature_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(val_emb, columns=feature_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(test_emb, columns=feature_cols, index=test_df.index)
    
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values
    
    # 重新包裝為 Yandex 數據集並 materialize
    dataset = Yandex(train_df_gnn, val_df_gnn, test_df_gnn, name=dataset_name, task_type=task_type)
    dataset.materialize()
    
    # split tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]
    
    # 類別特徵轉換
    categorical_transform = CatToNumTransform()
    categorical_transform.fit(train_tensor_frame, dataset.col_stats)
    train_tensor_frame = categorical_transform(train_tensor_frame)
    val_tensor_frame = categorical_transform(val_tensor_frame)
    test_tensor_frame = categorical_transform(test_tensor_frame)
    col_stats = categorical_transform.transformed_stats
    
    # 基於互信息的特徵排序
    mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
    mutual_info_sort.fit(train_tensor_frame, col_stats)
    train_tensor_frame = mutual_info_sort(train_tensor_frame)
    val_tensor_frame = mutual_info_sort(val_tensor_frame)
    test_tensor_frame = mutual_info_sort(test_tensor_frame)
    
    # 創建數據加載器
    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)
    
    return (train_loader, val_loader, test_loader,
            col_stats, mutual_info_sort,
            dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs)



def feature_mixup(
    x: Tensor,
    y: Tensor,
    num_classes: int,
    beta: float | Tensor = 0.5,
    mixup_type: str | None = None,
    mi_scores: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
    r"""Mixup input numerical feature tensor :obj:`x` by swapping some
    feature elements of two shuffled sample samples. The shuffle rates for
    each row is sampled from the Beta distribution. The target `y` is also
    linearly mixed up.

    Args:
        x (Tensor): The input numerical feature.
        y (Tensor): The target.
        num_classes (int): Number of classes.
        beta (float): The concentration parameter of the Beta distribution.
            (default: :obj:`0.5`)
        mixup_type (str, optional): The mixup methods. No mixup if set to
            :obj:`None`, options `feature` and `hidden` are `FEAT-MIX`
            (mixup at feature dimension) and `HIDDEN-MIX` (mixup at
            hidden dimension) proposed in ExcelFormer paper.
            (default: :obj:`None`)
        mi_scores (Tensor, optional): Mutual information scores only used in
            the mixup weight calculation for `FEAT-MIX`.
            (default: :obj:`None`)

    Returns:
        x_mixedup (Tensor): The mixedup numerical feature.
        y_mixedup (Tensor): Transformed target of size
            :obj:`[batch_size, num_classes]`
    """
    assert num_classes > 0
    assert mixup_type in [None, 'feature', 'hidden']

    beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
    beta_distribution = torch.distributions.beta.Beta(beta, beta)
    shuffle_rates = beta_distribution.sample(torch.Size((len(x), 1)))
    shuffled_idx = torch.randperm(len(x), device=x.device)
    assert x.ndim == 3, """
    FEAT-MIX or HIDDEN-MIX is for encoded numerical features
    of size [batch_size, num_cols, in_channels]."""
    b, f, d = x.shape
    if mixup_type == 'feature':
        assert mi_scores is not None
        mi_scores = mi_scores.to(x.device)
        # Hard mask (feature dimension)
        mixup_mask = torch.rand(torch.Size((b, f)),
                                device=x.device) < shuffle_rates
        # L1 normalized mutual information scores
        norm_mi_scores = mi_scores / mi_scores.sum()
        # Mixup weights
        lam = torch.sum(
            norm_mi_scores.unsqueeze(0) * mixup_mask, dim=1, keepdim=True)
        mixup_mask = mixup_mask.unsqueeze(2)
    elif mixup_type == 'hidden':
        # Hard mask (hidden dimension)
        mixup_mask = torch.rand(torch.Size((b, d)),
                                device=x.device) < shuffle_rates
        mixup_mask = mixup_mask.unsqueeze(1)
        # Mixup weights
        lam = shuffle_rates
    else:
        # No mixup
        mixup_mask = torch.ones_like(x, dtype=torch.bool)
        # Fake mixup weights
        lam = torch.ones_like(shuffle_rates)
    x_mixedup = mixup_mask * x + ~mixup_mask * x[shuffled_idx]

    y_shuffled = y[shuffled_idx]
    if num_classes == 1:
        # Regression task or binary classification
        lam = lam.squeeze(1)
        y_mixedup = lam * y + (1 - lam) * y_shuffled
    else:
        # Classification task
        one_hot_y = F.one_hot(y, num_classes=num_classes)
        one_hot_y_shuffled = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mixedup = (lam * one_hot_y + (1 - lam) * one_hot_y_shuffled)
    return x_mixedup, y_mixedup


def start_fn(train_df, val_df, test_df):
    return train_df, val_df, test_df


def materialize_fn(train_df, val_df, test_df, dataset_results, config):
    """
    階段1: Materialization - 將已切分的 train/val/test DataFrame 合併並轉換為張量格式
    """
    print("Executing materialize_fn")
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    # 獲取配置參數
    dataset_name = dataset_results['dataset']
    task_type = dataset_results['info']['task_type']

    # 設備設置
    device = resolve_device(config)
    print(f"[MATERIALIZE] Final device: {device}")

    # 數據集包裝（直接合併三份 DataFrame，標記 split_col）
    dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
    dataset.materialize()

    # 根據 split_col 取得三份 tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]



    # 類別特徵轉換為數值特徵
    categorical_transform = CatToNumTransform()
    categorical_transform.fit(train_tensor_frame, dataset.col_stats)
    train_tensor_frame = categorical_transform(train_tensor_frame)
    val_tensor_frame = categorical_transform(val_tensor_frame)
    test_tensor_frame = categorical_transform(test_tensor_frame)
    col_stats = categorical_transform.transformed_stats



    # 基於互信息的特徵排序
    mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
    mutual_info_sort.fit(train_tensor_frame, col_stats)
    train_tensor_frame = mutual_info_sort(train_tensor_frame)
    val_tensor_frame = mutual_info_sort(val_tensor_frame)
    test_tensor_frame = mutual_info_sort(test_tensor_frame)



    # 獲取分類任務信息
    is_classification = dataset.task_type.is_classification
    if is_classification:
        out_channels = dataset.num_classes
    else:
        out_channels = 1
    is_binary_class = is_classification and out_channels == 2

    # 創建數據加載器
    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    # 設置評估指標
    if is_binary_class:
        metric_computer = AUROC(task='binary')
        metric = 'AUC'
    elif is_classification:
        metric_computer = Accuracy(task='multiclass', num_classes=out_channels)
        metric = 'Acc'
    else:
        metric_computer = MeanSquaredError()
        metric = 'RMSE'

    metric_computer = metric_computer.to(device)

    return {
        'dataset': dataset,
        'train_tensor_frame': train_tensor_frame,
        'val_tensor_frame': val_tensor_frame,
        'test_tensor_frame': test_tensor_frame,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'col_stats': col_stats,
        'mutual_info_sort': mutual_info_sort,
        'metric_computer': metric_computer,
        'metric': metric,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'out_channels': out_channels,
        'device': device
    }

def excelformer_core_fn(material_outputs, config, task_type, gnn_stage=None):
    """
    ExcelFormer核心函數：將materialize_fn、encoding_fn、columnwise_fn和decoding_fn整合
    """
    print("Executing excelformer_core_fn")
    train_tensor_frame = material_outputs['train_tensor_frame']
    val_tensor_frame = material_outputs['val_tensor_frame']
    test_tensor_frame = material_outputs['test_tensor_frame']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    mutual_info_sort = material_outputs['mutual_info_sort']
    device = material_outputs['device']
    out_channels = material_outputs['out_channels']
    device = material_outputs['device']
    is_classification = material_outputs['is_classification']
    is_binary_class = material_outputs['is_binary_class']
    metric_computer = material_outputs['metric_computer']
    metric = material_outputs['metric']
    # print(f"Input Train TensorFrame shape: {train_tensor_frame.feat_dict[stype.numerical].shape}")
    # print(f"Input Val TensorFrame shape: {val_tensor_frame.feat_dict[stype.numerical].shape}")
    # print(f"Input Test TensorFrame shape: {test_tensor_frame.feat_dict[stype.numerical].shape}")
    # 獲取模型參數
    channels = config.get('channels', 256)
    mixup = config.get('mixup', None)
    beta = config.get('beta', 0.5)
    num_layers = config.get('num_layers', 5)
    num_heads = config.get('num_heads', 4)
    patience = config.get('patience', 10)

    
    # 【關鍵配置】根據 gnn_stage 決定是否使用 GNN
    use_gnn = gnn_stage in ['encoding', 'columnwise', 'decoding']
    gnn_knn = config.get('gnn_knn', 5)  # kNN 的 k 值
    gnn_hidden = config.get('gnn_hidden', 64)
    
    # 創建 GNN 和相關組件（如果需要）
    gnn = None
    dgm_module = None
    pool_query = None
    mlp_expand = None
    
    if use_gnn:
        if gnn_stage == 'decoding':
            # decoding 階段：Self-Attention + DGM + GNN 作為 decoder
            # 1. Multi-Head Self-Attention (列間交互)
            attn_heads = config.get('gnn_num_heads', 4)
            self_attn = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
            attn_norm = torch.nn.LayerNorm(channels).to(device)
            
            # 2. Column embedding (可學習的列位置編碼)
            column_embed = torch.nn.Parameter(torch.randn(train_tensor_frame.num_cols, channels, device=device))
            
            # 3. Attention pooling query (用於將 self-attention 後的結果聚合為 row-level)
            pool_query = torch.nn.Parameter(torch.randn(channels, device=device))
            
            # 4. DGM_d 動態圖模組
            dgm_k = config.get('dgm_k', config.get('gnn_knn', 5))
            dgm_distance = config.get('dgm_distance', 'euclidean')
            
            class DGMEmbedWrapper(torch.nn.Module):
                def forward(self, x, A=None):
                    return x
            
            dgm_embed_f = DGMEmbedWrapper()
            dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
            
            # 5. Batch GCN (輸入=channels, 輸出=out_channels) - 作為 decoder
            gnn = SimpleGCN(channels, gnn_hidden, out_channels, num_layers=2).to(device)
            
            print(f"✓ Decoding-Self-Attention-DGM-GNN-as-Decoder Pipeline created:")
            print(f"  - Multi-Head Self-Attention (heads={attn_heads}, 列間交互)")
            print(f"  - Column Embedding (learnable positional encoding)")
            print(f"  - Attention Pooling (列→行聚合)")
            print(f"  - DGM_d (k={dgm_k}, distance={dgm_distance})")
            print(f"  - GNN as Decoder (input={channels}, hidden={gnn_hidden}, output={out_channels})")
            print(f"  - PreNorm+FFN for decoding attention")
        
        elif gnn_stage == 'encoding':
            # encoding 階段：使用 Self-Attention + DGM 動態圖
            # 1. Multi-Head Self-Attention (列間交互)
            attn_heads = config.get('gnn_num_heads', 4)
            self_attn = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
            attn_norm = torch.nn.LayerNorm(channels).to(device)
            
            # 2. Column embedding (可學習的列位置編碼)
            column_embed = torch.nn.Parameter(torch.randn(train_tensor_frame.num_cols, channels, device=device))
            
            # 3. Attention pooling query (用於將 self-attention 後的結果聚合為 row-level)
            pool_query = torch.nn.Parameter(torch.randn(channels, device=device))
            
            # 4. DGM_d 動態圖模組
            dgm_k = config.get('dgm_k', config.get('gnn_knn', 5))
            dgm_distance = config.get('dgm_distance', 'euclidean')
            
            class DGMEmbedWrapper(torch.nn.Module):
                def forward(self, x, A=None):
                    return x
            
            dgm_embed_f = DGMEmbedWrapper()
            dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
            
            # 5. Batch GCN (輸入=channels, 輸出=channels)
            gnn = SimpleGCN(channels, gnn_hidden, channels, num_layers=2).to(device)
            
            # 6. Self-Attention 解碼層 (將 GCN 輸出重建回列級表示)
            self_attn_out = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
            gcn_to_attn = torch.nn.Linear(channels, channels).to(device)
            attn_out_norm = torch.nn.LayerNorm(channels).to(device)
            gnn_dropout = config.get('gnn_dropout', 0.1)
            ffn_pre = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(gnn_dropout),
                torch.nn.Linear(channels * 2, channels),
            ).to(device)
            ffn_post = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(gnn_dropout),
                torch.nn.Linear(channels * 2, channels),
            ).to(device)
            
            # 7. 可學習的融合權重
            fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))
            
            print(f"✓ Encoding-Self-Attention-DGM Pipeline created:")
            print(f"  - Multi-Head Self-Attention (heads={attn_heads}, 列間交互)")
            print(f"  - Column Embedding (learnable positional encoding)")
            print(f"  - DGM_d (k={dgm_k}, distance={dgm_distance})")
            print(f"  - Batch GCN (input={channels}, hidden={gnn_hidden}, output={channels})")
            print(f"  - Self-Attention Decoder (重建列級表示)")
            print(f"  - Learnable Fusion Alpha (init=0.3, trainable)")
        
        elif gnn_stage == 'columnwise':
            # columnwise 階段：使用 Self-Attention + DGM 動態圖
            # 1. Multi-Head Self-Attention (列間交互)
            attn_heads = config.get('gnn_num_heads', 4)
            self_attn = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
            attn_norm = torch.nn.LayerNorm(channels).to(device)
            
            # 2. Column embedding (可學習的列位置編碼)
            column_embed = torch.nn.Parameter(torch.randn(train_tensor_frame.num_cols, channels, device=device))
            
            # 3. Attention pooling query (用於將 self-attention 後的結果聚合為 row-level)
            pool_query = torch.nn.Parameter(torch.randn(channels, device=device))
            
            # 4. DGM_d 動態圖模組
            dgm_k = config.get('dgm_k', config.get('gnn_knn', 5))
            dgm_distance = config.get('dgm_distance', 'euclidean')
            
            class DGMEmbedWrapper(torch.nn.Module):
                def forward(self, x, A=None):
                    return x
            
            dgm_embed_f = DGMEmbedWrapper()
            dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
            
            # 5. Batch GCN (輸入=channels, 輸出=channels)
            gnn = SimpleGCN(channels, gnn_hidden, channels, num_layers=2).to(device)
            
            # 6. Self-Attention 解碼層 (將 GCN 輸出重建回列級表示)
            self_attn_out = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
            gcn_to_attn = torch.nn.Linear(channels, channels).to(device)
            attn_out_norm = torch.nn.LayerNorm(channels).to(device)
            gnn_dropout = config.get('gnn_dropout', 0.1)
            ffn_pre = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(gnn_dropout),
                torch.nn.Linear(channels * 2, channels),
            ).to(device)
            ffn_post = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(gnn_dropout),
                torch.nn.Linear(channels * 2, channels),
            ).to(device)
            
            # 7. 可學習的融合權重
            fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))
            
            print(f"✓ Columnwise-Self-Attention-DGM Pipeline created:")
            print(f"  - Multi-Head Self-Attention (heads={attn_heads}, 列間交互)")
            print(f"  - Column Embedding (learnable positional encoding)")
            print(f"  - DGM_d (k={dgm_k}, distance={dgm_distance})")
            print(f"  - Batch GCN (input={channels}, hidden={gnn_hidden}, output={channels})")
            print(f"  - Self-Attention Decoder (重建列級表示)")
            print(f"  - Learnable Fusion Alpha (init=0.3, trainable)")
    
    # 創建ExcelFormer的編碼器部分
    stype_encoder_dict = {
        stype.numerical: ExcelFormerEncoder(channels, na_strategy=NAStrategy.MEAN)
    }
    
    encoder = StypeWiseFeatureEncoder(
        out_channels=channels,
        col_stats=mutual_info_sort.transformed_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
    

    # 創建ExcelFormer的列間交互層
    convs = ModuleList([
        ExcelFormerConv(
            channels, 
            train_tensor_frame.num_cols, 
            num_heads, 
            diam_dropout=0.3,
            aium_dropout=0.0,
            residual_dropout=0.0
        ).to(device)
        for _ in range(num_layers)
    ])
    
    # 創建解碼器（如果 gnn_stage != 'decoding' 才需要）
    decoder = None
    if gnn_stage != 'decoding':
        decoder = ExcelFormerDecoder(
            channels, 
            out_channels, 
            train_tensor_frame.num_cols
        ).to(device)
        print(f"✓ Decoder created (GNN will NOT replace decoder)")
    else:
        print(f"✓ Decoder NOT created (GNN will REPLACE decoder)")
    
    # 【核心前向傳播】
    def model_forward(tf, mixup_encoded=False):
        """
        Args:
            tf: TensorFrame 對象
            mixup_encoded: 是否在編碼後應用 mixup
        Returns:
            out: 預測輸出 [batch, out_channels]
            y_mixedup: 混合後的標籤
        """
        # ======================== 編碼階段 ========================
        x, _ = encoder(tf)  # x: [batch, num_cols, channels]
        batch_size, num_cols, channels_ = x.shape
        
        # 【encoding 階段 Self-Attention + DGM-GNN】
        if gnn_stage == 'encoding' and dgm_module is not None:
            # Step 1: Self-Attention 列間交互 [batch, num_cols, channels] → [batch, num_cols, channels]
            # 添加列位置編碼
            tokens = x + column_embed.unsqueeze(0)  # [batch, num_cols, channels]
            # PreNorm + 殘差
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            # FFN + 殘差
            ffn_out1 = ffn_pre(attn_norm(tokens_attn))
            tokens_attn = tokens_attn + ffn_out1
            
            # Step 2: Attention Pooling [batch, num_cols, channels] → [batch, channels]
            # 在 self-attention 後的列表示上進行池化
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels_)  # [batch, num_cols]
            pool_weights = torch.softmax(pool_logits, dim=1)  # [batch, num_cols]
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)  # [batch, channels]
            
            # Step 3: Mini-batch DGM 動態建圖（標準化 + 對稱化 + 自迴路）
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)  # [1, batch, channels]
            # 依 batch 大小動態收斂 k，避免小批次下圖結構退化
            if hasattr(dgm_module, 'k'):
                Ns_enc = x_pooled_batched.shape[1]
                dgm_module.k = int(min(int(dgm_module.k), max(1, Ns_enc - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)  # [batch, channels]
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])
            
            # Step 4: Batch GCN 處理
            x_gnn_out = gnn(x_dgm, edge_index_dgm)  # [batch, channels]
            
            # Step 5: Self-Attention 解碼 [batch, channels] → [batch, num_cols, channels]
            # 將 GCN 輸出注入 self-attention 並重建列級表示
            gcn_ctx = gcn_to_attn(x_gnn_out).unsqueeze(1)  # [batch, 1, channels]
            tokens_with_ctx = tokens_attn + gcn_ctx  # 廣播到所有列
            # PreNorm + 殘差
            tokens_ctx_norm = attn_out_norm(tokens_with_ctx)
            attn_out2, _ = self_attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
            tokens_mid = tokens_with_ctx + attn_out2
            # FFN + 殘差
            ffn_out2 = ffn_post(attn_out_norm(tokens_mid))
            tokens_out = tokens_mid + ffn_out2
            
            # Step 6: 殘差融合
            fusion_alpha = torch.sigmoid(fusion_alpha_param)
            x = x + fusion_alpha * tokens_out  # [batch, num_cols, channels]
        
        # 如需 mixup，則在 GNN 後應用
        if mixup_encoded and mixup is not None:
            assert tf.y is not None
            x, y_mixedup = feature_mixup(
                x,
                tf.y,
                num_classes=out_channels,
                beta=beta,
                mixup_type=mixup,
                mi_scores=getattr(tf, 'mi_scores', None),
            )
        else:
            y_mixedup = tf.y
        
        # ======================== 列間交互階段 ========================
        for conv in convs:
            x = conv(x)
        
        # 【columnwise 階段 Self-Attention + DGM-GNN】
        if gnn_stage == 'columnwise' and dgm_module is not None:
            # Step 1: Self-Attention 列間交互 [batch, num_cols, channels] → [batch, num_cols, channels]
            # 添加列位置編碼
            tokens = x + column_embed.unsqueeze(0)  # [batch, num_cols, channels]
            # PreNorm + 殘差
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            # FFN + 殘差
            ffn_out1 = ffn_pre(attn_norm(tokens_attn))
            tokens_attn = tokens_attn + ffn_out1
            
            # Step 2: Attention Pooling [batch, num_cols, channels] → [batch, channels]
            # 在 self-attention 後的列表示上進行池化
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels_)  # [batch, num_cols]
            pool_weights = torch.softmax(pool_logits, dim=1)  # [batch, num_cols]
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)  # [batch, channels]
            
            # Step 3: Mini-batch DGM 動態建圖（標準化 + 對稱化 + 自迴路）
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)  # [1, batch, channels]
            # 依 batch 大小動態收斂 k，避免小批次下圖結構退化
            if hasattr(dgm_module, 'k'):
                Ns_col = x_pooled_batched.shape[1]
                dgm_module.k = int(min(int(dgm_module.k), max(1, Ns_col - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)  # [batch, channels]
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])
            
            # Step 4: Batch GCN 處理
            x_gnn_out = gnn(x_dgm, edge_index_dgm)  # [batch, channels]
            
            # Step 5: Self-Attention 解碼 [batch, channels] → [batch, num_cols, channels]
            # 將 GCN 輸出注入 self-attention 並重建列級表示
            gcn_ctx = gcn_to_attn(x_gnn_out).unsqueeze(1)  # [batch, 1, channels]
            tokens_with_ctx = tokens_attn + gcn_ctx  # 廣播到所有列
            # PreNorm + 殘差
            tokens_ctx_norm = attn_out_norm(tokens_with_ctx)
            attn_out2, _ = self_attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
            tokens_mid = tokens_with_ctx + attn_out2
            # FFN + 殘差
            ffn_out2 = ffn_post(attn_out_norm(tokens_mid))
            tokens_out = tokens_mid + ffn_out2
            
            # Step 6: 殘差融合
            fusion_alpha = torch.sigmoid(fusion_alpha_param)
            x = x + fusion_alpha * tokens_out  # [batch, num_cols, channels]
        
        # ======================== 解碼階段 ========================
        if gnn_stage == 'decoding' and dgm_module is not None:
            # Step 1: Self-Attention 列間交互 [batch, num_cols, channels] → [batch, num_cols, channels]
            # 添加列位置編碼
            tokens = x + column_embed.unsqueeze(0)  # [batch, num_cols, channels]
            # PreNorm + 殘差
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            
            # Step 2: Attention Pooling [batch, num_cols, channels] → [batch, channels]
            # 將列級表示聚合為行級表示
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels_)  # [batch, num_cols]
            pool_weights = torch.softmax(pool_logits, dim=1)  # [batch, num_cols]
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)  # [batch, channels]
            
            # Step 3: Mini-batch DGM 動態建圖（標準化 + 對稱化 + 自迴路）
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)  # [1, batch, channels]
            # 依 batch 大小動態收斂 k，避免小批次下圖結構退化
            if hasattr(dgm_module, 'k'):
                Ns_dec = x_pooled_batched.shape[1]
                dgm_module.k = int(min(int(dgm_module.k), max(1, Ns_dec - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)  # [batch, channels]
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])
            
            # Step 4: Batch GCN 作為 Decoder 直接輸出預測
            out = gnn(x_dgm, edge_index_dgm)  # [batch, out_channels]
        else:
            # 使用原有 decoder
            out = decoder(x)  # [batch, out_channels]
        
        return out, y_mixedup
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.001)
    gamma = config.get('gamma', 0.95)
    
    # 【聯合訓練】收集所有需要訓練的參數（包括 DGM、attention pooling、expand、fusion_alpha）
    all_params = list(encoder.parameters()) + [p for conv in convs for p in conv.parameters()]
    if gnn is not None:
        all_params += list(gnn.parameters())
    if decoder is not None:
        all_params += list(decoder.parameters())
    # encoding 階段的額外組件 (Self-Attention + DGM)
    if gnn_stage == 'encoding':
        if 'self_attn' in locals():
            all_params += list(self_attn.parameters())
        if 'attn_norm' in locals():
            all_params += list(attn_norm.parameters())
        if 'self_attn_out' in locals():
            all_params += list(self_attn_out.parameters())
        if 'attn_out_norm' in locals():
            all_params += list(attn_out_norm.parameters())
        if 'column_embed' in locals():
            all_params += [column_embed]
        if 'gcn_to_attn' in locals():
            all_params += list(gcn_to_attn.parameters())
        if 'ffn_pre' in locals():
            all_params += list(ffn_pre.parameters())
        if 'ffn_post' in locals():
            all_params += list(ffn_post.parameters())
        if dgm_module is not None:
            all_params += list(dgm_module.parameters())
        if pool_query is not None:
            all_params += [pool_query]
        if 'fusion_alpha_param' in locals():
            all_params += [fusion_alpha_param]
    # decoding 階段的額外組件 (Self-Attention + DGM, GNN 作為 decoder)
    elif gnn_stage == 'decoding':
        if 'self_attn' in locals():
            all_params += list(self_attn.parameters())
        if 'attn_norm' in locals():
            all_params += list(attn_norm.parameters())
        if 'column_embed' in locals():
            all_params += [column_embed]
        if dgm_module is not None:
            all_params += list(dgm_module.parameters())
        if pool_query is not None:
            all_params += [pool_query]
    # columnwise 階段的額外組件 (Self-Attention + DGM)
    elif gnn_stage == 'columnwise':
        if 'self_attn' in locals():
            all_params += list(self_attn.parameters())
        if 'attn_norm' in locals():
            all_params += list(attn_norm.parameters())
        if 'self_attn_out' in locals():
            all_params += list(self_attn_out.parameters())
        if 'attn_out_norm' in locals():
            all_params += list(attn_out_norm.parameters())
        if 'column_embed' in locals():
            all_params += [column_embed]
        if 'gcn_to_attn' in locals():
            all_params += list(gcn_to_attn.parameters())
        if 'ffn_pre' in locals():
            all_params += list(ffn_pre.parameters())
        if 'ffn_post' in locals():
            all_params += list(ffn_post.parameters())
        if dgm_module is not None:
            all_params += list(dgm_module.parameters())
        if pool_query is not None:
            all_params += [pool_query]
        if 'fusion_alpha_param' in locals():
            all_params += [fusion_alpha_param]
    
    optimizer = torch.optim.Adam(all_params, lr=lr)
    print(f"✓ Optimizer created with {len(all_params)} parameter groups")
    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        encoder.train()
        for conv in convs:
            conv.train()
        if decoder is not None:
            decoder.train()
        if gnn is not None:
            gnn.train()
        # encoding 階段組件 (Self-Attention + DGM)
        if gnn_stage == 'encoding':
            if 'self_attn' in locals() and self_attn is not None:
                self_attn.train()
            if 'attn_norm' in locals() and attn_norm is not None:
                attn_norm.train()
            if 'self_attn_out' in locals() and self_attn_out is not None:
                self_attn_out.train()
            if 'attn_out_norm' in locals() and attn_out_norm is not None:
                attn_out_norm.train()
            if 'gcn_to_attn' in locals() and gcn_to_attn is not None:
                gcn_to_attn.train()
            if 'ffn_pre' in locals() and ffn_pre is not None:
                ffn_pre.train()
            if 'ffn_post' in locals() and ffn_post is not None:
                ffn_post.train()
            if dgm_module is not None:
                dgm_module.train()
        # decoding 階段組件 (Self-Attention + DGM, GNN 作為 decoder)
        elif gnn_stage == 'decoding':
            if 'self_attn' in locals() and self_attn is not None:
                self_attn.train()
            if dgm_module is not None:
                dgm_module.train()
            if 'attn_norm' in locals() and attn_norm is not None:
                attn_norm.train()
        # columnwise 階段組件 (Self-Attention + DGM)
        elif gnn_stage == 'columnwise':
            if 'self_attn' in locals() and self_attn is not None:
                self_attn.train()
            if 'attn_norm' in locals() and attn_norm is not None:
                attn_norm.train()
            if 'self_attn_out' in locals() and self_attn_out is not None:
                self_attn_out.train()
            if 'attn_out_norm' in locals() and attn_out_norm is not None:
                attn_out_norm.train()
            if 'gcn_to_attn' in locals() and gcn_to_attn is not None:
                gcn_to_attn.train()
            if 'ffn_pre' in locals() and ffn_pre is not None:
                ffn_pre.train()
            if 'ffn_post' in locals() and ffn_post is not None:
                ffn_post.train()
            if dgm_module is not None:
                dgm_module.train()
        
        loss_accum = total_count = 0
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            # 使用FEAT-MIX或HIDDEN-MIX進行訓練
            pred_mixedup, y_mixedup = model_forward(tf, mixup_encoded=True)
            if is_classification:
                # 軟混合的one-hot標籤
                loss = F.cross_entropy(pred_mixedup, y_mixedup)
            else:
                loss = F.mse_loss(pred_mixedup.view(-1), y_mixedup.view(-1))
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(y_mixedup)
            total_count += len(y_mixedup)
            optimizer.step()
        return loss_accum / total_count
    
    # 定義測試函數
    @torch.no_grad()
    def test(loader):
        # 設置為評估模式
        encoder.eval()
        for conv in convs:
            conv.eval()
        if decoder is not None:
            decoder.eval()
        if gnn is not None:
            gnn.eval()
        # encoding 階段組件 (Self-Attention + DGM)
        if gnn_stage == 'encoding':
            if 'self_attn' in locals() and self_attn is not None:
                self_attn.eval()
            if 'attn_norm' in locals() and attn_norm is not None:
                attn_norm.eval()
            if 'self_attn_out' in locals() and self_attn_out is not None:
                self_attn_out.eval()
            if 'attn_out_norm' in locals() and attn_out_norm is not None:
                attn_out_norm.eval()
            if 'gcn_to_attn' in locals() and gcn_to_attn is not None:
                gcn_to_attn.eval()
            if 'ffn_pre' in locals() and ffn_pre is not None:
                ffn_pre.eval()
            if 'ffn_post' in locals() and ffn_post is not None:
                ffn_post.eval()
            if dgm_module is not None:
                dgm_module.eval()
        # decoding 階段組件 (Self-Attention + DGM, GNN 作為 decoder)
        elif gnn_stage == 'decoding':
            if 'self_attn' in locals() and self_attn is not None:
                self_attn.eval()
            if dgm_module is not None:
                dgm_module.eval()
            if 'attn_norm' in locals() and attn_norm is not None:
                attn_norm.eval()
        # columnwise 階段組件 (Self-Attention + DGM)
        elif gnn_stage == 'columnwise':
            if 'self_attn' in locals() and self_attn is not None:
                self_attn.eval()
            if 'attn_norm' in locals() and attn_norm is not None:
                attn_norm.eval()
            if 'self_attn_out' in locals() and self_attn_out is not None:
                self_attn_out.eval()
            if 'attn_out_norm' in locals() and attn_out_norm is not None:
                attn_out_norm.eval()
            if 'gcn_to_attn' in locals() and gcn_to_attn is not None:
                gcn_to_attn.eval()
            if 'ffn_pre' in locals() and ffn_pre is not None:
                ffn_pre.eval()
            if 'ffn_post' in locals() and ffn_post is not None:
                ffn_post.eval()
            if dgm_module is not None:
                dgm_module.eval()
        
        metric_computer.reset()
        loss_accum = total_count = 0
        
        for tf in loader:
            tf = tf.to(device)
            pred, _ = model_forward(tf)
            
            # 计算loss
            if is_classification:
                loss = F.cross_entropy(pred, tf.y)
            else:
                loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            
            if is_binary_class:
                metric_computer.update(pred[:, 1], tf.y)
            elif is_classification:
                pred_class = pred.argmax(dim=-1)
                metric_computer.update(pred_class, tf.y)
            else:
                metric_computer.update(pred.view(-1), tf.y.view(-1))
        avg_loss = loss_accum / total_count
        if is_classification:
            return metric_computer.compute().item(), avg_loss
        else:
            return metric_computer.compute().item()**0.5, avg_loss
    
    # 初始化最佳指標（基于val_loss）
    best_val_loss = float('inf')
    best_val_metric = 0 if is_classification else float('inf')
    best_epoch = 0
    early_stop_counter = 0
    train_losses = []
    train_metrics = []
    val_metrics = []
    val_losses = []
    best_states = None
    loss_threshold = config.get('loss_threshold', 1e-4)

    epochs = config.get('epochs', 200)
    early_stop_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        train_metric, _ = test(train_loader)
        val_metric, val_loss = test(val_loader)

        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        val_losses.append(val_loss)
        
        # 基于val_loss判定是否改进
        improved = val_loss < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss
            best_val_metric = val_metric
            best_epoch = epoch
            early_stop_counter = 0
            # 保存最佳权重
            best_states = {
                'encoder': encoder.state_dict(),
                'convs': [conv.state_dict() for conv in convs],
                'decoder': decoder.state_dict() if decoder is not None else None,
                'gnn': gnn.state_dict() if gnn is not None else None,
            }
            # 保存encoding阶段的额外组件 (Self-Attention + DGM)
            if gnn_stage == 'encoding':
                if 'self_attn' in locals() and self_attn is not None:
                    best_states['self_attn'] = self_attn.state_dict()
                if 'attn_norm' in locals() and attn_norm is not None:
                    best_states['attn_norm'] = attn_norm.state_dict()
                if 'self_attn_out' in locals() and self_attn_out is not None:
                    best_states['self_attn_out'] = self_attn_out.state_dict()
                if 'attn_out_norm' in locals() and attn_out_norm is not None:
                    best_states['attn_out_norm'] = attn_out_norm.state_dict()
                if 'column_embed' in locals():
                    best_states['column_embed'] = column_embed.detach().clone()
                if 'gcn_to_attn' in locals() and gcn_to_attn is not None:
                    best_states['gcn_to_attn'] = gcn_to_attn.state_dict()
                if 'ffn_pre' in locals() and ffn_pre is not None:
                    best_states['ffn_pre'] = ffn_pre.state_dict()
                if 'ffn_post' in locals() and ffn_post is not None:
                    best_states['ffn_post'] = ffn_post.state_dict()
                if dgm_module is not None:
                    best_states['dgm_module'] = dgm_module.state_dict()
                if pool_query is not None:
                    best_states['pool_query'] = pool_query.detach().clone()
                if 'fusion_alpha_param' in locals():
                    best_states['fusion_alpha_param'] = fusion_alpha_param.detach().clone()
            # 保存decoding阶段的额外组件 (Self-Attention + DGM, GNN 作為 decoder)
            elif gnn_stage == 'decoding':
                if 'self_attn' in locals() and self_attn is not None:
                    best_states['self_attn'] = self_attn.state_dict()
                if 'column_embed' in locals():
                    best_states['column_embed'] = column_embed.detach().clone()
                if dgm_module is not None:
                    best_states['dgm_module'] = dgm_module.state_dict()
                if pool_query is not None:
                    best_states['pool_query'] = pool_query.detach().clone()
            # 保存columnwise阶段的额外组件 (Self-Attention + DGM)
            elif gnn_stage == 'columnwise':
                if 'self_attn' in locals() and self_attn is not None:
                    best_states['self_attn'] = self_attn.state_dict()
                if 'attn_norm' in locals() and attn_norm is not None:
                    best_states['attn_norm'] = attn_norm.state_dict()
                if 'self_attn_out' in locals() and self_attn_out is not None:
                    best_states['self_attn_out'] = self_attn_out.state_dict()
                if 'attn_out_norm' in locals() and attn_out_norm is not None:
                    best_states['attn_out_norm'] = attn_out_norm.state_dict()
                if 'column_embed' in locals():
                    best_states['column_embed'] = column_embed.detach().clone()
                if 'gcn_to_attn' in locals() and gcn_to_attn is not None:
                    best_states['gcn_to_attn'] = gcn_to_attn.state_dict()
                if 'ffn_pre' in locals() and ffn_pre is not None:
                    best_states['ffn_pre'] = ffn_pre.state_dict()
                if 'ffn_post' in locals() and ffn_post is not None:
                    best_states['ffn_post'] = ffn_post.state_dict()
                if dgm_module is not None:
                    best_states['dgm_module'] = dgm_module.state_dict()
                if pool_query is not None:
                    best_states['pool_query'] = pool_query.detach().clone()
                if 'fusion_alpha_param' in locals():
                    best_states['fusion_alpha_param'] = fusion_alpha_param.detach().clone()
        else:
            early_stop_counter += 1

        print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val {metric}: {val_metric:.4f}'
              f'{" ↓ (improved)" if improved else ""}')

        lr_scheduler.step()
        if early_stop_counter >= patience:
            early_stop_epochs = epoch
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})")
            break

    # 恢复最佳权重
    if best_states is not None:
        encoder.load_state_dict(best_states['encoder'])
        for i, conv in enumerate(convs):
            conv.load_state_dict(best_states['convs'][i])
        if decoder is not None and best_states['decoder'] is not None:
            decoder.load_state_dict(best_states['decoder'])
        if gnn is not None and best_states['gnn'] is not None:
            gnn.load_state_dict(best_states['gnn'])
        if gnn_stage == 'encoding':
            if 'self_attn' in locals() and self_attn is not None and 'self_attn' in best_states:
                self_attn.load_state_dict(best_states['self_attn'])
            if 'self_attn_out' in locals() and self_attn_out is not None and 'self_attn_out' in best_states:
                self_attn_out.load_state_dict(best_states['self_attn_out'])
            if 'column_embed' in locals() and 'column_embed' in best_states:
                with torch.no_grad():
                    column_embed.copy_(best_states['column_embed'])
            if 'gcn_to_attn' in locals() and gcn_to_attn is not None and 'gcn_to_attn' in best_states:
                gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
            if dgm_module is not None and 'dgm_module' in best_states:
                dgm_module.load_state_dict(best_states['dgm_module'])
            if pool_query is not None and 'pool_query' in best_states:
                with torch.no_grad():
                    pool_query.copy_(best_states['pool_query'])
            if 'fusion_alpha_param' in locals() and 'fusion_alpha_param' in best_states:
                with torch.no_grad():
                    fusion_alpha_param.copy_(best_states['fusion_alpha_param'])
        elif gnn_stage == 'decoding':
            if 'self_attn' in locals() and self_attn is not None and 'self_attn' in best_states:
                self_attn.load_state_dict(best_states['self_attn'])
            if 'column_embed' in locals() and 'column_embed' in best_states:
                with torch.no_grad():
                    column_embed.copy_(best_states['column_embed'])
            if dgm_module is not None and 'dgm_module' in best_states:
                dgm_module.load_state_dict(best_states['dgm_module'])
            if pool_query is not None and 'pool_query' in best_states:
                with torch.no_grad():
                    pool_query.copy_(best_states['pool_query'])
        elif gnn_stage == 'columnwise':
            if 'self_attn' in locals() and self_attn is not None and 'self_attn' in best_states:
                self_attn.load_state_dict(best_states['self_attn'])
            if 'self_attn_out' in locals() and self_attn_out is not None and 'self_attn_out' in best_states:
                self_attn_out.load_state_dict(best_states['self_attn_out'])
            if 'column_embed' in locals() and 'column_embed' in best_states:
                with torch.no_grad():
                    column_embed.copy_(best_states['column_embed'])
            if 'gcn_to_attn' in locals() and gcn_to_attn is not None and 'gcn_to_attn' in best_states:
                gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
            if dgm_module is not None and 'dgm_module' in best_states:
                dgm_module.load_state_dict(best_states['dgm_module'])
            if pool_query is not None and 'pool_query' in best_states:
                with torch.no_grad():
                    pool_query.copy_(best_states['pool_query'])
            if 'fusion_alpha_param' in locals() and 'fusion_alpha_param' in best_states:
                with torch.no_grad():
                    fusion_alpha_param.copy_(best_states['fusion_alpha_param'])
        print(f"✓ Restored best weights from epoch {best_epoch}")

    # 最終測試
    print(f'✓ Training complete. Best Val Loss: {best_val_loss:.4f}, Best Val {metric}: {best_val_metric:.4f}')
    test_metric, test_loss = test(test_loader)
    print(f'✓ Final Test {metric}: {test_metric:.4f}, Test Loss: {test_loss:.4f}')

    return {
        'gnn_early_stop_epochs': material_outputs.get('gnn_early_stop_epochs', 0),
        'train_losses': train_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_val_metric': best_val_metric,
        'final_metric': best_val_metric,
        'best_test_metric': test_metric,
        'encoder': encoder,
        'convs': convs,
        'decoder': decoder,
        'gnn': gnn,
        'model_forward': model_forward,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'metric_computer': metric_computer,
        'metric': metric,
        'early_stop_epochs': early_stop_epochs,
        'gnn_stage': gnn_stage,
    }





def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    print("ExcelFormer - 五階段執行")
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    print(f"device used: {config.get('device', 'cpu')}")
    # df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    # class_count_dict = df['target'].value_counts().to_dict()
    # # 按 key 排序
    # class_count_dict_sorted = dict(sorted(class_count_dict.items()))
    # print("各類別數量統計（dict，已排序）：", class_count_dict_sorted)

    # print("總共有幾個類別：", df['target'].nunique())
    # 獲取配置參數
    try:
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        gnn_early_stop_epochs=0
        if gnn_stage=='start':
            # 在 start_fn 和 materialize_fn 之間插入 GNN
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        if gnn_stage == 'materialize':
            # 在 materialize_fn 和 encoding_fn 之間插入 GNN
            train_tensor_frame = material_outputs['train_tensor_frame']
            val_tensor_frame = material_outputs['val_tensor_frame']
            test_tensor_frame = material_outputs['test_tensor_frame']
            (train_loader, val_loader, test_loader,col_stats, mutual_info_sort, dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs) = gnn_after_materialize_fn(
                train_tensor_frame, val_tensor_frame, test_tensor_frame, config, dataset_results['dataset'], task_type)
            material_outputs.update({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'col_stats': col_stats,
                'mutual_info_sort': mutual_info_sort,
                'dataset': dataset,
                'train_tensor_frame': train_tensor_frame,
                'val_tensor_frame': val_tensor_frame,
                'test_tensor_frame': test_tensor_frame,
            })
            material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs

        results=excelformer_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)
    

    except Exception as e:
        is_classification = dataset_results['info']['task_type'] == 'classification'
        results = {
            'train_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': [],
            'best_val_metric': float('-inf') if is_classification else float('inf'),
            'best_test_metric': float('-inf') if is_classification else float('inf'),
            'error': str(e),
        }
    return results


#  python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages all --epochs 2
#  python main.py --dataset eye --models excelformer --gnn_stages all --epochs 2
#  python main.py --dataset house --models excelformer --gnn_stages all --epochs 2
#  python main.py --dataset credit --models excelformer --gnn_stages all --epochs 2
#  python main.py --dataset openml_The_Office_Dataset --models excelformer --gnn_stages all --epochs 2