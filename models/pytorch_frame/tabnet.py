from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GLU, BatchNorm1d, Identity, Linear, Module, ModuleList

import torch_frame
from torch_frame import stype
from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    StackEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame.typing import NAStrategy
from torch_frame.datasets.yandex import Yandex

"""Reported (reproduced) results of of TabNet model in the original paper
https://arxiv.org/abs/1908.07442.

Forest Cover Type: 96.99 (96.53)
KDD Census Income: 95.5 (95.41)
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import ForestCoverType, KDDCensusIncome
from torch_frame.nn import TabNet
import torch
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import sys
import numpy as np

# 引入 DGM 模組
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
    """
    # 基礎公式：k ≈ sqrt(N) 作為起點
    base_k = int(np.sqrt(num_samples))
    
    # 特徵維度修正：高維時需更多候選來探索不同相似性角度
    feature_factor = 1.0 + np.log1p(num_features) / 10  # log 增長，避免過度增加
    adjusted_k = int(base_k * feature_factor)
    
    # 樣本密度修正：樣本少時 relative connectivity 要高
    density_factor = 1.0
    if num_samples < 500:
        density_factor = 1.3  # +30%
    elif num_samples > 5000:
        density_factor = 0.9  # -10%
    
    adaptive_k = int(adjusted_k * density_factor)
    
    # 硬性邊界：防止過度稀疏或過度密集
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
    在 start_fn 和 materialize_fn 之間插入「自注意力 → DGM動態圖 → GCN → 自注意力」的管線。

    流程：
    1) 針對每筆樣本（row），先以 Multi-Head Self-Attention 在欄位維度做上下文建模，
       得到每列的 contextualized column tokens，並透過注意力池化得到 row-level 向量。
    2) 以 row-level 向量作為節點特徵，用DGM動態圖模組（DGM_d）學習最優的圖結構，
       並用GCN訓練（監督式 loss 與任務一致）。DGM的temperature參數和GCN權重一同反向傳播。
    3) 將 GCN 輸出的 row-level embedding 再注入第二個 self-attention，重建回
       [num_rows, num_cols] 形狀的特徵矩陣，回傳與原欄位數一致的 DataFrame。
    """

    # 根據 config 選擇 GPU 編號（如 gpu=0 或 gpu=1）；若不可用則回退到 CPU
    device = resolve_device(config)
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

        # DGM_d 動態圖
        row_emb_batched = row_emb_std.unsqueeze(0)  # [1, Ns, attn_dim]
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
        
        E = edge_index_dgm.shape[1]
        avg_deg = E / max(1, Ns)
        print(f"[START-GNN-DGM] split Ns={Ns}, edges={E}, avg_deg={avg_deg:.2f}")
        
        return logits, recon, logprobs_dgm

    # 訓練循環
    import math
    for epoch in range(gnn_epochs):
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

        dgm_reg = -logprobs_dgm.mean() * 0.01
        train_loss = train_loss + dgm_reg

        train_loss.backward()
        optimizer.step()

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
        
        improved = val_loss_val < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss_val
            early_stop_counter = 0
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
    import math
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
    
    # 創建數據加載器
    batch_size = config.get('batch_size', 4096)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)
    
    return (train_loader, val_loader, test_loader,
            dataset.col_stats, dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs)


    # 取得所有 row 的 embedding


class TabNet(Module):
    r"""The TabNet model introduced in the
    `"TabNet: Attentive Interpretable Tabular Learning"
    <https://arxiv.org/abs/1908.07442>`_ paper.

    .. note::

        For an example of using TabNet, see `examples/tabnet.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        tabnet.py>`_.

    Args:
        out_channels (int): Output dimensionality
        num_layers (int): Number of TabNet layers.
        split_feat_channels (int): Dimensionality of feature channels.
        split_attn_channels (int): Dimensionality of attention channels.
        gamma (float): The gamma value for updating the prior for the attention
            mask.
        col_stats (Dict[str,Dict[torch_frame.data.stats.StatType,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[torch_frame.stype, List[str]]): A
            dictionary that maps :class:`~torch_frame.stype` to a list of
            column names. The column names are sorted based on the ordering
            that appear in :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`StackEncoder()` for
            numerical feature)
        num_shared_glu_layers (int): Number of GLU layers shared across the
            :obj:`num_layers` :class:`FeatureTransformer`s. (default: :obj:`2`)
        num_dependent_glu_layers (int, optional): Number of GLU layers to use
            in each of :obj:`num_layers` :class:`FeatureTransformer`s.
            (default: :obj:`2`)
        cat_emb_channels (int, optional): The categorical embedding
            dimensionality.
    """
    def __init__(
        self,
        out_channels: int,
        num_layers: int,
        split_feat_channels: int,
        split_attn_channels: int,
        gamma: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        num_shared_glu_layers: int = 2,
        num_dependent_glu_layers: int = 2,
        cat_emb_channels: int = 2,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.split_feat_channels = split_feat_channels
        self.split_attn_channels = split_attn_channels
        self.num_layers = num_layers
        self.gamma = gamma

        num_cols = sum([len(v) for v in col_names_dict.values()])
        # if there is no categorical feature, we just set cat_emb_channels to 1
        cat_emb_channels = (cat_emb_channels if torch_frame.categorical
                            in col_names_dict else 1)
        in_channels = cat_emb_channels * num_cols

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical:
                EmbeddingEncoder(na_strategy=NAStrategy.MOST_FREQUENT),
                stype.numerical:
                StackEncoder(na_strategy=NAStrategy.MEAN),
            }

        # Map input tensor frame into (batch_size, num_cols, cat_emb_channels),
        # which is flattened into (batch_size, in_channels)
        self.feature_encoder = StypeWiseFeatureEncoder(
            out_channels=cat_emb_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        # Batch norm applied to input feature.
        self.bn = BatchNorm1d(in_channels)

        shared_glu_block: Module
        if num_shared_glu_layers > 0:
            shared_glu_block = GLUBlock(
                in_channels=in_channels,
                out_channels=split_feat_channels + split_attn_channels,
                no_first_residual=True,
            )
        else:
            shared_glu_block = Identity()

        self.feat_transformers = ModuleList()
        for _ in range(self.num_layers + 1):
            self.feat_transformers.append(
                FeatureTransformer(
                    in_channels,
                    split_feat_channels + split_attn_channels,
                    num_dependent_glu_layers=num_dependent_glu_layers,
                    shared_glu_block=shared_glu_block,
                ))

        self.attn_transformers = ModuleList()
        for _ in range(self.num_layers):
            self.attn_transformers.append(
                AttentiveTransformer(
                    in_channels=split_attn_channels,
                    out_channels=in_channels,
                ))

        self.lin = Linear(self.split_feat_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.feature_encoder.reset_parameters()
        self.bn.reset_parameters()
        for feat_transformer in self.feat_transformers:
            feat_transformer.reset_parameters()
        for attn_transformer in self.attn_transformers:
            attn_transformer.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        tf: TensorFrame,
        return_reg: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Transform :class:`TensorFrame` object into output embeddings.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.
            return_reg (bool): Whether to return the entropy regularization.

        Returns:
            Union[torch.Tensor, (torch.Tensor, torch.Tensor)]: The output
                embeddings of size :obj:`[batch_size, out_channels]`.
                If :obj:`return_reg` is :obj:`True`, return the entropy
                regularization as well.
        """
        # [batch_size, num_cols, cat_emb_channels]
        x, _ = self.feature_encoder(tf)
        batch_size = x.shape[0]
        # [batch_size, num_cols * cat_emb_channels]
        x = x.view(batch_size, math.prod(x.shape[1:]))
        x = self.bn(x)

        # [batch_size, num_cols * cat_emb_channels]
        prior = torch.ones_like(x)
        reg = torch.tensor(0., device=x.device)

        # [batch_size, split_attn_channels]
        attention_x = self.feat_transformers[0](x)
        attention_x = attention_x[:, self.split_feat_channels:]

        outs = []
        for i in range(self.num_layers):
            # [batch_size, num_cols * cat_emb_channels]
            attention_mask = self.attn_transformers[i](attention_x, prior)

            # [batch_size, num_cols * cat_emb_channels]
            masked_x = attention_mask * x
            # [batch_size, split_feat_channels + split_attn_channel]
            out = self.feat_transformers[i + 1](masked_x)

            # Get the split feature
            # [batch_size, split_feat_channels]
            feature_x = F.relu(out[:, :self.split_feat_channels])
            outs.append(feature_x)
            # Get the split attention
            # [batch_size, split_attn_channels]
            attention_x = out[:, self.split_feat_channels:]

            # Update prior
            prior = (self.gamma - attention_mask) * prior

            # Compute entropy regularization
            if return_reg and batch_size > 0:
                entropy = -torch.sum(
                    attention_mask * torch.log(attention_mask + 1e-15),
                    dim=1).mean()
                reg += entropy

        out = sum(outs)
        out = self.lin(out)

        if return_reg:
            return out, reg / self.num_layers
        else:
            return out


class FeatureTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dependent_glu_layers: int,
        shared_glu_block: Module,
    ) -> None:
        super().__init__()

        self.shared_glu_block = shared_glu_block

        self.dependent: Module
        if num_dependent_glu_layers == 0:
            self.dependent = Identity()
        else:
            if not isinstance(self.shared_glu_block, Identity):
                in_channels = out_channels
                no_first_residual = False
            else:
                no_first_residual = True
            self.dependent = GLUBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                no_first_residual=no_first_residual,
                num_glu_layers=num_dependent_glu_layers,
            )
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.shared_glu_block(x)
        x = self.dependent(x)
        return x

    def reset_parameters(self) -> None:
        if not isinstance(self.shared_glu_block, Identity):
            self.shared_glu_block.reset_parameters()
        if not isinstance(self.dependent, Identity):
            self.dependent.reset_parameters()


class GLUBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_glu_layers: int = 2,
        no_first_residual: bool = False,
    ) -> None:
        super().__init__()
        self.no_first_residual = no_first_residual
        self.glu_layers = ModuleList()

        for i in range(num_glu_layers):
            if i == 0:
                glu_layer = GLULayer(in_channels, out_channels)
            else:
                glu_layer = GLULayer(out_channels, out_channels)
            self.glu_layers.append(glu_layer)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for i, glu_layer in enumerate(self.glu_layers):
            if self.no_first_residual and i == 0:
                x = glu_layer(x)
            else:
                x = x * math.sqrt(0.5) + glu_layer(x)
        return x

    def reset_parameters(self) -> None:
        for glu_layer in self.glu_layers:
            glu_layer.reset_parameters()


class GLULayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lin = Linear(in_channels, out_channels * 2, bias=False)
        self.glu = GLU()
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        return self.glu(x)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()


class AttentiveTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bn = GhostBatchNorm1d(out_channels)
        self.reset_parameters()

    def forward(self, x: Tensor, prior: Tensor) -> Tensor:
        x = self.lin(x)
        x = self.bn(x)
        x = prior * x
        # Using softmax instead of sparsemax since softmax performs better.
        x = F.softmax(x, dim=-1)
        return x

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.bn.reset_parameters()


class GhostBatchNorm1d(torch.nn.Module):
    r"""Ghost Batch Normalization https://arxiv.org/abs/1705.08741."""
    def __init__(
        self,
        input_dim: int,
        virtual_batch_size: int = 512,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        if len(x) > 0:
            num_chunks = math.ceil(len(x) / self.virtual_batch_size)
            chunks = torch.chunk(x, num_chunks, dim=0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

    def reset_parameters(self) -> None:
        self.bn.reset_parameters()



def start_fn(train_df, val_df, test_df):
    return train_df, val_df, test_df



def materialize_fn(train_df, val_df, test_df, dataset_results, config):
    """
    階段1: Materialization - 將已切分的 train/val/test DataFrame 合併並轉換為張量格式
    """
    print("Executing materialize_fn")
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    dataset_name = dataset_results['dataset']
    task_type = dataset_results['info']['task_type']

    # 使用統一的設備管理
    device = resolve_device(config)
    print(f"[MATERIALIZE] Final device: {device}")

    # 數據集包裝（直接合併三份 DataFrame，標記 split_col）
    dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    # 根據 split_col 取得三份 tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 4096)  # TabNet通常使用較大的批次
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    if is_classification:
        out_channels = dataset.num_classes
    else:
        out_channels = 1

    is_binary_class = is_classification and out_channels == 2

    # 設置評估指標
    if is_binary_class:
        from torchmetrics import AUROC
        metric_computer = AUROC(task='binary')
        metric = 'AUC'
    elif is_classification:
        from torchmetrics import Accuracy
        metric_computer = Accuracy(task='multiclass', num_classes=out_channels)
        metric = 'Acc'
    else:
        from torchmetrics import MeanSquaredError
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
        'col_stats': dataset.col_stats,
        'metric_computer': metric_computer,
        'metric': metric,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'out_channels': out_channels,
        'device': device
    }





def tabnet_core_fn(material_outputs, config, task_type, gnn_stage=None):
    # 從上一階段獲取數據
    train_tensor_frame = material_outputs['train_tensor_frame']
    val_tensor_frame = material_outputs['val_tensor_frame']
    test_tensor_frame = material_outputs['test_tensor_frame']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    col_stats = material_outputs['col_stats']
    col_names_dict = train_tensor_frame.col_names_dict
    device = material_outputs['device']
    out_channels = material_outputs['out_channels']
    # 獲取TabNet的參數
    cat_emb_channels = config.get('cat_emb_channels', 2)
    print(f"Encoding with cat_emb_channels: {cat_emb_channels}")
    
    # 設置編碼器字典
    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(na_strategy=NAStrategy.MOST_FREQUENT),
        stype.numerical: StackEncoder(na_strategy=NAStrategy.MEAN),
    }
    
    # 創建TabNet的特徵編碼器
    feature_encoder = StypeWiseFeatureEncoder(
        out_channels=cat_emb_channels,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
    
    # 計算輸入通道數
    num_cols = sum([len(v) for v in col_names_dict.values()])
    in_channels = cat_emb_channels * num_cols
    print(f"Input channels: {in_channels}")
    
    # 創建批次標準化層
    bn = BatchNorm1d(in_channels).to(device)
    # 對批次數據進行編碼的函數
    def encode_batch(tf, debug=False):
        # 編碼特徵
        x, _ = feature_encoder(tf)
        if debug:
            print(f"[TabNet] Raw feature encoding: batch_size={x.shape[0]}, num_cols={x.shape[1]}, channels={x.shape[2]}")
        batch_size = x.shape[0]
        # 展平特徵
        x_flat = x.view(batch_size, math.prod(x.shape[1:]))
        if debug:
            print(f"[TabNet] After flattening: batch_size={x_flat.shape[0]}, flattened_features={x_flat.shape[1]}")
        # 應用批次標準化
        x_norm = bn(x_flat)
        if debug:
            print(f"[TabNet] After batch norm: batch_size={x_norm.shape[0]}, features={x_norm.shape[1]}")
        return x_norm
    # 獲取TabNet的參數
    split_feat_channels = config.get('channels', 128)
    split_attn_channels = config.get('channels', 128)
    num_layers = config.get('num_layers', 6)
    gamma = config.get('gamma', 1.2)
    num_shared_glu_layers = config.get('num_shared_glu_layers', 2)
    num_dependent_glu_layers = config.get('num_dependent_glu_layers', 2)
    patience = config.get('patience', 10)
    # 獲取模型參數
    channels = config.get('channels', 256)
    
    # 為 encoding 和 columnwise 階段創建 Self-Attention + DGM 組件（與 ExcelFormer 對齊）
    gnn_encoding_components = None
    gnn_columnwise_components = None
    
    if gnn_stage == 'encoding':
        # Encoding 階段：Self-Attention + DGM
        # 注意：attn_heads 必須能整除 cat_emb_channels (預設為 2)
        attn_heads = config.get('gnn_num_heads', 2)  # 改為 2 以適配 cat_emb_channels=2
        gnn_hidden = config.get('gnn_hidden', 64)
        dgm_k = config.get('dgm_k', 10)
        dgm_distance = config.get('dgm_distance', 'euclidean')
        gnn_dropout = config.get('gnn_dropout', 0.1)
        
        # 注意：encoding 階段操作的是 flattened features [batch, in_channels]
        # 需要先 reshape 回 [batch, num_cols, cat_emb_channels] 才能做列間 Self-Attention
        
        self_attn_enc = torch.nn.MultiheadAttention(embed_dim=cat_emb_channels, num_heads=attn_heads, batch_first=True).to(device)
        attn_norm_enc = torch.nn.LayerNorm(cat_emb_channels).to(device)
        column_embed_enc = torch.nn.Parameter(torch.randn(num_cols, cat_emb_channels, device=device))
        pool_query_enc = torch.nn.Parameter(torch.randn(cat_emb_channels, device=device))
        
        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x
        dgm_embed_f = DGMEmbedWrapper()
        dgm_module_enc = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
        
        gnn_enc = SimpleGCN(cat_emb_channels, gnn_hidden, cat_emb_channels, num_layers=2).to(device)
        
        self_attn_out_enc = torch.nn.MultiheadAttention(embed_dim=cat_emb_channels, num_heads=attn_heads, batch_first=True).to(device)
        gcn_to_attn_enc = torch.nn.Linear(cat_emb_channels, cat_emb_channels).to(device)
        attn_out_norm_enc = torch.nn.LayerNorm(cat_emb_channels).to(device)
        ffn_pre_enc = torch.nn.Sequential(
            torch.nn.Linear(cat_emb_channels, cat_emb_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(gnn_dropout),
            torch.nn.Linear(cat_emb_channels * 2, cat_emb_channels),
        ).to(device)
        ffn_post_enc = torch.nn.Sequential(
            torch.nn.Linear(cat_emb_channels, cat_emb_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(gnn_dropout),
            torch.nn.Linear(cat_emb_channels * 2, cat_emb_channels),
        ).to(device)
        fusion_alpha_param_enc = torch.nn.Parameter(torch.tensor(-0.847, device=device))
        
        gnn_encoding_components = {
            'self_attn': self_attn_enc,
            'attn_norm': attn_norm_enc,
            'column_embed': column_embed_enc,
            'pool_query': pool_query_enc,
            'dgm_module': dgm_module_enc,
            'gnn': gnn_enc,
            'self_attn_out': self_attn_out_enc,
            'gcn_to_attn': gcn_to_attn_enc,
            'attn_out_norm': attn_out_norm_enc,
            'ffn_pre': ffn_pre_enc,
            'ffn_post': ffn_post_enc,
            'fusion_alpha_param': fusion_alpha_param_enc,
        }
        print(f"✓ Encoding-Self-Attention-DGM Pipeline created (TabNet):")
        print(f"  - Multi-Head Self-Attention (heads={attn_heads}, 列間交互)")
        print(f"  - DGM_d (k={dgm_k}, distance={dgm_distance})")
        print(f"  - Batch GCN (input={cat_emb_channels}, hidden={gnn_hidden}, output={cat_emb_channels})")
        print(f"  - Learnable Fusion Alpha (init=-0.847, trainable)")
    
    if gnn_stage == 'columnwise':
        # Columnwise 階段：Self-Attention + DGM（操作在 split_feat_channels）
        attn_heads = config.get('gnn_num_heads', 4)
        gnn_hidden = config.get('gnn_hidden', 64)
        dgm_k = config.get('dgm_k', 10)
        dgm_distance = config.get('dgm_distance', 'euclidean')
        gnn_dropout = config.get('gnn_dropout', 0.1)
        
        # Columnwise 處理的是 feature_outputs (每層輸出 [batch, split_feat_channels])
        # 無法直接做列間 self-attention，因為已經是 aggregated feature
        # 改為：將所有層的 feature 堆疊成 [batch, num_layers, split_feat_channels]，做層間 self-attention
        
        self_attn_col = torch.nn.MultiheadAttention(embed_dim=split_feat_channels, num_heads=attn_heads, batch_first=True).to(device)
        attn_norm_col = torch.nn.LayerNorm(split_feat_channels).to(device)
        pool_query_col = torch.nn.Parameter(torch.randn(split_feat_channels, device=device))
        
        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x
        dgm_embed_f = DGMEmbedWrapper()
        dgm_module_col = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
        
        gnn_col = SimpleGCN(split_feat_channels, gnn_hidden, split_feat_channels, num_layers=2).to(device)
        
        self_attn_out_col = torch.nn.MultiheadAttention(embed_dim=split_feat_channels, num_heads=attn_heads, batch_first=True).to(device)
        gcn_to_attn_col = torch.nn.Linear(split_feat_channels, split_feat_channels).to(device)
        attn_out_norm_col = torch.nn.LayerNorm(split_feat_channels).to(device)
        ffn_pre_col = torch.nn.Sequential(
            torch.nn.Linear(split_feat_channels, split_feat_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(gnn_dropout),
            torch.nn.Linear(split_feat_channels * 2, split_feat_channels),
        ).to(device)
        ffn_post_col = torch.nn.Sequential(
            torch.nn.Linear(split_feat_channels, split_feat_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(gnn_dropout),
            torch.nn.Linear(split_feat_channels * 2, split_feat_channels),
        ).to(device)
        fusion_alpha_param_col = torch.nn.Parameter(torch.tensor(-0.847, device=device))
        
        gnn_columnwise_components = {
            'self_attn': self_attn_col,
            'attn_norm': attn_norm_col,
            'pool_query': pool_query_col,
            'dgm_module': dgm_module_col,
            'gnn': gnn_col,
            'self_attn_out': self_attn_out_col,
            'gcn_to_attn': gcn_to_attn_col,
            'attn_out_norm': attn_out_norm_col,
            'ffn_pre': ffn_pre_col,
            'ffn_post': ffn_post_col,
            'fusion_alpha_param': fusion_alpha_param_col,
        }
        print(f"✓ Columnwise-Self-Attention-DGM Pipeline created (TabNet):")
        print(f"  - Multi-Head Self-Attention (heads={attn_heads}, 層間交互)")
        print(f"  - DGM_d (k={dgm_k}, distance={dgm_distance})")
        print(f"  - Batch GCN (input={split_feat_channels}, hidden={gnn_hidden}, output={split_feat_channels})")
        print(f"  - Learnable Fusion Alpha (init=-0.847, trainable)")
    
    print(f"=== TabNet Architecture Info ===")
    print(f"Input channels (after encoding): {in_channels}")
    print(f"Split feature channels: {split_feat_channels}")
    print(f"Split attention channels: {split_attn_channels}")
    print(f"Number of layers: {num_layers}")
    print(f"Output channels: {out_channels}")
    if gnn_encoding_components is not None:
        print(f"GNN stage: encoding (Self-Attention + DGM)")
    if gnn_columnwise_components is not None:
        print(f"GNN stage: columnwise (Self-Attention + DGM)")
    print("================================")
    
    print(f"Building TabNet with {num_layers} layers, split channels: {split_feat_channels}")
    
    # 創建共享的GLU塊
    shared_glu_block = None
    if num_shared_glu_layers > 0:
        shared_glu_block = GLUBlock(
            in_channels=in_channels,
            out_channels=split_feat_channels + split_attn_channels,
            num_glu_layers=num_shared_glu_layers,
            no_first_residual=True,
        ).to(device)
    else:
        shared_glu_block = Identity().to(device)
    
    # 創建特徵變換器
    feat_transformers = ModuleList([
        FeatureTransformer(
            in_channels,
            split_feat_channels + split_attn_channels,
            num_dependent_glu_layers=num_dependent_glu_layers,
            shared_glu_block=shared_glu_block,
        ).to(device)
        for _ in range(num_layers + 1)
    ])
    
    # 創建注意力變換器
    attn_transformers = ModuleList([
        AttentiveTransformer(
            in_channels=split_attn_channels,
            out_channels=in_channels,
        ).to(device)
        for _ in range(num_layers)
    ])
    
    # 定義列間交互處理函數
    def process_batch_interaction(x, return_reg=False):
        """
        處理一個批次的特徵通過特徵變換器和注意力變換器
        
        參數:
        - x: 編碼後的特徵張量 [batch_size, in_channels]
        - return_reg: 是否返回熵正則化
        
        返回:
        - feature_outputs: 列表，包含每層的輸出特徵
        - reg: 熵正則化值 (如果return_reg=True)
        """
        batch_size = x.shape[0]
        prior = torch.ones_like(x)
        reg = torch.tensor(0., device=x.device)
        
        # 第一層特徵變換器處理，分離注意力部分
        attention_x = feat_transformers[0](x)
        attention_x = attention_x[:, split_feat_channels:]
        
        feature_outputs = []
        for i in range(num_layers):
            # 應用注意力變換器生成掩碼
            attention_mask = attn_transformers[i](attention_x, prior)
            
            # 應用掩碼到輸入特徵
            masked_x = attention_mask * x
            
            # 應用特徵變換器
            out = feat_transformers[i + 1](masked_x)
            
            # 分離特徵和注意力部分
            feature_x = F.relu(out[:, :split_feat_channels])
            feature_outputs.append(feature_x)
            attention_x = out[:, split_feat_channels:]
            
            # 更新prior
            prior = (gamma - attention_mask) * prior
            
            # 計算熵正則化
            if return_reg and batch_size > 0:
                entropy = -torch.sum(
                    attention_mask * torch.log(attention_mask + 1e-15),
                    dim=1).mean()
                reg += entropy
        
        if return_reg:
            return feature_outputs, reg / num_layers
        else:
            return feature_outputs
        

    is_classification = material_outputs['is_classification']
    is_binary_class = material_outputs['is_binary_class']
    metric_computer = material_outputs['metric_computer']
    metric = material_outputs['metric']

    # 創建輸出層
    lin = Linear(split_feat_channels, out_channels).to(device)
    
    # 重置參數
    lin.reset_parameters()
    



    # 定義完整的TabNet前向傳播函數
    def forward(tf, return_reg=False, debug=False):
        # Stage 0: 編碼特徵
        x = encode_batch(tf, debug=debug)  # 返回 (batch_size, in_channels)
        batch_size = x.shape[0]
        if debug:
            print(f"[TabNet] After encoding: batch_size={batch_size}, channels={x.shape[1]} (flattened)")
        
        # Stage 1: Encoding階段 Self-Attention + DGM 處理（與 ExcelFormer 對齊）
        if gnn_stage == 'encoding' and gnn_encoding_components is not None:
            if debug:
                print(f"[TabNet] Applying Self-Attention+DGM at encoding stage")
            
            # Step 1: Reshape flattened features 回 [batch, num_cols, cat_emb_channels]
            x_reshaped = x.view(batch_size, num_cols, cat_emb_channels)
            
            # Step 2: Self-Attention 列間交互
            comp = gnn_encoding_components
            tokens = x_reshaped + comp['column_embed'].unsqueeze(0)
            tokens_norm = comp['attn_norm'](tokens)
            attn_out1, _ = comp['self_attn'](tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            ffn_out1 = comp['ffn_pre'](comp['attn_norm'](tokens_attn))
            tokens_attn = tokens_attn + ffn_out1
            
            # Step 3: Attention Pooling [batch, num_cols, cat_emb_channels] → [batch, cat_emb_channels]
            pool_logits = (tokens_attn * comp['pool_query']).sum(dim=-1) / math.sqrt(cat_emb_channels)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)
            
            # Step 4: Mini-batch DGM 動態建圖
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)
            if hasattr(comp['dgm_module'], 'k'):
                comp['dgm_module'].k = int(min(int(comp['dgm_module'].k), max(1, batch_size - 1)))
            x_dgm, edge_index_dgm, _ = comp['dgm_module'](x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])
            
            # Step 5: Batch GCN
            x_gnn_out = comp['gnn'](x_dgm, edge_index_dgm)
            
            # Step 6: Self-Attention 解碼 [batch, cat_emb_channels] → [batch, num_cols, cat_emb_channels]
            gcn_ctx = comp['gcn_to_attn'](x_gnn_out).unsqueeze(1)
            tokens_with_ctx = tokens_attn + gcn_ctx
            tokens_ctx_norm = comp['attn_out_norm'](tokens_with_ctx)
            attn_out2, _ = comp['self_attn_out'](tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
            tokens_mid = tokens_with_ctx + attn_out2
            ffn_out2 = comp['ffn_post'](comp['attn_out_norm'](tokens_mid))
            tokens_out = tokens_mid + ffn_out2
            
            # Step 7: 殘差融合
            fusion_alpha = torch.sigmoid(comp['fusion_alpha_param'])
            x_reshaped = x_reshaped + fusion_alpha * tokens_out
            
            # Step 8: Flatten 回 [batch, in_channels]
            x = x_reshaped.view(batch_size, in_channels)
            
            if debug:
                print(f"[TabNet] After encoding Self-Attention+DGM: batch_size={batch_size}, channels={x.shape[1]}")
        
        # Stage 2: 通過特徵變換器和注意力變換器處理
        if debug:
            print(f"[TabNet] Starting TabNet interaction layers...")
        if return_reg:
            feature_outputs, reg = process_batch_interaction(x, return_reg=True)
        else:
            feature_outputs = process_batch_interaction(x, return_reg=False)
        
        if debug:
            print(f"[TabNet] After TabNet layers: {len(feature_outputs)} feature outputs")
            for i, feat in enumerate(feature_outputs):
                print(f"[TabNet]   Layer {i+1}: batch_size={feat.shape[0]}, channels={feat.shape[1]}")
            
        # Stage 3: Columnwise階段 Self-Attention + DGM 處理（與 ExcelFormer 對齊）
        if gnn_stage == 'columnwise' and gnn_columnwise_components is not None:
            if debug:
                print(f"[TabNet] Applying Self-Attention+DGM at columnwise stage")
            
            # 將所有層的 feature 堆疊成 [batch, num_layers, split_feat_channels]
            feat_stack = torch.stack(feature_outputs, dim=1)  # [batch, num_layers, split_feat_channels]
            comp = gnn_columnwise_components
            
            # Step 1: Self-Attention 層間交互
            tokens_norm = comp['attn_norm'](feat_stack)
            attn_out1, _ = comp['self_attn'](tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = feat_stack + attn_out1
            ffn_out1 = comp['ffn_pre'](comp['attn_norm'](tokens_attn))
            tokens_attn = tokens_attn + ffn_out1
            
            # Step 2: Attention Pooling [batch, num_layers, split_feat_channels] → [batch, split_feat_channels]
            pool_logits = (tokens_attn * comp['pool_query']).sum(dim=-1) / math.sqrt(split_feat_channels)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)
            
            # Step 3: Mini-batch DGM 動態建圖
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)
            if hasattr(comp['dgm_module'], 'k'):
                comp['dgm_module'].k = int(min(int(comp['dgm_module'].k), max(1, batch_size - 1)))
            x_dgm, edge_index_dgm, _ = comp['dgm_module'](x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])
            
            # Step 4: Batch GCN
            x_gnn_out = comp['gnn'](x_dgm, edge_index_dgm)
            
            # Step 5: Self-Attention 解碼 [batch, split_feat_channels] → [batch, num_layers, split_feat_channels]
            gcn_ctx = comp['gcn_to_attn'](x_gnn_out).unsqueeze(1)
            tokens_with_ctx = tokens_attn + gcn_ctx
            tokens_ctx_norm = comp['attn_out_norm'](tokens_with_ctx)
            attn_out2, _ = comp['self_attn_out'](tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
            tokens_mid = tokens_with_ctx + attn_out2
            ffn_out2 = comp['ffn_post'](comp['attn_out_norm'](tokens_mid))
            tokens_out = tokens_mid + ffn_out2
            
            # Step 6: 殘差融合
            fusion_alpha = torch.sigmoid(comp['fusion_alpha_param'])
            feat_stack = feat_stack + fusion_alpha * tokens_out
            
            # Step 7: Unstack 回 list of [batch, split_feat_channels]
            feature_outputs = [feat_stack[:, i, :] for i in range(feat_stack.shape[1])]
            
            if debug:
                print(f"[TabNet] After columnwise Self-Attention+DGM: {len(feature_outputs)} layers processed")
            
        # Stage 4: 合併所有層的特徵輸出
        out = sum(feature_outputs)
        if debug:
            print(f"[TabNet] After summing all layers: batch_size={out.shape[0]}, channels={out.shape[1]}")
        
        # Stage 5: 應用輸出層
        out = lin(out)
        if debug:
            print(f"[TabNet] Final output: batch_size={out.shape[0]}, out_channels={out.shape[1]}")
        
        if return_reg:
            return out, reg
        else:
            return out
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.001)  # TabNet默認學習率
    
    # 收集所有參數（去除重複，避免optimizer警告）
    all_params = []
    all_params.extend(feature_encoder.parameters())
    all_params.extend(bn.parameters())
    
    # 避免重複添加shared_glu_block的參數
    for ft in feat_transformers:
        all_params.extend(ft.parameters())
    for at in attn_transformers:
        all_params.extend(at.parameters())
    all_params.extend(lin.parameters())
    
    # 添加GNN參數（Self-Attention + DGM 組件）
    if gnn_encoding_components is not None:
        comp = gnn_encoding_components
        all_params.extend(comp['self_attn'].parameters())
        all_params.extend(comp['attn_norm'].parameters())
        all_params.append(comp['column_embed'])
        all_params.append(comp['pool_query'])
        all_params.extend(comp['dgm_module'].parameters())
        all_params.extend(comp['gnn'].parameters())
        all_params.extend(comp['self_attn_out'].parameters())
        all_params.extend(comp['gcn_to_attn'].parameters())
        all_params.extend(comp['attn_out_norm'].parameters())
        all_params.extend(comp['ffn_pre'].parameters())
        all_params.extend(comp['ffn_post'].parameters())
        all_params.append(comp['fusion_alpha_param'])
    if gnn_columnwise_components is not None:
        comp = gnn_columnwise_components
        all_params.extend(comp['self_attn'].parameters())
        all_params.extend(comp['attn_norm'].parameters())
        all_params.append(comp['pool_query'])
        all_params.extend(comp['dgm_module'].parameters())
        all_params.extend(comp['gnn'].parameters())
        all_params.extend(comp['self_attn_out'].parameters())
        all_params.extend(comp['gcn_to_attn'].parameters())
        all_params.extend(comp['attn_out_norm'].parameters())
        all_params.extend(comp['ffn_pre'].parameters())
        all_params.extend(comp['ffn_post'].parameters())
        all_params.append(comp['fusion_alpha_param'])
    
    # 去除重複參數，避免optimizer警告
    unique_params = list(set(all_params))
    print(f"Total parameters collected: {len(all_params)}, Unique parameters: {len(unique_params)}")
    
    optimizer = torch.optim.Adam(unique_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        feature_encoder.train()
        bn.train()
        for ft in feat_transformers:
            ft.train()
        for at in attn_transformers:
            at.train()
        if gnn_encoding_components is not None:
            comp = gnn_encoding_components
            comp['self_attn'].train()
            comp['attn_norm'].train()
            comp['dgm_module'].train()
            comp['gnn'].train()
            comp['self_attn_out'].train()
            comp['gcn_to_attn'].train()
            comp['attn_out_norm'].train()
            comp['ffn_pre'].train()
            comp['ffn_post'].train()
        if gnn_columnwise_components is not None:
            comp = gnn_columnwise_components
            comp['self_attn'].train()
            comp['attn_norm'].train()
            comp['dgm_module'].train()
            comp['gnn'].train()
            comp['self_attn_out'].train()
            comp['gcn_to_attn'].train()
            comp['attn_out_norm'].train()
            comp['ffn_pre'].train()
            comp['ffn_post'].train()
        lin.train()
        
        loss_accum = total_count = 0
        first_batch = True
        
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            # 只在第一個epoch的第一個batch啟用調試
            debug = (epoch == 1 and first_batch)
            pred, reg = forward(tf, return_reg=True, debug=debug)
            first_batch = False
            
            if is_classification:
                loss = F.cross_entropy(pred, tf.y)
            else:
                loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
            
            # 添加熵正則化
            loss = loss + 0.01 * reg  # 熵正則化係數
            
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
        
        return loss_accum / total_count
    
    # 定義測試函數
    @torch.no_grad()
    def test(loader):
        # 設置為評估模式
        feature_encoder.eval()
        bn.eval()
        for ft in feat_transformers:
            ft.eval()
        for at in attn_transformers:
            at.eval()
        if gnn_encoding_components is not None:
            comp = gnn_encoding_components
            comp['self_attn'].eval()
            comp['attn_norm'].eval()
            comp['dgm_module'].eval()
            comp['gnn'].eval()
            comp['self_attn_out'].eval()
            comp['gcn_to_attn'].eval()
            comp['attn_out_norm'].eval()
            comp['ffn_pre'].eval()
            comp['ffn_post'].eval()
        if gnn_columnwise_components is not None:
            comp = gnn_columnwise_components
            comp['self_attn'].eval()
            comp['attn_norm'].eval()
            comp['dgm_module'].eval()
            comp['gnn'].eval()
            comp['self_attn_out'].eval()
            comp['gcn_to_attn'].eval()
            comp['attn_out_norm'].eval()
            comp['ffn_pre'].eval()
            comp['ffn_post'].eval()
        lin.eval()
        
        metric_computer.reset()
        
        for tf in loader:
            tf = tf.to(device)
            pred = forward(tf, debug=False)
            
            if is_binary_class:
                metric_computer.update(pred[:, 1], tf.y)
            elif is_classification:
                pred_class = pred.argmax(dim=-1)
                metric_computer.update(pred_class, tf.y)
            else:
                metric_computer.update(pred.view(-1), tf.y.view(-1))
        
        if is_classification:
            return metric_computer.compute().item()
        else:
            return metric_computer.compute().item()**0.5
    
    # 初始化最佳指標
    if is_classification:
        best_val_metric = 0
        best_test_metric = 0
    else:
        best_val_metric = float('inf')
        best_test_metric = float('inf')
    
    # 記錄訓練過程
    best_epoch = 0
    early_stop_counter = 0
    early_stop_epochs = 0  # 初始化 early_stop_epochs
    train_losses = []
    train_metrics = []
    val_metrics = []
    
    # 訓練循環
    epochs = config.get('epochs', 200)
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        train_metric = test(train_loader)
        val_metric = test(val_loader)
        # test_metric = test(test_loader)
        
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        # test_metrics.append(test_metric)
        
        improved = (val_metric > best_val_metric) if is_classification else (val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if is_classification and val_metric > best_val_metric:
            best_val_metric = val_metric
        elif not is_classification and val_metric < best_val_metric:
            best_val_metric = val_metric

        print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
              f'Val {metric}: {val_metric:.4f}')

        lr_scheduler.step()
        if early_stop_counter >= patience:
            early_stop_epochs = epoch
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 決定最終 metric 輸出 (decoding 階段已經整合到聯合訓練中，不需要單獨的 gnn_decoding_eval)
    final_metric = best_val_metric
    print(f'Best Val {metric}: {final_metric:.4f}')
    test_metric = test(test_loader)
    
    return {
        'train_losses': train_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_metric': best_val_metric,
        'final_metric': final_metric,
        'best_test_metric': test_metric,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'metric_computer': metric_computer,
        'metric': metric,
        'early_stop_epochs': early_stop_epochs,
        'gnn_early_stop_epochs': material_outputs.get('gnn_early_stop_epochs', 0),
    }

def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    主函數：按順序調用四個階段函數
    """
    print("TabNet - 五階段執行")
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    gnn_early_stop_epochs = 0  # 初始化變數
    try:
        # 階段0: 開始
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        if gnn_stage=='start':
            # 在 start_fn 和 materialize_fn 之間插入 GNN
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)
        # 階段1: Materialization
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        if gnn_stage == 'materialize':
            # 在 materialize_fn 和 encoding_fn 之間插入 GNN
            train_tensor_frame = material_outputs['train_tensor_frame']
            val_tensor_frame = material_outputs['val_tensor_frame']
            test_tensor_frame = material_outputs['test_tensor_frame']
            (train_loader, val_loader, test_loader,col_stats, dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs) = gnn_after_materialize_fn(
                train_tensor_frame, val_tensor_frame, test_tensor_frame, config, dataset_results['dataset'], task_type)
            material_outputs.update({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'col_stats': col_stats,
                'dataset': dataset,
                'train_tensor_frame': train_tensor_frame,
                'val_tensor_frame': val_tensor_frame,
                'test_tensor_frame': test_tensor_frame,
            })
            material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs

        results=tabnet_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)
    
        # # 階段2: Encoding
        # encoding_outputs = encoding_fn(material_outputs, config)
        # # 這裡可以插入GNN處理編碼後的數據
        # # encoding_outputs = gnn_process(encoding_outputs, config)
        # # 階段3: Column-wise Interaction
        # columnwise_outputs = columnwise_fn(encoding_outputs, config)
        # # 這裡可以插入GNN處理列間交互後的數據
        # # columnwise_outputs = gnn_process(columnwise_outputs, config)
        # # 階段4: Decoding
        # results = decoding_fn(columnwise_outputs, config)
    except Exception as e:
        is_classification = task_type in ['binclass', 'multiclass']
        results = {
            'train_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': [],
            'best_val_metric': float('-inf') if is_classification else float('inf'),
            'best_test_metric': float('-inf') if is_classification else float('inf'),
            'early_stop_epochs': 0,
            'gnn_early_stop_epochs': gnn_early_stop_epochs,
            'error': str(e),
        }
    return results


#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models tabnet --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models tabnet --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models tabnet --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models tabnet --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models tabnet --gnn_stages all --epochs 2
