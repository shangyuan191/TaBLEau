"""
TabM (Tabular Model with Multiple predictions) 集成到 TaBLEau 框架

五階段拆分設計：
- start: dummy stage（為了讓 GNN 能在 materialize 之前插入）
- materialize: 數據預處理 + 特徵嵌入（可在此做 precompute GNN，採 train-only 訓練、對 val/test 做 inductive 推論）
- encoding: EnsembleView + Backbone 前半部分（可在此插入 row-level Dynamic-Graph + Attention，與這些層 joint-train）
- columnwise: Backbone 後半部分（同上，可插入 row-level Dynamic-Graph + Attention，與這些層 joint-train）
- decoding: 以 GNN 作為 decoder 的端到端 joint training/推論（row-level self-attn pooling → DGM 動態建圖 → GCN decoder）
"""

import sys
import os
import math
from pathlib import Path

# 添加 TabM 包的路徑（避免硬編碼到其他使用者目錄）
# 期望 repo 結構：.../ModelComparison/TaBLEau/models/custom/tabm.py
#            TabM 原始碼：.../ModelComparison/tabm/tabm.py
_this_file = Path(__file__).resolve()
_modelcomparison_root = _this_file.parents[3]  # .../ModelComparison
_tabm_dir_candidates = [
    _modelcomparison_root / 'tabm',
    Path('/home/skyler/ModelComparison/tabm'),
]
for _tabm_dir in _tabm_dir_candidates:
    try:
        if _tabm_dir.exists() and str(_tabm_dir) not in sys.path:
            sys.path.insert(0, str(_tabm_dir))
            break
    except Exception:
        continue

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

# DGM 動態圖模組（對齊 ExcelFormer 風格）
sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
try:
    from DGMlib.layers import DGM_d
    DGM_AVAILABLE = True
except Exception:
    DGM_AVAILABLE = False
    print("[WARNING] DGM_d not available for TabM; will fall back to kNN graphs where needed")


def resolve_device(config: dict) -> torch.device:
    """統一的裝置選擇函式：從 config 選擇 GPU，否則回退到可用裝置（對齊 ExcelFormer）。"""
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
        print("[DEVICE] resolve_device: Using cuda (default)")
        return device
    print("[DEVICE] resolve_device: Using cpu (CUDA not available)")
    return torch.device('cpu')


def compute_adaptive_dgm_k(num_samples: int, num_features: int, dataset_name: str = '', use_config_override: bool = True) -> int:
    """對齊 ExcelFormer：根據 N/D 自適應計算 DGM 候選池大小。"""
    base_k = int(np.sqrt(max(num_samples, 1)))
    feature_factor = 1.0 + np.log1p(max(num_features, 1)) / 10
    adjusted_k = int(base_k * feature_factor)
    if num_samples < 500:
        density_factor = 1.3
    elif num_samples > 5000:
        density_factor = 0.9
    else:
        density_factor = 1.0
    adaptive_k = int(adjusted_k * density_factor)
    upper_limit = min(30, max(15, int(4 * np.sqrt(max(num_samples, 1)))))
    if num_samples < 1000:
        upper_limit = min(20, int(3 * np.sqrt(max(num_samples, 1))))
    adaptive_k = max(5, min(adaptive_k, upper_limit))
    print(f"[DGM-K] dataset={dataset_name or 'unknown'} N={num_samples} D={num_features} -> k={adaptive_k}")
    return adaptive_k

# TabM 相關導入
import tabm
from tabm import (
    TabM,
    EnsembleView,
    LinearEnsemble,
    MLPBackboneBatchEnsemble,
    ElementwiseAffine
)
import rtdl_num_embeddings
from rtdl_num_embeddings import (
    PiecewiseLinearEmbeddings,
    PeriodicEmbeddings,
    LinearReLUEmbeddings
)


class SimpleGCN(torch.nn.Module):
    """簡化的 GCN，用於聯合訓練（對齊 ExcelFormer：可配置層數）。"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (max(2, int(num_layers)) - 1) + [out_dim]
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(GCNConv(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def knn_graph(x: torch.Tensor, k: int, directed: bool = False) -> torch.Tensor:
    """構建 k-NN 圖（對齊 ExcelFormer：detach + 可選雙向邊 + 自迴路備援）。"""
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    N = int(x_np.shape[0])
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    actual_k = min(int(k), N - 1)
    nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_list = []
    for i in range(N):
        for j in indices[i][1:]:
            edge_list.append([i, int(j)])
            if not directed:
                edge_list.append([int(j), i])
    if len(edge_list) == 0:
        edge_list = [[i, i] for i in range(N)]
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index.device
    if edge_index.numel() == 0:
        loops = torch.arange(num_nodes, device=device)
        return torch.stack([loops, loops], dim=0)
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    loops = torch.arange(num_nodes, device=device)
    self_edges = torch.stack([loops, loops], dim=0)
    ei = torch.cat([edge_index, rev, self_edges], dim=1)
    edge_ids = ei[0] * num_nodes + ei[1]
    unique_ids = torch.unique(edge_ids, sorted=False)
    return torch.stack([unique_ids // num_nodes, unique_ids % num_nodes], dim=0)


def start_fn(train_df, val_df, test_df):
    """
    Start 階段 - dummy stage
    不做任何處理，只是為了讓 GNN 能在 materialize 之前插入
    """
    return train_df, val_df, test_df


def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    """
    在 start 和 materialize 之間插入 GNN
    採 train-only 訓練（val/test 不參與 supervision），並對每個 split 做 inductive 推論後組回 df 往下游傳遞
    """
    print("[TABM][START-GNN-DGM] Executing Dynamic-Graph + Attention (DGM + Self-Attn) between start_fn and materialize_fn")
    device = resolve_device(config)

    # Align with excelformer.py: fixed dgm_k default (user override allowed)
    dgm_k = int(config['dgm_k']) if 'dgm_k' in config else 10
    dgm_distance = config.get('dgm_distance', 'euclidean')

    gnn_epochs = int(config.get('epochs', 200))
    patience = int(config.get('gnn_patience', 10))
    loss_threshold = float(config.get('gnn_loss_threshold', 1e-4))
    attn_dim = int(config.get('gnn_attn_dim', config.get('gnn_hidden', 64)))
    gnn_hidden = int(config.get('gnn_hidden', 64))
    gnn_out_dim = int(config.get('gnn_out_dim', attn_dim))
    attn_heads = int(config.get('gnn_num_heads', 4))
    lr = float(config.get('gnn_lr', 1e-3))

    # Split tensors (train-only training, inductive inference per split)
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    feature_cols = [c for c in all_df.columns if c != 'target']
    num_cols = len(feature_cols)
    x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
    y_train_np = train_df['target'].values
    y_val_np = val_df['target'].values
    y_test_np = test_df['target'].values

    # Determine out_dim consistently
    if task_type in ['binclass', 'multiclass']:
        y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
        num_classes = len(pd.unique(y_all_np))
        if task_type == 'binclass' and num_classes != 2:
            num_classes = 2
        out_dim = int(num_classes)
    else:
        out_dim = 1

    if attn_heads <= 0:
        attn_heads = 1

    # Dynamic-Graph + Attention token pipeline on features
    input_proj = torch.nn.Linear(1, attn_dim).to(device)
    out_proj = torch.nn.Linear(attn_dim, 1).to(device)
    attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    # DGM_d (train-only k cap)
    if not DGM_AVAILABLE:
        raise ImportError("DGM is required for the Dynamic-Graph + Attention START-GNN stage, but DGM is not available.")

    class DGMEmbedWrapper(torch.nn.Module):
        def forward(self, x, A=None):
            return x

    dgm_k_train = int(min(dgm_k, max(1, int(x_train.shape[0]) - 1)))
    dgm_module = DGM_d(DGMEmbedWrapper(), k=dgm_k_train, distance=dgm_distance).to(device)

    gnn_layers = int(config.get('gnn_layers', 2))
    gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim, num_layers=gnn_layers).to(device)
    gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
    pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)

    params = list(input_proj.parameters()) + list(out_proj.parameters()) + list(attn_in.parameters()) + list(attn_out.parameters())
    params += list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters())
    if dgm_module is not None:
        params += list(dgm_module.parameters())
    params += [column_embed, pool_query]
    optimizer = torch.optim.Adam(params, lr=lr)

    best_val_loss = float('inf')
    best_states = None
    patience_counter = 0
    gnn_early_stop_epochs = 0

    def forward_pass(x_tensor: torch.Tensor):
        # x_tensor: [Ns, num_cols]
        Ns = int(x_tensor.shape[0])
        x_in = input_proj(x_tensor.unsqueeze(-1))
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_w = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_w.unsqueeze(-1) * tokens_attn).sum(dim=1)
        row_emb_std = _standardize(row_emb, dim=0)

        row_emb_batched = row_emb_std.unsqueeze(0)
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)
        edge_index = _symmetrize_and_self_loop(edge_index_dgm.to(device), Ns)

        gcn_out = gnn(row_emb_dgm, edge_index)
        logits = pred_head(gcn_out)
        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)
        tokens_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_ctx, tokens_ctx, tokens_ctx)
        recon = out_proj(tokens_out).squeeze(-1)

        return logits, recon, logprobs_dgm

    for epoch in range(gnn_epochs):
        input_proj.train(); out_proj.train(); attn_in.train(); attn_out.train(); gcn_to_attn.train(); pred_head.train(); gnn.train()
        if dgm_module is not None:
            dgm_module.train()
        optimizer.zero_grad()
        logits, _, logprobs_dgm = forward_pass(x_train)
        if task_type in ['binclass', 'multiclass']:
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            train_loss = F.cross_entropy(logits, y_train)
        else:
            y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
            train_loss = F.mse_loss(logits.squeeze(-1), y_train)
        # DGM regularizer (aligned with excelformer.py)
        train_loss = train_loss + (-logprobs_dgm.mean() * 0.01)
        train_loss.backward()
        optimizer.step()

        input_proj.eval(); out_proj.eval(); attn_in.eval(); attn_out.eval(); gcn_to_attn.eval(); pred_head.eval(); gnn.eval()
        if dgm_module is not None:
            dgm_module.eval()
        with torch.no_grad():
            logits_val, _, _ = forward_pass(x_val)
            if task_type in ['binclass', 'multiclass']:
                y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
                val_loss = F.cross_entropy(logits_val, y_val)
            else:
                y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
                val_loss = F.mse_loss(logits_val.squeeze(-1), y_val)
        val_loss_val = float(val_loss.item())
        improved = val_loss_val < best_val_loss - loss_threshold
        print(f"[TABM][START-GNN-DGM] Epoch {epoch+1}/{gnn_epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss_val:.4f}{' ↓ (improved)' if improved else ''}")

        if improved:
            best_val_loss = val_loss_val
            patience_counter = 0
            best_states = {
                'input_proj': input_proj.state_dict(),
                'out_proj': out_proj.state_dict(),
                'attn_in': attn_in.state_dict(),
                'attn_out': attn_out.state_dict(),
                'gnn': gnn.state_dict(),
                'gcn_to_attn': gcn_to_attn.state_dict(),
                'pred_head': pred_head.state_dict(),
                'column_embed': column_embed.detach().clone(),
                'pool_query': pool_query.detach().clone(),
                'dgm_module': (dgm_module.state_dict() if dgm_module is not None else None),
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[TABM][START-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})")
                gnn_early_stop_epochs = epoch + 1
                break

    if best_states is not None:
        input_proj.load_state_dict(best_states['input_proj'])
        out_proj.load_state_dict(best_states['out_proj'])
        attn_in.load_state_dict(best_states['attn_in'])
        attn_out.load_state_dict(best_states['attn_out'])
        gnn.load_state_dict(best_states['gnn'])
        gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
        pred_head.load_state_dict(best_states['pred_head'])
        if best_states.get('dgm_module') is not None and dgm_module is not None:
            dgm_module.load_state_dict(best_states['dgm_module'])
        with torch.no_grad():
            column_embed.copy_(best_states['column_embed'])
            pool_query.copy_(best_states['pool_query'])

    input_proj.eval(); out_proj.eval(); attn_in.eval(); attn_out.eval(); gcn_to_attn.eval(); pred_head.eval(); gnn.eval()
    if dgm_module is not None:
        dgm_module.eval()
    with torch.no_grad():
        _, recon_train, _ = forward_pass(x_train)
        _, recon_val, _ = forward_pass(x_val)
        _, recon_test, _ = forward_pass(x_test)
        train_out = recon_train.detach().cpu().numpy()
        val_out = recon_val.detach().cpu().numpy()
        test_out = recon_test.detach().cpu().numpy()

    train_df_gnn = pd.DataFrame(train_out, columns=feature_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(val_out, columns=feature_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(test_out, columns=feature_cols, index=test_df.index)
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values
    return train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs


def gnn_after_materialize_fn(X_train, y_train, X_val, y_val, X_test, y_test, config, task_type):
    """
    在 materialize 後插入 GNN
    採 train-only 訓練（val/test 不參與 supervision），並對每個 split 做 inductive 推論
    """
    print("[TABM][MATERIALIZE-GNN-DGM] Executing Dynamic-Graph + Attention (DGM + Self-Attn) after materialize_fn")
    device = config.get('device', resolve_device(config))

    dgm_k = int(config['dgm_k']) if 'dgm_k' in config else 10
    dgm_distance = config.get('dgm_distance', 'euclidean')

    gnn_epochs = int(config.get('epochs', 200))
    patience = int(config.get('gnn_patience', 10))
    loss_threshold = float(config.get('gnn_loss_threshold', 1e-4))
    attn_dim = int(config.get('gnn_attn_dim', config.get('gnn_hidden', 64)))
    gnn_hidden = int(config.get('gnn_hidden', 64))
    gnn_out_dim = int(config.get('gnn_out_dim', attn_dim))
    attn_heads = int(config.get('gnn_num_heads', 4))
    lr = float(config.get('gnn_lr', 1e-3))

    # Train-only training, inductive inference per split
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    X_test = X_test.to(device)

    if task_type in ['binclass', 'multiclass']:
        y_all = torch.cat([y_train, y_val, y_test], dim=0).to(device).long()
        out_dim = int(torch.unique(y_all).numel())
    else:
        out_dim = 1

    if attn_heads <= 0:
        attn_heads = 1

    Fcols = int(X_train.shape[1])
    input_proj = torch.nn.Linear(1, attn_dim).to(device)
    out_proj = torch.nn.Linear(attn_dim, 1).to(device)
    attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    column_embed = torch.nn.Parameter(torch.randn(Fcols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    if not DGM_AVAILABLE:
        raise ImportError("DGM is required for the Dynamic-Graph + Attention MATERIALIZE-GNN stage, but DGM is not available.")

    class DGMEmbedWrapper(torch.nn.Module):
        def forward(self, x, A=None):
            return x

    dgm_k_train = int(min(dgm_k, max(1, int(X_train.shape[0]) - 1)))
    dgm_module = DGM_d(DGMEmbedWrapper(), k=dgm_k_train, distance=dgm_distance).to(device)

    gnn_layers = int(config.get('gnn_layers', 2))
    gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim, num_layers=gnn_layers).to(device)
    gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
    pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)

    params = list(input_proj.parameters()) + list(out_proj.parameters()) + list(attn_in.parameters()) + list(attn_out.parameters())
    params += list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters())
    if dgm_module is not None:
        params += list(dgm_module.parameters())
    params += [column_embed, pool_query]
    optimizer = torch.optim.Adam(params, lr=lr)

    best_val_loss = float('inf')
    best_states = None
    patience_counter = 0
    gnn_early_stop_epochs = 0

    def forward_pass(x_tensor: torch.Tensor):
        Ns = int(x_tensor.shape[0])
        x_in = input_proj(x_tensor.unsqueeze(-1))
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_w = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_w.unsqueeze(-1) * tokens_attn).sum(dim=1)
        row_emb_std = _standardize(row_emb, dim=0)

        row_emb_batched = row_emb_std.unsqueeze(0)
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)
        edge_index = _symmetrize_and_self_loop(edge_index_dgm.to(device), Ns)

        gcn_out = gnn(row_emb_dgm, edge_index)
        logits = pred_head(gcn_out)
        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)
        tokens_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_ctx, tokens_ctx, tokens_ctx)
        recon = out_proj(tokens_out).squeeze(-1)
        return logits, recon, logprobs_dgm

    for epoch in range(gnn_epochs):
        input_proj.train(); out_proj.train(); attn_in.train(); attn_out.train(); gcn_to_attn.train(); pred_head.train(); gnn.train()
        if dgm_module is not None:
            dgm_module.train()
        optimizer.zero_grad()
        logits, _, logprobs_dgm = forward_pass(X_train)
        if task_type in ['binclass', 'multiclass']:
            train_loss = F.cross_entropy(logits, y_train.long())
        else:
            train_loss = F.mse_loss(logits.squeeze(-1), y_train.float())
        # DGM regularizer (aligned with excelformer.py)
        train_loss = train_loss + (-logprobs_dgm.mean() * 0.01)
        train_loss.backward()
        optimizer.step()

        input_proj.eval(); out_proj.eval(); attn_in.eval(); attn_out.eval(); gcn_to_attn.eval(); pred_head.eval(); gnn.eval()
        if dgm_module is not None:
            dgm_module.eval()
        with torch.no_grad():
            logits_val, _, _ = forward_pass(X_val)
            if task_type in ['binclass', 'multiclass']:
                val_loss = F.cross_entropy(logits_val, y_val.long())
            else:
                val_loss = F.mse_loss(logits_val.squeeze(-1), y_val.float())
        val_loss_val = float(val_loss.item())
        improved = val_loss_val < best_val_loss - loss_threshold
        print(f"[TABM][MATERIALIZE-GNN-DGM] Epoch {epoch+1}/{gnn_epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss_val:.4f}{' ↓ (improved)' if improved else ''}")

        if improved:
            best_val_loss = val_loss_val
            patience_counter = 0
            best_states = {
                'input_proj': input_proj.state_dict(),
                'out_proj': out_proj.state_dict(),
                'attn_in': attn_in.state_dict(),
                'attn_out': attn_out.state_dict(),
                'gnn': gnn.state_dict(),
                'gcn_to_attn': gcn_to_attn.state_dict(),
                'pred_head': pred_head.state_dict(),
                'column_embed': column_embed.detach().clone(),
                'pool_query': pool_query.detach().clone(),
                'dgm_module': (dgm_module.state_dict() if dgm_module is not None else None),
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[TABM][MATERIALIZE-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})")
                gnn_early_stop_epochs = epoch + 1
                break

    if best_states is not None:
        input_proj.load_state_dict(best_states['input_proj'])
        out_proj.load_state_dict(best_states['out_proj'])
        attn_in.load_state_dict(best_states['attn_in'])
        attn_out.load_state_dict(best_states['attn_out'])
        gnn.load_state_dict(best_states['gnn'])
        gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
        pred_head.load_state_dict(best_states['pred_head'])
        if best_states.get('dgm_module') is not None and dgm_module is not None:
            dgm_module.load_state_dict(best_states['dgm_module'])
        with torch.no_grad():
            column_embed.copy_(best_states['column_embed'])
            pool_query.copy_(best_states['pool_query'])

    input_proj.eval(); out_proj.eval(); attn_in.eval(); attn_out.eval(); gcn_to_attn.eval(); pred_head.eval(); gnn.eval()
    if dgm_module is not None:
        dgm_module.eval()
    with torch.no_grad():
        _, X_train_out, _ = forward_pass(X_train)
        _, X_val_out, _ = forward_pass(X_val)
        _, X_test_out, _ = forward_pass(X_test)

    X_train_out = X_train_out.detach()
    X_val_out = X_val_out.detach()
    X_test_out = X_test_out.detach()
    return X_train_out, X_val_out, X_test_out, gnn_early_stop_epochs


def materialize_fn(train_df, val_df, test_df, dataset_results, config):
    """
    Materialize 階段
    - 數據預處理（QuantileTransformer）
    - 創建特徵嵌入（PiecewiseLinearEmbeddings）
    - 標籤處理
    """
    print("TabM Materializing dataset...")
    device = config.get('device', resolve_device(config))
    task_type = dataset_results['info']['task_type']
    
    # 提取特徵和標籤
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values.astype(np.float32)
    y_val = val_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values
    
    n_num_features = X_train.shape[1]
    
    # 檢查並移除常數列（只有一個唯一值的列）
    unique_counts = np.array([len(np.unique(X_train[:, i])) for i in range(X_train.shape[1])])
    valid_cols = unique_counts > 1
    
    if not valid_cols.all():
        print(f"Warning: Removing {(~valid_cols).sum()} constant columns")
        X_train = X_train[:, valid_cols]
        X_val = X_val[:, valid_cols]
        X_test = X_test[:, valid_cols]
        n_num_features = X_train.shape[1]
    
    # 數值特徵標準化
    noise = np.random.default_rng(0).normal(0.0, 1e-5, X_train.shape).astype(np.float32)
    
    # 動態調整 n_quantiles：不能超過訓練樣本數
    n_quantiles = max(min(len(X_train) // 30, 1000), 10)
    n_quantiles = min(n_quantiles, len(X_train))
    
    preprocessor = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='normal',
        subsample=10**9
    ).fit(X_train + noise)
    
    X_train = preprocessor.transform(X_train).astype(np.float32)
    X_val = preprocessor.transform(X_val).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)
    
    # 標籤處理（對齊 ExcelFormer：loss 在同一尺度上計算；RMSE 報告使用原尺度）
    if task_type == 'regression':
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() if y_train.std() > 1e-12 else 1.0)
        y_train_normalized = (y_train - y_mean) / y_std
        y_val_normalized = (y_val - y_mean) / y_std
        y_test_normalized = (y_test - y_mean) / y_std
        n_classes = None
        label_stats = {'mean': y_mean, 'std': y_std}
    else:
        y_train_normalized = y_train
        y_val_normalized = y_val
        y_test_normalized = y_test
        n_classes = int(len(np.unique(y_train)))
        label_stats = None
    
    # 轉換為 tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32 if task_type == 'regression' else torch.long, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val_normalized, dtype=torch.float32 if task_type == 'regression' else torch.long, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32 if task_type == 'regression' else torch.long, device=device)
    
    # 創建特徵嵌入（使用 PiecewiseLinearEmbeddings）
    # 動態調整 n_bins：必須 < len(X_train) 且 > 1
    n_bins = min(48, len(X_train_tensor) - 1)
    n_bins = max(n_bins, 2)  # 至少 2 個 bins
    
    num_embeddings = PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(
            X_train_tensor,
            n_bins=n_bins
        ),
        d_embedding=16,
        activation=False,
        version='B'
    ).to(device)
    
    return {
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_val': X_val_tensor,
        'y_val': y_val_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'n_num_features': n_num_features,
        'n_classes': n_classes,
        'task_type': task_type,
        'label_stats': label_stats,
        'num_embeddings': num_embeddings,
        'device': device
    }


def tabm_core_fn(material_outputs, config, gnn_stage):
    """TabM 核心訓練（對齊 ExcelFormer）：
    - encoding/columnwise：row-level self-attn pooling → DGM 動態建圖 → GCN → self-attn decode 回寫 + 殘差融合
    - decoding：row-level self-attn pooling → DGM 動態建圖 → GCN 作為 decoder（端到端 joint training）
    - early stopping：基於 val_loss，並恢復最佳權重
    """

    device = material_outputs['device']
    task_type = material_outputs['task_type']
    n_classes = material_outputs['n_classes']
    num_embeddings = material_outputs['num_embeddings']
    label_stats = material_outputs.get('label_stats', None)

    X_train = material_outputs['X_train']
    y_train = material_outputs['y_train']
    X_val = material_outputs['X_val']
    y_val = material_outputs['y_val']
    X_test = material_outputs['X_test']
    y_test = material_outputs['y_test']

    # TabM 超參數
    k = int(config.get('tabm_k', 32))
    n_blocks = int(config.get('tabm_n_blocks', 2))
    d_block = int(config.get('tabm_d_block', 512))
    dropout = float(config.get('tabm_dropout', 0.1))

    # 計算輸入維度（經過 embedding 後）
    with torch.no_grad():
        sample_x = X_train[:1]
        if num_embeddings is not None:
            embedded = num_embeddings(sample_x)
            d_in = int(embedded.shape[-1] * embedded.shape[-2])
        else:
            d_in = int(material_outputs['n_num_features'])

    ensemble_view = EnsembleView(k=k)
    backbone = MLPBackboneBatchEnsemble(
        d_in=d_in,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        k=k,
        tabm_init=True,
        scaling_init='normal',
        start_scaling_init_chunks=None,
    ).to(device)

    n_blocks_total = len(backbone.blocks)
    split_idx = max(1, n_blocks_total // 2)
    encoding_blocks = nn.ModuleList(backbone.blocks[:split_idx])
    columnwise_blocks = nn.ModuleList(backbone.blocks[split_idx:])

    d_out = 1 if task_type == 'regression' else int(n_classes)
    output_layer = None if gnn_stage == 'decoding' else LinearEnsemble(d_block, d_out, k=k).to(device)

    # ---------------------------
    # Dynamic-Graph + Attention row GNN blocks
    # ---------------------------
    use_row_gnn = gnn_stage in ['encoding', 'columnwise', 'decoding']
    gnn = None
    gnn_decoder = None
    dgm_module = None
    token_embed = None
    pool_query = None
    self_attn = None
    attn_norm = None
    self_attn_out = None
    attn_out_norm = None
    gcn_to_attn = None
    ffn_pre = None
    ffn_post = None
    fusion_alpha_param = None

    if use_row_gnn:
        attn_heads = int(config.get('gnn_num_heads', 4))
        if attn_heads <= 0:
            attn_heads = 1

        self_attn = torch.nn.MultiheadAttention(embed_dim=d_block, num_heads=attn_heads, batch_first=True).to(device)
        attn_norm = torch.nn.LayerNorm(d_block).to(device)
        token_embed = torch.nn.Parameter(torch.randn(k, d_block, device=device))
        pool_query = torch.nn.Parameter(torch.randn(d_block, device=device))

        if not DGM_AVAILABLE:
            raise ImportError("DGM is required for the Dynamic-Graph + Attention CORE gnn_stage, but DGM is not available.")

        dgm_k = int(config.get('dgm_k', config.get('gnn_knn', 5)))
        dgm_distance = config.get('dgm_distance', 'euclidean')

        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x

        dgm_module = DGM_d(DGMEmbedWrapper(), k=int(dgm_k), distance=dgm_distance).to(device)

        gnn_hidden = int(config.get('gnn_hidden', 64))
        gnn_layers = int(config.get('gnn_layers', 2))
        gnn_dropout = float(config.get('gnn_dropout', 0.1))
        if gnn_stage == 'decoding':
            gnn_decoder = SimpleGCN(d_block, gnn_hidden, d_out, num_layers=max(2, gnn_layers)).to(device)
        else:
            gnn = SimpleGCN(d_block, gnn_hidden, d_block, num_layers=max(2, gnn_layers)).to(device)
            self_attn_out = torch.nn.MultiheadAttention(embed_dim=d_block, num_heads=attn_heads, batch_first=True).to(device)
            attn_out_norm = torch.nn.LayerNorm(d_block).to(device)
            gcn_to_attn = torch.nn.Linear(d_block, d_block).to(device)
            ffn_pre = torch.nn.Sequential(
                torch.nn.Linear(d_block, d_block * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(gnn_dropout),
                torch.nn.Linear(d_block * 2, d_block),
            ).to(device)
            ffn_post = torch.nn.Sequential(
                torch.nn.Linear(d_block, d_block * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(gnn_dropout),
                torch.nn.Linear(d_block * 2, d_block),
            ).to(device)
            fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))

    def _row_graph_step(x_tokens: torch.Tensor, *, decoder: bool) -> torch.Tensor:
        """x_tokens: [B, k, d_block]"""
        B = int(x_tokens.size(0))
        tokens = x_tokens + token_embed.unsqueeze(0)
        tokens_norm = attn_norm(tokens)
        attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
        tokens_attn = tokens + attn_out1
        # Align with excelformer.py: decoding path does not apply FFN before pooling
        if not decoder:
            tokens_attn = tokens_attn + ffn_pre(attn_norm(tokens_attn))

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(d_block)
        pool_w = torch.softmax(pool_logits, dim=1)
        x_pooled = (pool_w.unsqueeze(-1) * tokens_attn).sum(dim=1)  # [B, d]
        x_std = _standardize(x_pooled, dim=0)

        x_batched = x_std.unsqueeze(0)
        if hasattr(dgm_module, 'k'):
            dgm_module.k = int(min(int(dgm_module.k), max(1, B - 1)))
        x_dgm, edge_index_dgm, _ = dgm_module(x_batched, A=None)
        x_dgm = x_dgm.squeeze(0)
        edge_index = _symmetrize_and_self_loop(edge_index_dgm.to(device), B)

        if decoder:
            return gnn_decoder(x_dgm, edge_index)  # [B, d_out]

        gcn_out = gnn(x_dgm, edge_index)  # [B, d]
        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)  # [B,1,d]
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_ctx_norm = attn_out_norm(tokens_with_ctx)
        attn_out2, _ = self_attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
        tokens_mid = tokens_with_ctx + attn_out2
        tokens_out = tokens_mid + ffn_post(attn_out_norm(tokens_mid))
        fusion_alpha = torch.sigmoid(fusion_alpha_param)
        return x_tokens + fusion_alpha * tokens_out

    def forward(x_num: torch.Tensor, include_gnn: bool = True):
        # Feature embedding
        if num_embeddings is not None:
            x = num_embeddings(x_num)
            x = x.flatten(1, -1)
        else:
            x = x_num

        # EnsembleView -> [B,k,d_in]
        x = ensemble_view(x)

        # Encoding blocks
        for block in encoding_blocks:
            x = block(x)

        if include_gnn and gnn_stage == 'encoding' and use_row_gnn:
            x = _row_graph_step(x, decoder=False)

        # Columnwise blocks
        for block in columnwise_blocks:
            x = block(x)

        if include_gnn and gnn_stage == 'columnwise' and use_row_gnn:
            x = _row_graph_step(x, decoder=False)

        if include_gnn and gnn_stage == 'decoding' and use_row_gnn:
            return _row_graph_step(x, decoder=True)

        return x

    # ---------------------------
    # Optimizer params
    # ---------------------------
    params = []
    if num_embeddings is not None:
        params += list(num_embeddings.parameters())
    params += list(encoding_blocks.parameters())
    params += list(columnwise_blocks.parameters())
    if output_layer is not None:
        params += list(output_layer.parameters())
    if use_row_gnn:
        params += list(self_attn.parameters()) + list(attn_norm.parameters()) + [token_embed, pool_query]
        if dgm_module is not None:
            params += list(dgm_module.parameters())
        if gnn_stage == 'decoding':
            params += list(gnn_decoder.parameters())
        else:
            params += list(gnn.parameters()) + list(self_attn_out.parameters()) + list(attn_out_norm.parameters()) + list(gcn_to_attn.parameters())
            params += list(ffn_pre.parameters()) + list(ffn_post.parameters()) + [fusion_alpha_param]

    lr = float(config.get('lr', 0.002))
    weight_decay = float(config.get('weight_decay', 3e-4))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _loss_train(pred, y_true):
        if task_type == 'regression':
            if pred.dim() == 3:
                pred_flat = pred.flatten(0, 1).squeeze(-1)
                y_rep = y_true.repeat_interleave(k)
                return F.mse_loss(pred_flat, y_rep)
            return F.mse_loss(pred.squeeze(-1), y_true)
        else:
            if pred.dim() == 3:
                pred_flat = pred.flatten(0, 1)
                y_rep = y_true.repeat_interleave(k)
                return F.cross_entropy(pred_flat, y_rep)
            return F.cross_entropy(pred, y_true)

    def _eval_metrics():
        # returns (val_loss, val_metric, test_metric)
        with torch.no_grad():
            x_val = forward(X_val, include_gnn=True)
            x_test = forward(X_test, include_gnn=True)

            if gnn_stage == 'decoding':
                val_logits = x_val
                test_logits = x_test
            else:
                val_logits = output_layer(x_val)
                test_logits = output_layer(x_test)

            val_loss = _loss_train(val_logits, y_val)

            if task_type == 'regression':
                # logits -> predict mean over ensemble if needed
                if val_logits.dim() == 3:
                    val_pred = val_logits.mean(dim=1).squeeze(-1)
                    test_pred = test_logits.mean(dim=1).squeeze(-1)
                else:
                    val_pred = val_logits.squeeze(-1)
                    test_pred = test_logits.squeeze(-1)
                # denorm
                if label_stats is None:
                    val_rmse = float(math.sqrt(mean_squared_error(val_pred.detach().cpu().numpy(), y_val.detach().cpu().numpy())))
                    test_rmse = float(math.sqrt(mean_squared_error(test_pred.detach().cpu().numpy(), y_test.detach().cpu().numpy())))
                else:
                    std = float(label_stats['std'])
                    mean = float(label_stats['mean'])
                    val_pred_denorm = val_pred * std + mean
                    test_pred_denorm = test_pred * std + mean
                    y_val_denorm = y_val * std + mean
                    y_test_denorm = y_test * std + mean
                    val_rmse = float(torch.sqrt(F.mse_loss(val_pred_denorm, y_val_denorm)).item())
                    test_rmse = float(torch.sqrt(F.mse_loss(test_pred_denorm, y_test_denorm)).item())
                return float(val_loss.item()), val_rmse, test_rmse

            # classification
            if val_logits.dim() == 3:
                val_prob = torch.softmax(val_logits, dim=-1).mean(dim=1)
                test_prob = torch.softmax(test_logits, dim=-1).mean(dim=1)
            else:
                val_prob = torch.softmax(val_logits, dim=-1)
                test_prob = torch.softmax(test_logits, dim=-1)

            if task_type == 'binclass' and val_prob.size(-1) == 2:
                y_val_np = y_val.detach().cpu().numpy()
                y_test_np = y_test.detach().cpu().numpy()
                val_score = val_prob[:, 1].detach().cpu().numpy()
                test_score = test_prob[:, 1].detach().cpu().numpy()
                try:
                    val_auc = float(roc_auc_score(y_val_np, val_score))
                    test_auc = float(roc_auc_score(y_test_np, test_score))
                    return float(val_loss.item()), val_auc, test_auc
                except Exception:
                    # fall back to accuracy if AUC is undefined
                    pass

            val_pred = val_prob.argmax(dim=-1)
            test_pred = test_prob.argmax(dim=-1)
            val_acc = float((val_pred == y_val).float().mean().item())
            test_acc = float((test_pred == y_test).float().mean().item())
            return float(val_loss.item()), val_acc, test_acc

    # ---------------------------
    # Training loop (val-loss early stop + best restore)
    # ---------------------------
    batch_size = int(config.get('batch_size', 256))
    epochs = int(config.get('epochs', 200))
    patience = int(config.get('patience', 10))
    loss_threshold = float(config.get('loss_threshold', 1e-6))

    best_val_loss = float('inf')
    best_val_metric = float('-inf') if task_type != 'regression' else float('inf')
    best_test_metric = float('-inf') if task_type != 'regression' else float('inf')
    best_states = None
    patience_counter = 0
    early_stop_epochs = 0

    for epoch in range(1, epochs + 1):
        # train mode
        if num_embeddings is not None:
            num_embeddings.train()
        for b in encoding_blocks:
            b.train()
        for b in columnwise_blocks:
            b.train()
        if output_layer is not None:
            output_layer.train()
        if use_row_gnn:
            self_attn.train()
            attn_norm.train()
            if dgm_module is not None:
                dgm_module.train()
            if gnn_stage == 'decoding':
                gnn_decoder.train()
            else:
                gnn.train(); self_attn_out.train(); attn_out_norm.train(); gcn_to_attn.train(); ffn_pre.train(); ffn_post.train()

        indices = torch.randperm(len(X_train), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            feats_or_logits = forward(x_batch, include_gnn=True)
            if gnn_stage == 'decoding':
                loss = _loss_train(feats_or_logits, y_batch)
            else:
                logits = output_layer(feats_or_logits)
                loss = _loss_train(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        # eval mode
        if num_embeddings is not None:
            num_embeddings.eval()
        for b in encoding_blocks:
            b.eval()
        for b in columnwise_blocks:
            b.eval()
        if output_layer is not None:
            output_layer.eval()
        if use_row_gnn:
            self_attn.eval()
            attn_norm.eval()
            if dgm_module is not None:
                dgm_module.eval()
            if gnn_stage == 'decoding':
                gnn_decoder.eval()
            else:
                gnn.eval(); self_attn_out.eval(); attn_out_norm.eval(); gcn_to_attn.eval(); ffn_pre.eval(); ffn_post.eval()

        val_loss, val_metric, test_metric = _eval_metrics()
        improved = val_loss < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            best_val_metric = val_metric
            best_test_metric = test_metric

            best_states = {
                'num_embeddings': (num_embeddings.state_dict() if num_embeddings is not None else None),
                'encoding_blocks': encoding_blocks.state_dict(),
                'columnwise_blocks': columnwise_blocks.state_dict(),
                'output_layer': (output_layer.state_dict() if output_layer is not None else None),
                'use_row_gnn': use_row_gnn,
                'self_attn': (self_attn.state_dict() if use_row_gnn else None),
                'attn_norm': (attn_norm.state_dict() if use_row_gnn and attn_norm is not None else None),
                'dgm_module': (dgm_module.state_dict() if dgm_module is not None else None),
                'token_embed': (token_embed.detach().clone() if token_embed is not None else None),
                'pool_query': (pool_query.detach().clone() if pool_query is not None else None),
                'gnn': (gnn.state_dict() if gnn is not None else None),
                'self_attn_out': (self_attn_out.state_dict() if self_attn_out is not None else None),
                'attn_out_norm': (attn_out_norm.state_dict() if attn_out_norm is not None else None),
                'gcn_to_attn': (gcn_to_attn.state_dict() if gcn_to_attn is not None else None),
                'ffn_pre': (ffn_pre.state_dict() if ffn_pre is not None else None),
                'ffn_post': (ffn_post.state_dict() if ffn_post is not None else None),
                'fusion_alpha_param': (fusion_alpha_param.detach().clone() if fusion_alpha_param is not None else None),
                'gnn_decoder': (gnn_decoder.state_dict() if gnn_decoder is not None else None),
            }
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            if task_type == 'regression':
                metric_name = 'RMSE'
            elif task_type == 'binclass':
                metric_name = 'AUC'
            else:
                metric_name = 'Acc'
            print(f"[TABM][CORE] Epoch {epoch}/{epochs}, Train Loss: {epoch_loss/max(1,n_batches):.4f}, Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.4f}{' ↓ (improved)' if improved else ''}")

        if patience_counter >= patience:
            early_stop_epochs = epoch
            print(f"[TABM][CORE] Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    # restore best
    if best_states is not None:
        if num_embeddings is not None and best_states['num_embeddings'] is not None:
            num_embeddings.load_state_dict(best_states['num_embeddings'])
        encoding_blocks.load_state_dict(best_states['encoding_blocks'])
        columnwise_blocks.load_state_dict(best_states['columnwise_blocks'])
        if output_layer is not None and best_states['output_layer'] is not None:
            output_layer.load_state_dict(best_states['output_layer'])
        if use_row_gnn:
            self_attn.load_state_dict(best_states['self_attn'])
            if best_states.get('attn_norm') is not None and attn_norm is not None:
                attn_norm.load_state_dict(best_states['attn_norm'])
            if best_states.get('token_embed') is not None:
                with torch.no_grad():
                    token_embed.copy_(best_states['token_embed'])
            if best_states.get('pool_query') is not None:
                with torch.no_grad():
                    pool_query.copy_(best_states['pool_query'])
            if dgm_module is not None and best_states.get('dgm_module') is not None:
                dgm_module.load_state_dict(best_states['dgm_module'])
            if gnn_stage == 'decoding' and gnn_decoder is not None and best_states.get('gnn_decoder') is not None:
                gnn_decoder.load_state_dict(best_states['gnn_decoder'])
            if gnn_stage != 'decoding' and gnn is not None:
                if best_states.get('gnn') is not None:
                    gnn.load_state_dict(best_states['gnn'])
                if best_states.get('self_attn_out') is not None:
                    self_attn_out.load_state_dict(best_states['self_attn_out'])
                if best_states.get('attn_out_norm') is not None and attn_out_norm is not None:
                    attn_out_norm.load_state_dict(best_states['attn_out_norm'])
                if best_states.get('gcn_to_attn') is not None:
                    gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
                if best_states.get('ffn_pre') is not None and ffn_pre is not None:
                    ffn_pre.load_state_dict(best_states['ffn_pre'])
                if best_states.get('ffn_post') is not None and ffn_post is not None:
                    ffn_post.load_state_dict(best_states['ffn_post'])
                if best_states.get('fusion_alpha_param') is not None:
                    with torch.no_grad():
                        fusion_alpha_param.copy_(best_states['fusion_alpha_param'])

    if early_stop_epochs == 0:
        early_stop_epochs = epochs

    return {
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'early_stop_epochs': early_stop_epochs,
        # decoding stage is now end-to-end, so there is no standalone GNN pre-stage early stop
        'gnn_early_stop_epochs': 0,
    }


def tabm_decoding_with_gnn(X_train, y_train, X_val, y_val, X_test, y_test,
                            num_embeddings, ensemble_view, encoding_blocks, columnwise_blocks,
                            forward, task_type, label_stats, k, d_block, config):
    """
    [Legacy] 舊版 decoding 流程：先訓練 encoder+columnwise，再用（靜態 kNN 圖上的）GNN 作為 decoder。

    注意：目前 `tabm_core_fn` 在 `gnn_stage == 'decoding'` 已改為端到端 joint training（Dynamic-Graph + Attention / DGM + Self-Attn），
    因此此函式不再是主路徑；保留僅供回溯/對照實驗使用。
    """
    print("TabM Decoding with GNN as decoder...")
    device = config['device']
    
    # 第一步：訓練 encoder + columnwise
    print("Step 1: Training encoder + columnwise...")
    all_params = []
    if num_embeddings is not None:
        all_params += list(num_embeddings.parameters())
    all_params += list(encoding_blocks.parameters())
    all_params += list(columnwise_blocks.parameters())
    
    # 臨時輸出層用於訓練 encoder
    n_classes = len(torch.unique(y_train)) if task_type != 'regression' else None
    d_out = 1 if task_type == 'regression' else n_classes
    temp_output = LinearEnsemble(d_block, d_out, k=k).to(device)
    all_params += list(temp_output.parameters())
    
    optimizer = torch.optim.AdamW(all_params, lr=config.get('lr', 0.002), weight_decay=config.get('weight_decay', 3e-4))
    
    def loss_fn(y_pred, y_true):
        y_pred_flat = y_pred.flatten(0, 1)
        y_true_repeated = y_true.repeat_interleave(k)
        if task_type == 'regression':
            return F.mse_loss(y_pred_flat.squeeze(-1), y_true_repeated)
        else:
            return F.cross_entropy(y_pred_flat, y_true_repeated)
    
    batch_size = config.get('batch_size', 256)
    encoder_epochs = config.get('epochs', 100)
    
    for epoch in range(1, encoder_epochs + 1):
        if num_embeddings is not None:
            num_embeddings.train()
        for block in encoding_blocks:
            block.train()
        for block in columnwise_blocks:
            block.train()
        temp_output.train()
        
        indices = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            x_encoded = forward(x_batch, include_gnn=False)
            y_pred = temp_output(x_encoded)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    print("Step 2: Extracting embeddings and training GNN decoder...")
    # 第二步：將所有數據通過 encoder+columnwise 獲取 embedding
    with torch.no_grad():
        if num_embeddings is not None:
            num_embeddings.eval()
        for block in encoding_blocks:
            block.eval()
        for block in columnwise_blocks:
            block.eval()
        
        X_train_emb = forward(X_train, include_gnn=False).mean(dim=1)  # 平均 k 個 ensemble
        X_val_emb = forward(X_val, include_gnn=False).mean(dim=1)
        X_test_emb = forward(X_test, include_gnn=False).mean(dim=1)
    
    # 合併所有 embedding
    X_all_emb = torch.cat([X_train_emb, X_val_emb, X_test_emb], dim=0)
    y_all = torch.cat([y_train, y_val, y_test], dim=0)
    
    # 建圖
    k_gnn = 5
    edge_index = knn_graph(X_all_emb, k_gnn).to(device)
    
    # 創建 GNN decoder
    gnn_hidden = config.get('gnn_hidden', 64)
    d_out = 1 if task_type == 'regression' else n_classes
    gnn_decoder = SimpleGCN(d_block, gnn_hidden, d_out).to(device)
    
    # 訓練 GNN
    optimizer_gnn = torch.optim.Adam(gnn_decoder.parameters(), lr=0.01)
    gnn_epochs = config.get('gnn_epochs', 200)
    patience = config.get('gnn_patience', 10)
    
    n_train = len(X_train)
    n_val = len(X_val)
    
    best_val_metric = float('-inf') if task_type != 'regression' else float('inf')
    best_test_metric = 0
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    
    for epoch in range(gnn_epochs):
        gnn_decoder.train()
        optimizer_gnn.zero_grad()
        
        out = gnn_decoder(X_all_emb, edge_index)
        
        if task_type == 'regression':
            loss = F.mse_loss(out[:n_train].squeeze(-1), y_train)
        else:
            loss = F.cross_entropy(out[:n_train], y_train)
        
        loss.backward()
        optimizer_gnn.step()
        
        # 驗證
        gnn_decoder.eval()
        with torch.no_grad():
            out_eval = gnn_decoder(X_all_emb, edge_index)
            val_out = out_eval[n_train:n_train+n_val]
            
            if task_type == 'regression':
                val_pred_denorm = val_out.squeeze(-1) * label_stats['std'] + label_stats['mean']
                val_metric = torch.sqrt(F.mse_loss(val_pred_denorm, y_val)).item()
            else:
                val_pred_class = val_out.argmax(dim=-1)
                val_metric = (val_pred_class == y_val).float().mean().item()
        
        improved = (val_metric > best_val_metric) if task_type != 'regression' else (val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            early_stop_counter = 0
            
            # 測試
            test_out = out_eval[n_train+n_val:]
            if task_type == 'regression':
                test_pred_denorm = test_out.squeeze(-1) * label_stats['std'] + label_stats['mean']
                best_test_metric = torch.sqrt(F.mse_loss(test_pred_denorm, y_test)).item()
            else:
                test_pred_class = test_out.argmax(dim=-1)
                best_test_metric = (test_pred_class == y_test).float().mean().item()
        else:
            early_stop_counter += 1
        
        if (epoch+1) % 10 == 0:
            metric_name = 'RMSE' if task_type == 'regression' else 'Acc'
            print(f'GNN Epoch {epoch+1}/{gnn_epochs}, Val {metric_name}: {val_metric:.4f}')
        
        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"GNN Early stopping at epoch {epoch+1}")
            break
    
    return {
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'early_stop_epochs': encoder_epochs,
        'gnn_early_stop_epochs': gnn_early_stop_epochs,
    }


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    主函數：按五階段執行 TabM
    """
    print("=" * 60)
    print(f"TabM - Five Stage Execution (gnn_stage: {gnn_stage})")
    print("=" * 60)
    
    task_type = dataset_results['info']['task_type']
    gnn_early_stop_epochs = 0
    
    try:
        # 階段 0: Start (dummy)
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        
        # GNN after start
        if gnn_stage == 'start':
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(
                train_df, val_df, test_df, config, task_type
            )
        
        # 階段 1: Materialize
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        
        # GNN after materialize
        if gnn_stage == 'materialize':
            X_train_gnn, X_val_gnn, X_test_gnn, gnn_early_stop_epochs = gnn_after_materialize_fn(
                material_outputs['X_train'], material_outputs['y_train'],
                material_outputs['X_val'], material_outputs['y_val'],
                material_outputs['X_test'], material_outputs['y_test'],
                config, task_type
            )
            material_outputs['X_train'] = X_train_gnn
            material_outputs['X_val'] = X_val_gnn
            material_outputs['X_test'] = X_test_gnn

            # 重要：TabM 的數值 embedding (PiecewiseLinearEmbeddings) 依賴訓練特徵分佈的 bins。
            # 如果在 materialize 後重寫了 X，必須用新的 X_train 重新計算 bins，否則會嚴重 mismatch。
            if material_outputs.get('num_embeddings', None) is not None:
                try:
                    device = material_outputs['device']
                    X_train_for_bins = material_outputs['X_train']
                    n_bins = min(48, int(X_train_for_bins.shape[0]) - 1)
                    n_bins = max(n_bins, 2)
                    material_outputs['num_embeddings'] = PiecewiseLinearEmbeddings(
                        rtdl_num_embeddings.compute_bins(
                            X_train_for_bins,
                            n_bins=n_bins,
                        ),
                        d_embedding=16,
                        activation=False,
                        version='B',
                    ).to(device)
                    print(f"[TABM][MATERIALIZE-GNN-DGM] Recomputed num_embeddings bins with n_bins={n_bins}")
                except Exception as _e:
                    print(f"[TABM][MATERIALIZE-GNN-DGM][WARNING] Failed to recompute num_embeddings bins: {_e}")
        
        # 階段 2-4: 核心訓練（encoding, columnwise, decoding 都在這裡處理）
        results = tabm_core_fn(material_outputs, config, gnn_stage)
        
        # 如果 results 中已經有 gnn_early_stop_epochs（decoding 階段），使用它
        # 否則使用之前階段設置的值（start/materialize）或默認值 0
        if 'gnn_early_stop_epochs' not in results:
            results['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        
    except Exception as e:
        import traceback
        print(f"Error in TabM: {str(e)}")
        traceback.print_exc()
        
        is_classification = task_type != 'regression'
        results = {
            'best_val_metric': float('-inf') if is_classification else float('inf'),
            'best_test_metric': float('-inf') if is_classification else float('inf'),
            'error': str(e),
            'early_stop_epochs': 0,
            'gnn_early_stop_epochs': 0,
        }
    
    return results


# 測試命令
#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models tabm --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models tabm --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models tabm --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models tabm --gnn_stages all --epochs 2
#  python main.py --dataset helena --models tabm --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models tabm --gnn_stages all --epochs 2
