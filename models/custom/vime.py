
from __future__ import annotations

import copy
import math
import sys
import time
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from tqdm import tqdm

from torch_geometric.nn import GCNConv

sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
from DGMlib.layers import DGM_d


def resolve_device(config: Dict[str, Any]) -> torch.device:
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
        print("[DEVICE] resolve_device: Using cuda (default)")
        return torch.device('cuda')
    print("[DEVICE] resolve_device: Using cpu (CUDA not available)")
    return torch.device('cpu')


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
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


def _task_type_from_info(dataset_results: Dict[str, Any]) -> str:
    info = dataset_results.get('info', {}) if isinstance(dataset_results, dict) else {}
    return info.get('task_type', info.get('task', 'binclass'))


def _split_xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    feat_cols = [c for c in df.columns if c != 'target']
    X = df[feat_cols].values.astype(np.float32)
    y = df['target'].values
    return X, y


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _dbg_enabled(config: Dict[str, Any]) -> bool:
    try:
        return bool(config.get('debug_metrics', True))
    except Exception:
        return False


def _dbg_print(config: Dict[str, Any], msg: str):
    if _dbg_enabled(config):
        print(msg)


class VIMEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mask_head = nn.Linear(hidden, in_dim)
        self.recon_head = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mask_logits = self.mask_head(h)
        recon = self.recon_head(h)
        return h, mask_logits, recon


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(GCNConv(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: [N, D]
    # ALWAYS use CPU-based sklearn NearestNeighbors to avoid CUDA OOM from O(N^2) torch.cdist
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError(
            "sklearn is required for knn_graph (CPU-based KNN to avoid CUDA OOM). "
            "Install it with: pip install scikit-learn"
        )

    # Move data to CPU numpy for sklearn
    x_cpu = x.detach().cpu().numpy()
    N = x_cpu.shape[0]
    # sklearn's n_neighbors must be <= N
    nn_k = min(k + 1, N)

    # Use sklearn NearestNeighbors (CPU, memory-friendly)
    nbrs = NearestNeighbors(n_neighbors=nn_k, algorithm='auto', n_jobs=-1).fit(x_cpu)
    distances, indices = nbrs.kneighbors(x_cpu)
    # drop self neighbor (first column)
    if indices.shape[1] > 1:
        knn = indices[:, 1:nn_k]
    else:
        knn = indices[:, :]

    src = np.repeat(np.arange(N), knn.shape[1])
    dst = knn.reshape(-1)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return edge_index.to(x.device)


def _run_attention_dgm_pipeline(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               config: Dict[str, Any],
                               task_type: str,
                               stage_tag: str = 'START'):
    """ExcelFormer-style: Self-Attn pooling -> DGM dynamic graph -> GCN -> Self-Attn decode.

    - Train uses train split only; early stopping uses val_loss; inference is inductive per split.
    - Output is reconstructed feature matrix with original columns.
    """

    device = resolve_device(config)
    tag = f"[VIME][{stage_tag}-GNN-DGM]"
    print(f"{tag} Running attention + DGM pipeline (inductive)")

    feature_cols = [c for c in train_df.columns if c != 'target']
    num_cols = len(feature_cols)
    if num_cols <= 0:
        return train_df, val_df, test_df, 0

    dgm_k = int(config.get('dgm_k', 10))
    dgm_distance = config.get('dgm_distance', 'euclidean')

    # Strict alignment with excelformer.py: stage epochs use config['epochs'] only.
    gnn_epochs = int(config.get('epochs', 200))
    patience = int(config.get('gnn_patience', 10))
    loss_threshold = float(config.get('gnn_loss_threshold', 1e-4))
    attn_dim = int(config.get('gnn_attn_dim', config.get('gnn_hidden', 64)))
    gnn_hidden = int(config.get('gnn_hidden', 64))
    gnn_out_dim = int(config.get('gnn_out_dim', attn_dim))
    attn_heads = int(config.get('gnn_num_heads', config.get('gnn_attn_heads', 4)))
    lr = float(config.get('gnn_lr', 1e-3))

    x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)

    y_train_np = train_df['target'].values
    y_val_np = val_df['target'].values
    y_test_np = test_df['target'].values

    if task_type in ['binclass', 'multiclass']:
        y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
        num_classes = len(pd.unique(y_all_np))
        if task_type == 'binclass' and num_classes != 2:
            num_classes = 2
        out_dim = int(num_classes)
    else:
        out_dim = 1

    n_train = int(x_train.shape[0])
    if n_train <= 1:
        raise ValueError(f"{tag} Need at least 2 train samples for DGM (got n_train={n_train})")
    dgm_k_train = int(min(dgm_k, max(1, n_train - 1)))

    attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)

    input_proj = torch.nn.Linear(1, attn_dim).to(device)
    gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim, num_layers=2).to(device)
    gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
    pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)
    out_proj = torch.nn.Linear(attn_dim, 1).to(device)
    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    class DGMEmbedWrapper(torch.nn.Module):
        def forward(self, x, A=None):
            return x

    dgm_embed_f = DGMEmbedWrapper()
    dgm_module = DGM_d(dgm_embed_f, k=dgm_k_train, distance=dgm_distance).to(device)

    params = (
        list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters())
        + list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters())
        + list(out_proj.parameters()) + [column_embed, pool_query] + list(dgm_module.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    def forward_pass(x_tensor: torch.Tensor):
        Ns = int(x_tensor.shape[0])
        x_in = input_proj(x_tensor.unsqueeze(-1))  # [Ns, num_cols, attn_dim]
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_weights = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)
        row_emb_std = _standardize(row_emb, dim=0)

        row_emb_batched = row_emb_std.unsqueeze(0)
        if hasattr(dgm_module, 'k'):
            dgm_module.k = int(min(int(dgm_module.k), max(1, Ns - 1)))
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)
        edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)

        gcn_out = gnn(row_emb_dgm, edge_index_dgm)
        logits = pred_head(gcn_out)

        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_with_ctx, tokens_with_ctx, tokens_with_ctx)
        recon = out_proj(tokens_out).squeeze(-1)
        return logits, recon, logprobs_dgm

    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_states = None

    for epoch in range(gnn_epochs):
        attn_in.train(); attn_out.train(); input_proj.train(); gnn.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()
        dgm_module.train()

        optimizer.zero_grad()
        logits_train, _, logprobs_dgm = forward_pass(x_train)
        if task_type in ['binclass', 'multiclass']:
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            train_loss = F.cross_entropy(logits_train, y_train)
        else:
            y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
            train_loss = F.mse_loss(logits_train.squeeze(), y_train)
        dgm_reg = -logprobs_dgm.mean() * 0.01
        train_loss = train_loss + dgm_reg
        train_loss.backward()
        optimizer.step()

        attn_in.eval(); attn_out.eval(); input_proj.eval(); gnn.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
        dgm_module.eval()
        with torch.no_grad():
            logits_val, _, _ = forward_pass(x_val)
            if task_type in ['binclass', 'multiclass']:
                y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
                val_loss = F.cross_entropy(logits_val, y_val)
            else:
                y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
                val_loss = F.mse_loss(logits_val.squeeze(), y_val)

        val_loss_val = float(val_loss.item())
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

        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"{tag} Early stopping at epoch {gnn_early_stop_epochs}")
            break

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

    attn_in.eval(); attn_out.eval(); input_proj.eval(); gnn.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval(); dgm_module.eval()
    with torch.no_grad():
        _, recon_train, _ = forward_pass(x_train)
        _, recon_val, _ = forward_pass(x_val)
        _, recon_test, _ = forward_pass(x_test)

    train_df_gnn = pd.DataFrame(recon_train.detach().cpu().numpy(), columns=feature_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(recon_val.detach().cpu().numpy(), columns=feature_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(recon_test.detach().cpu().numpy(), columns=feature_cols, index=test_df.index)
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values

    return train_df_gnn, val_df_gnn, test_df_gnn, int(gnn_early_stop_epochs)


def mask_generator(p_m: float, x: torch.Tensor) -> torch.Tensor:
    return (torch.rand_like(x) < p_m).float()


def pretext_generator(m: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Shuffle each column independently
    x_bar = x[torch.randperm(x.size(0))]
    x_tilde = x * (1 - m) + x_bar * m
    m_new = (x != x_tilde).float()
    return m_new, x_tilde


def start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    # Identity; GNN at start stage is handled by gnn_after_start_fn
    return train_df, val_df, test_df

def gnn_after_start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict[str, Any], task_type: str):
    return _run_attention_dgm_pipeline(train_df, val_df, test_df, config, task_type, stage_tag='START')


def materialize_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, dataset_results, config):
    X_train, y_train = _split_xy_from_df(train_df)
    X_val, y_val = _split_xy_from_df(val_df)
    X_test, y_test = _split_xy_from_df(test_df)

    X_train, mean, std = _standardize_fit(X_train)
    X_val = _standardize_apply(X_val, mean, std)
    X_test = _standardize_apply(X_test, mean, std)
    # Optional: standardize y for regression losses/early stopping (metrics will be reported in original units)
    task_type = _task_type_from_info(dataset_results)
    y_mean = None
    y_std = None
    if task_type == 'regression' and bool(config.get('standardize_y', True)):
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train) + 1e-8)
    if task_type == 'regression':
        _dbg_print(config, f"[debug] materialize y stats -> train(mean,std)={(float(np.mean(y_train)), float(np.std(y_train)))} val={(float(np.mean(y_val)), float(np.std(y_val)))} test={(float(np.mean(y_test)), float(np.std(y_test)))}")
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'mean': mean,
        'std': std,
        'y_mean': y_mean,
        'y_std': y_std,
    }

def gnn_after_materialize_arrays(mat: Dict[str, Any], config: Dict[str, Any], task_type: str) -> Tuple[Dict[str, Any], int]:
    in_dim = int(mat['X_train'].shape[1])
    feature_cols = [f'N_feature_{i + 1}' for i in range(in_dim)]

    train_df = pd.DataFrame(mat['X_train'], columns=feature_cols)
    val_df = pd.DataFrame(mat['X_val'], columns=feature_cols)
    test_df = pd.DataFrame(mat['X_test'], columns=feature_cols)
    train_df['target'] = mat['y_train']
    val_df['target'] = mat['y_val']
    test_df['target'] = mat['y_test']

    train_g, val_g, test_g, gnn_early_stop_epochs = _run_attention_dgm_pipeline(
        train_df, val_df, test_df, config, task_type, stage_tag='MATERIALIZE'
    )

    mat['X_train'] = train_g[feature_cols].values.astype(np.float32)
    mat['X_val'] = val_g[feature_cols].values.astype(np.float32)
    mat['X_test'] = test_g[feature_cols].values.astype(np.float32)
    return mat, int(gnn_early_stop_epochs)


def vime_core_fn(mat: Dict[str, Any], config: Dict[str, Any], task_type: str, gnn_stage: str = 'none'):
    device = resolve_device(config)
    X_train = torch.tensor(mat['X_train'], dtype=torch.float32, device=device)
    X_val = torch.tensor(mat['X_val'], dtype=torch.float32, device=device)
    X_test = torch.tensor(mat['X_test'], dtype=torch.float32, device=device)

    in_dim = X_train.size(1)
    hidden = int(config.get('hidden_dim', 256))
    dropout = float(config.get('dropout', 0.1))
    epochs = int(config.get('epochs', 200))
    batch_size = int(config.get('batch_size', 256))
    lr = float(config.get('lr', 1e-3))
    patience = int(config.get('patience', 10))
    alpha = float(config.get('vime_alpha', 2.0))
    p_m = float(config.get('vime_p_m', 0.3))

    enc = VIMEEncoder(in_dim, hidden, dropout).to(device)
    weight_decay = float(config.get('weight_decay', 1e-5))

    # decoding: strict alignment with excelformer_core_fn(decoding)
    # - GNN replaces downstream decoder/head
    # - Jointly trained with encoder in the same loop
    # - Minibatch Self-Attn -> pooling -> DGM -> GCN(out_dim)
    if gnn_stage == 'decoding':
        tag = '[VIME][DECODING-JOINT]'
        is_class = task_type in ['binclass', 'multiclass']

        y_train = mat['y_train']
        y_val = mat['y_val']
        y_test = mat['y_test']

        if is_class:
            y_all = np.concatenate([y_train, y_val, y_test])
            num_classes = int(len(np.unique(y_all)))
            if task_type == 'binclass' and num_classes != 2:
                num_classes = 2
            out_dim = int(num_classes) if task_type != 'binclass' else 2
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_val_t = torch.tensor(y_val, dtype=torch.long)
            y_test_t = torch.tensor(y_test, dtype=torch.long)
            y_mean = None
            y_std = None
        else:
            out_dim = 1
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32)
            y_mean = mat.get('y_mean', float(np.mean(y_train)))
            y_std = mat.get('y_std', float(np.std(y_train) + 1e-8))

        x_train = X_train.detach().cpu()
        x_val = X_val.detach().cpu()
        x_test = X_test.detach().cpu()

        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train_t),
            batch_size=batch_size,
            shuffle=True,
        )
        val_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_val, y_val_t),
            batch_size=batch_size,
            shuffle=False,
        )
        test_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test_t),
            batch_size=batch_size,
            shuffle=False,
        )

        hidden_dim = int(config.get('gnn_hidden', 64))
        patience = int(config.get('gnn_patience', 10))
        loss_threshold = float(config.get('gnn_loss_threshold', 1e-4))
        gnn_lr = float(config.get('gnn_lr', lr))
        attn_dim = int(config.get('gnn_attn_dim', hidden_dim))
        attn_heads = int(config.get('gnn_num_heads', config.get('gnn_attn_heads', 4)))
        dgm_k = int(config.get('dgm_k', config.get('gnn_knn', 5)))
        dgm_distance = config.get('dgm_distance', 'euclidean')

        # Strict alignment with excelformer_core_fn(decoding): tokens are true columns.
        num_cols = int(in_dim)
        self_attn = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
        attn_norm = torch.nn.LayerNorm(attn_dim).to(device)
        input_proj = torch.nn.Linear(1, attn_dim).to(device)
        column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
        pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

        # VIME core component: keep encoder in the joint loop by injecting a row-context into column tokens.
        row_ctx_proj = torch.nn.Linear(hidden, attn_dim).to(device)

        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x

        dgm_module = DGM_d(DGMEmbedWrapper(), k=dgm_k, distance=dgm_distance).to(device)
        gnn = SimpleGCN(attn_dim, hidden_dim, out_dim, num_layers=2).to(device)

        optimizer = torch.optim.Adam(
            list(enc.parameters())
            + list(self_attn.parameters())
            + list(attn_norm.parameters())
            + list(input_proj.parameters())
            + list(row_ctx_proj.parameters())
            + list(dgm_module.parameters())
            + list(gnn.parameters())
            + [column_embed, pool_query],
            lr=gnn_lr,
            weight_decay=weight_decay,
        )

        def forward_pass(xb: torch.Tensor):
            # xb: [B, in_dim]
            B = int(xb.shape[0])
            if B <= 1:
                raise ValueError(f"{tag} Need batch size >= 2 for DGM (got B={B})")
            xb = xb.to(device)
            z_row = enc.net(xb)  # [B, hidden]
            row_ctx = row_ctx_proj(z_row).unsqueeze(1)  # [B, 1, attn_dim]

            tokens = input_proj(xb.unsqueeze(-1)) + column_embed.unsqueeze(0) + row_ctx
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1

            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(attn_dim)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

            x_std = _standardize(x_pooled, dim=0)
            x_batched = x_std.unsqueeze(0)
            if hasattr(dgm_module, 'k'):
                dgm_module.k = int(min(int(dgm_module.k), max(1, B - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, B)
            logits = gnn(x_dgm, edge_index_dgm)
            return logits, logprobs_dgm

        best_val_loss = float('inf')
        best_val_metric = None
        best_test_metric = None
        best_state = None
        early_stop_counter = 0
        total_epochs = epochs

        # Preserve VIME core in decoding: jointly optimize supervised objective + VIME pretext objective.
        decoding_vime_lambda = float(config.get('decoding_vime_lambda', 0.1))

        for epoch in range(epochs):
            enc.train(); self_attn.train(); attn_norm.train(); input_proj.train(); row_ctx_proj.train(); dgm_module.train(); gnn.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                xb = xb.to(device)
                yb = yb.to(device)
                if xb.size(0) <= 1:
                    continue

                # 1) Supervised head (ExcelFormer-style decoding pipeline)
                logits, logprobs_dgm = forward_pass(xb)
                if is_class:
                    sup_loss = F.cross_entropy(logits, yb.long())
                else:
                    yb_loss = yb.float()
                    yb_loss = (yb_loss - float(y_mean)) / float(y_std)

                    sup_loss = F.mse_loss(logits.squeeze(), yb_loss)

                # 2) VIME core pretext (mask prediction + reconstruction)
                m = mask_generator(p_m, xb)
                m_label, x_tilde = pretext_generator(m, xb)
                _, mask_logits, recon = enc(x_tilde)
                mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_label)
                recon_loss = F.mse_loss(recon, xb)
                vime_loss = mask_loss + alpha * recon_loss

                loss = sup_loss + decoding_vime_lambda * vime_loss
                if logprobs_dgm is not None:
                    loss = loss + (-logprobs_dgm.mean() * 0.01)
                loss.backward()
                optimizer.step()

            enc.eval(); self_attn.eval(); attn_norm.eval(); input_proj.eval(); row_ctx_proj.eval(); dgm_module.eval(); gnn.eval()
            with torch.no_grad():
                val_losses = []
                val_logits_all = []
                val_y_all = []
                for xb, yb in val_dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    if xb.size(0) <= 1:
                        continue
                    logits, _ = forward_pass(xb)
                    if is_class:
                        vloss = F.cross_entropy(logits, yb.long())
                    else:
                        yb_loss = yb.float()
                        yb_loss = (yb_loss - float(y_mean)) / float(y_std)
                        vloss = F.mse_loss(logits.squeeze(), yb_loss)
                    val_losses.append(float(vloss.item()) * xb.size(0))
                    val_logits_all.append(logits.detach().cpu())
                    val_y_all.append(yb.detach().cpu())

                if not val_losses:
                    continue

                val_loss_val = float(sum(val_losses) / max(1, int(len(y_val))))
                val_logits_cat = torch.cat(val_logits_all, dim=0)
                val_y_cat = torch.cat(val_y_all, dim=0)
                if task_type == 'binclass':
                    val_prob = torch.softmax(val_logits_cat, dim=-1)[:, 1]
                    val_metric = float(roc_auc_score(val_y_cat.numpy(), val_prob.numpy()))
                elif task_type == 'multiclass':
                    val_pred = val_logits_cat.argmax(dim=-1)
                    val_metric = float(accuracy_score(val_y_cat.numpy(), val_pred.numpy()))
                else:
                    val_pred = val_logits_cat.squeeze().numpy()
                    val_pred = val_pred * float(y_std) + float(y_mean)
                    val_metric = float(math.sqrt(mean_squared_error(val_y_cat.numpy(), val_pred)))

                improved = float(val_loss_val) < best_val_loss - loss_threshold
                if improved:
                    best_val_loss = float(val_loss_val)
                    best_val_metric = float(val_metric)
                    best_state = {
                        'enc': copy.deepcopy(enc.state_dict()),
                        'self_attn': copy.deepcopy(self_attn.state_dict()),
                        'attn_norm': copy.deepcopy(attn_norm.state_dict()),
                        'input_proj': copy.deepcopy(input_proj.state_dict()),
                        'row_ctx_proj': copy.deepcopy(row_ctx_proj.state_dict()),
                        'dgm_module': copy.deepcopy(dgm_module.state_dict()),
                        'gnn': copy.deepcopy(gnn.state_dict()),
                        'column_embed': column_embed.detach().clone(),
                        'pool_query': pool_query.detach().clone(),
                    }
                    early_stop_counter = 0

                    # Evaluate test metric under current best
                    test_logits_all = []
                    test_y_all = []
                    for xb, yb in test_dl:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        if xb.size(0) <= 1:
                            continue
                        logits, _ = forward_pass(xb)
                        test_logits_all.append(logits.detach().cpu())
                        test_y_all.append(yb.detach().cpu())
                    if test_logits_all:
                        test_logits_cat = torch.cat(test_logits_all, dim=0)
                        test_y_cat = torch.cat(test_y_all, dim=0)
                        if task_type == 'binclass':
                            te_prob = torch.softmax(test_logits_cat, dim=-1)[:, 1]
                            best_test_metric = float(roc_auc_score(test_y_cat.numpy(), te_prob.numpy()))
                        elif task_type == 'multiclass':
                            te_pred = test_logits_cat.argmax(dim=-1)
                            best_test_metric = float(accuracy_score(test_y_cat.numpy(), te_pred.numpy()))
                        else:
                            te_pred = test_logits_cat.squeeze().numpy()
                            te_pred = te_pred * float(y_std) + float(y_mean)
                            best_test_metric = float(math.sqrt(mean_squared_error(test_y_cat.numpy(), te_pred)))
                else:
                    early_stop_counter += 1

            if early_stop_counter >= patience:
                total_epochs = epoch + 1
                print(f"{tag} Early stopping at epoch {total_epochs}")
                break

        if best_state is not None:
            enc.load_state_dict(best_state['enc'])
            self_attn.load_state_dict(best_state['self_attn'])
            attn_norm.load_state_dict(best_state['attn_norm'])
            input_proj.load_state_dict(best_state['input_proj'])
            row_ctx_proj.load_state_dict(best_state['row_ctx_proj'])
            dgm_module.load_state_dict(best_state['dgm_module'])
            gnn.load_state_dict(best_state['gnn'])
            with torch.no_grad():
                column_embed.copy_(best_state['column_embed'])
                pool_query.copy_(best_state['pool_query'])

        enc.eval()
        with torch.no_grad():
            Z_train = enc.net(X_train)
            Z_val = enc.net(X_val)
            Z_test = enc.net(X_test)

        return (
            enc,
            Z_train.detach().cpu(),
            Z_val.detach().cpu(),
            Z_test.detach().cpu(),
            int(total_epochs),
            {
                'best_val_metric': float(best_val_metric) if best_val_metric is not None else float('nan'),
                'best_test_metric': float(best_test_metric) if best_test_metric is not None else float('nan'),
            },
        )

    # encoding/columnwise: ExcelFormer-style mini-batch Self-Attn -> pooling -> DGM -> GCN -> Self-Attn decode as feature-transform.
    use_feature_gnn = gnn_stage in ['encoding', 'columnwise']
    feature_gnn = None
    if use_feature_gnn:
        dgm_k = int(config.get('dgm_k', config.get('gnn_knn', 5)))
        dgm_distance = config.get('dgm_distance', 'euclidean')
        attn_dim = int(config.get('gnn_attn_dim', int(config.get('gnn_hidden', 64))))
        attn_heads = int(config.get('gnn_num_heads', config.get('gnn_attn_heads', 4)))
        gnn_hidden = int(config.get('gnn_hidden', 64))
        gnn_out_dim = int(config.get('gnn_out_dim', attn_dim))
        gnn_dropout = float(config.get('gnn_dropout', 0.1))

        class FeatureGNN(nn.Module):
            def __init__(self, num_cols: int):
                super().__init__()
                self.num_cols = int(num_cols)
                self.attn_in = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)
                self.attn_norm = nn.LayerNorm(attn_dim)
                self.attn_out = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)
                self.attn_out_norm = nn.LayerNorm(attn_dim)
                self.input_proj = nn.Linear(1, attn_dim)
                self.gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim, num_layers=2)
                self.gcn_to_attn = nn.Linear(gnn_out_dim, attn_dim)
                self.out_proj = nn.Linear(attn_dim, 1)
                self.column_embed = nn.Parameter(torch.randn(self.num_cols, attn_dim))
                self.pool_query = nn.Parameter(torch.randn(attn_dim))

                self.ffn_pre = nn.Sequential(
                    nn.Linear(attn_dim, attn_dim * 2),
                    nn.GELU(),
                    nn.Dropout(gnn_dropout),
                    nn.Linear(attn_dim * 2, attn_dim),
                )
                self.ffn_post = nn.Sequential(
                    nn.Linear(attn_dim, attn_dim * 2),
                    nn.GELU(),
                    nn.Dropout(gnn_dropout),
                    nn.Linear(attn_dim * 2, attn_dim),
                )
                self.fusion_alpha_param = nn.Parameter(torch.tensor(-0.847))

                class DGMEmbedWrapper(nn.Module):
                    def forward(self, x, A=None):
                        return x

                self.dgm_module = DGM_d(DGMEmbedWrapper(), k=dgm_k, distance=dgm_distance)
                self.dropout = nn.Dropout(gnn_dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, num_cols]
                B = int(x.shape[0])
                tokens = self.input_proj(x.unsqueeze(-1)) + self.column_embed.unsqueeze(0)
                tokens_norm = self.attn_norm(tokens)
                attn_out1, _ = self.attn_in(tokens_norm, tokens_norm, tokens_norm)
                tokens_attn = tokens + attn_out1
                tokens_attn = tokens_attn + self.ffn_pre(self.attn_norm(tokens_attn))

                pool_logits = (tokens_attn * self.pool_query).sum(dim=2) / math.sqrt(attn_dim)
                pool_weights = torch.softmax(pool_logits, dim=1)
                row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)
                row_emb_std = _standardize(row_emb, dim=0)

                row_emb_batched = row_emb_std.unsqueeze(0)
                if hasattr(self.dgm_module, 'k'):
                    self.dgm_module.k = int(min(int(self.dgm_module.k), max(1, B - 1)))
                row_emb_dgm, edge_index_dgm, logprobs_dgm = self.dgm_module(row_emb_batched, A=None)
                row_emb_dgm = row_emb_dgm.squeeze(0)
                edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, B)

                gcn_out = self.gnn(row_emb_dgm, edge_index_dgm)
                gcn_ctx = self.gcn_to_attn(gcn_out).unsqueeze(1)
                tokens_with_ctx = tokens_attn + gcn_ctx
                tokens_ctx_norm = self.attn_out_norm(tokens_with_ctx)
                attn_out2, _ = self.attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
                tokens_mid = tokens_with_ctx + attn_out2
                tokens_out = tokens_mid + self.ffn_post(self.attn_out_norm(tokens_mid))
                recon = self.out_proj(self.dropout(tokens_out)).squeeze(-1)

                fusion_alpha = torch.sigmoid(self.fusion_alpha_param)
                x_fused = x + fusion_alpha * (recon - x)
                return x_fused, logprobs_dgm

        feature_gnn = FeatureGNN(in_dim).to(device)

    params = list(enc.parameters())
    if use_feature_gnn and feature_gnn is not None:
        params += list(feature_gnn.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def run_epoch(split='train'):
        if split == 'train':
            enc.train()
            X = X_train
        elif split == 'val':
            enc.eval()
            X = X_val
        else:
            enc.eval()
            X = X_test
        total = 0.0
        n = 0
        for i in range(0, X.size(0), batch_size):
            xb = X[i:i + batch_size]
            m = mask_generator(p_m, xb)
            m_label, x_tilde = pretext_generator(m, xb)
            if split == 'train':
                opt.zero_grad()
            logprobs_dgm = None
            if use_feature_gnn and feature_gnn is not None:
                x_tilde, logprobs_dgm = feature_gnn(x_tilde)

            h, mask_logits, recon = enc(x_tilde)

            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_label)
            recon_loss = F.mse_loss(recon, xb)
            loss = mask_loss + alpha * recon_loss
            if logprobs_dgm is not None:
                loss = loss + (-logprobs_dgm.mean() * 0.01)
            if split == 'train':
                loss.backward()
                opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(1, n)

    best_val = float('inf')
    no_improve = 0
    early_epoch = 0
    total_epochs = epochs
    min_epochs = patience + 1
    for ep in tqdm(range(epochs), desc="VIME Core Training"):
        train_loss = run_epoch('train')
        val_loss = run_epoch('val')
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            no_improve = 0
            early_epoch = ep + 1
            best_state = {
                'enc': {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                'feature_gnn': None,
            }
            if use_feature_gnn and feature_gnn is not None:
                best_state['feature_gnn'] = copy.deepcopy(feature_gnn.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience and (ep + 1) >= min_epochs:
                total_epochs = ep + 1
                break

    # Restore best
    enc.load_state_dict(best_state['enc'])
    if use_feature_gnn and feature_gnn is not None and best_state.get('feature_gnn', None) is not None:
        feature_gnn.load_state_dict(best_state['feature_gnn'])
    enc.eval()
    with torch.no_grad():
        Z_train = enc.net(X_train)
        Z_val = enc.net(X_val)
        Z_test = enc.net(X_test)

    return enc, Z_train.detach().cpu(), Z_val.detach().cpu(), Z_test.detach().cpu(), total_epochs, None


def _evaluate_baseline(Z_train: np.ndarray, y_train: np.ndarray,
                       Z_val: np.ndarray, y_val: np.ndarray,
                       Z_test: np.ndarray, y_test: np.ndarray,
                       task_type: str,
                       config: Dict[str, Any] | None = None,
                       stage: str | None = None):
    if task_type in ['binclass', 'multiclass']:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z_train, y_train)
        if task_type == 'binclass':
            val_prob = clf.predict_proba(Z_val)[:, 1]
            test_prob = clf.predict_proba(Z_test)[:, 1]
            val_metric = roc_auc_score(y_val, val_prob)
            test_metric = roc_auc_score(y_test, test_prob)
        else:
            val_pred = clf.predict(Z_val)
            test_pred = clf.predict(Z_test)
            val_metric = accuracy_score(y_val, val_pred)
            test_metric = accuracy_score(y_test, test_pred)
    else:
        cfg = config or {}
        # 1) Drop near-constant columns (to reduce ill-conditioning)
        drop_min_std = float(cfg.get('baseline_drop_min_std', 1e-6))
        z_mean = Z_train.mean(axis=0, keepdims=True)
        z_std = Z_train.std(axis=0, keepdims=True)
        keep_mask = (z_std >= drop_min_std).reshape(-1)
        Z_train_k = Z_train[:, keep_mask] if keep_mask.any() else Z_train
        Z_val_k = Z_val[:, keep_mask] if keep_mask.any() else Z_val
        Z_test_k = Z_test[:, keep_mask] if keep_mask.any() else Z_test
        z_mean_k = z_mean[:, keep_mask] if keep_mask.any() else z_mean
        z_std_k = z_std[:, keep_mask] if keep_mask.any() else z_std

        # 2) Standardize with safe minimum std to avoid explosions from tiny std
        min_std = float(cfg.get('baseline_min_std', 1e-3))
        small_mask = z_std_k < min_std
        z_std_safe = np.where(small_mask, 1.0, z_std_k)
        Z_train_s = (Z_train_k - z_mean_k) / z_std_safe
        Z_val_s = (Z_val_k - z_mean_k) / z_std_safe
        Z_test_s = (Z_test_k - z_mean_k) / z_std_safe

        # 3) Optional per-sample L2 normalization (unit-length rows)
        if bool(cfg.get('baseline_l2norm', True)):
            def l2norm(a: np.ndarray) -> np.ndarray:
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                return a / np.maximum(nrm, 1e-6)
            Z_train_s = l2norm(Z_train_s)
            Z_val_s = l2norm(Z_val_s)
            Z_test_s = l2norm(Z_test_s)

        if _dbg_enabled(cfg):
            num_dropped = int((~keep_mask).sum()) if keep_mask.size else 0
            num_small = int(np.sum(small_mask)) if small_mask.size else 0
            _dbg_print(cfg, f"[debug][{stage or 'baseline'}] cols: kept={Z_train_s.shape[1]}, dropped={num_dropped}; Z std min={float(z_std_k.min()) if z_std_k.size>0 else 0:.3e}, max={float(z_std_k.max()) if z_std_k.size>0 else 0:.3e}, small_cols={num_small} (min_std={min_std})")

        # Standardize y for fitting, report RMSE in original units
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train) + 1e-8)
        y_train_s = (y_train - y_mean) / y_std

        # Use Ridge for better conditioning on high-dim embeddings
        ridge_alpha = float(cfg.get('ridge_alpha', 10.0))
        reg = Ridge(alpha=ridge_alpha)
        reg.fit(Z_train_s, y_train_s)
        val_pred_s = reg.predict(Z_val_s)
        test_pred_s = reg.predict(Z_test_s)
        val_pred = val_pred_s * y_std + y_mean
        test_pred = test_pred_s * y_std + y_mean

        # Debug: check prediction stats
        if _dbg_enabled(cfg):
            def stats(a: np.ndarray) -> str:
                return f"mean={float(np.mean(a)):.4g}, std={float(np.std(a)):.4g}, min={float(np.min(a)):.4g}, max={float(np.max(a)):.4g}, nan={int(np.isnan(a).sum())}, inf={int(np.isinf(a).sum())}"
            _dbg_print(cfg, f"[debug][{stage or 'baseline'}] y_val stats: {stats(y_val)} | pred stats: {stats(val_pred)}")
            _dbg_print(cfg, f"[debug][{stage or 'baseline'}] y_test stats: {stats(y_test)} | pred stats: {stats(test_pred)}")

        val_metric = math.sqrt(mean_squared_error(y_val, val_pred))
        test_metric = math.sqrt(mean_squared_error(y_test, test_pred))
        if _dbg_enabled(cfg):
            naive_val = math.sqrt(mean_squared_error(y_val, np.full_like(y_val, y_mean)))
            naive_test = math.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_mean)))
            _dbg_print(cfg, f"[debug][{stage or 'baseline'}] RMSE: val={val_metric:.6g}, test={test_metric:.6g}; naive(mean) val={naive_val:.6g}, test={naive_test:.6g}")
    return val_metric, test_metric


def encoding_fn(*args, **kwargs):
    # Placeholder to satisfy injector naming; actual encoding happens in vime_core_fn
    return args[0]


def columnwise_fn(*args, **kwargs):
    # Placeholder to satisfy injector naming; actual encoding/columnwise happens inside vime_core_fn.
    return args[0] if len(args) > 0 else None


def decoding_fn(*args, **kwargs):
    # Placeholder
    return args[0]


def main(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
         dataset_results: Dict[str, Any], config: Dict[str, Any], gnn_stage: str):
    task_type = _task_type_from_info(dataset_results)
    # Stage: start
    gnn_early_stop_epochs = 0
    if gnn_stage == 'start':
        train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)

    # Stage: materialize
    mat = materialize_fn(train_df, val_df, test_df, dataset_results, config)
    if gnn_stage == 'materialize':
        mat, gnn_early_stop_epochs = gnn_after_materialize_arrays(mat, config, task_type)

    # Core VIME (self-supervised) with possible encoding/columnwise GNN forward
    enc, Z_train, Z_val, Z_test, early_stop_epochs, core_metrics = vime_core_fn(mat, config, task_type, gnn_stage)

    # Decoding stage: train GNN as downstream classifier/regressor on Z
    best_val_metric = None
    best_test_metric = None
    if gnn_stage == 'decoding':
        best_val_metric = (core_metrics or {}).get('best_val_metric', float('nan'))
        best_test_metric = (core_metrics or {}).get('best_test_metric', float('nan'))
        # Strict alignment: decoding stage does not expose GNN early-stop epochs (only start/materialize).
        gnn_early_stop_epochs = 0
    else:
        # Baseline linear/logistic on embeddings
        y_train = mat['y_train']; y_val = mat['y_val']; y_test = mat['y_test']
        best_val_metric, best_test_metric = _evaluate_baseline(
            Z_train.numpy(), y_train, Z_val.numpy(), y_val, Z_test.numpy(), y_test,
            task_type, config=config, stage=gnn_stage
        )

    results = {
        'stage': gnn_stage,
        'early_stop_epochs': early_stop_epochs,  # 訓練結束的總輪數
        'gnn_early_stop_epochs': int(gnn_early_stop_epochs),
        'best_val_metric': float(best_val_metric if best_val_metric is not None else -1.0),
        'best_test_metric': float(best_test_metric if best_test_metric is not None else -1.0),
    }
    return results










#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models vime --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models vime --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models vime --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models vime --gnn_stages all --epochs 2
#  python main.py --dataset helena --models vime --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models vime --gnn_stages all --epochs 2


