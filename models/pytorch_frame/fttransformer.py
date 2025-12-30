from __future__ import annotations

import argparse
import math
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.nn import GCNConv
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data import DataLoader
from torch_frame.data.stats import StatType
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.datasets import Yandex
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    ResNet,
)
from torch_frame.nn.conv import FTTransformerConvs
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame.transforms import CatToNumTransform, MutualInformationSort
from torch_frame.typing import NAStrategy

sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
from DGMlib.layers import DGM_d


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

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
                print(f"[DEVICE] Using cuda:{int(gpu_id)}")
                return device
            except Exception:
                pass
        print("[DEVICE] Using default cuda")
        return torch.device('cuda')
    print("[DEVICE] Using cpu")
    return torch.device('cpu')


def compute_adaptive_dgm_k(num_samples: int, num_features: int, dataset_name: str = '') -> int:
    base_k = int(np.sqrt(num_samples))
    feature_factor = 1.0 + np.log1p(num_features) / 10.0
    adjusted_k = int(base_k * feature_factor)
    if num_samples < 500:
        density_factor = 1.3
    elif num_samples > 5000:
        density_factor = 0.9
    else:
        density_factor = 1.0
    adaptive_k = int(adjusted_k * density_factor)
    upper_limit = min(30, max(15, int(4 * np.sqrt(num_samples))))
    if num_samples < 1000:
        upper_limit = min(20, int(3 * np.sqrt(num_samples)))
    adaptive_k = max(5, min(adaptive_k, upper_limit))
    print(f"[DGM-K] dataset={dataset_name or 'unknown'}, N={num_samples}, D={num_features}, k={adaptive_k}")
    return adaptive_k


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = torch.nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def knn_graph(x: Tensor, k: int, directed: bool = False) -> Tensor:
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_list = []
    N = x_np.shape[0]
    for i in range(N):
        for j in indices[i][1:]:
            edge_list.append([i, j])
            if not directed:
                edge_list.append([j, i])
    if not edge_list:
        edge_list = [[i, i] for i in range(N)]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def _standardize(x: Tensor, dim: int = 0, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _symmetrize_and_self_loop(edge_index: Tensor, num_nodes: int) -> Tensor:
    device = edge_index.device
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    self_nodes = torch.arange(num_nodes, device=device)
    self_edges = torch.stack([self_nodes, self_nodes], dim=0)
    ei = torch.cat([edge_index, rev, self_edges], dim=1)
    edge_ids = ei[0] * num_nodes + ei[1]
    unique_ids = torch.unique(edge_ids, sorted=False)
    ei0 = unique_ids // num_nodes
    ei1 = unique_ids % num_nodes
    return torch.stack([ei0, ei1], dim=0)


def row_level_embedding(x: Tensor, mode: str = 'mean') -> Tensor:
    if mode == 'mean':
        return x.mean(dim=1)
    if mode == 'percol_mean':
        return x.mean(dim=2)
    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------------------------------------------------------
# GNN pipelines around start/materialize stages (self-attn + DGM + GCN)
# -----------------------------------------------------------------------------

def gnn_after_start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, task_type: str):
    # 與 ExcelFormer 版本保持一致的裝置處理：優先 config['gpu']，其次預設 cuda，再退回 cpu
    gpu_id = config.get('gpu', None)
    if torch.cuda.is_available() and gpu_id is not None:
        try:
            device = torch.device(f'cuda:{int(gpu_id)}')
        except Exception:
            device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[START-GNN-DGM] Using device: {device}")

    # 固定 dgm_k（預設 10），不做資料集自適應，僅在 split 時做安全上限
    dgm_k = int(config.get('dgm_k', 10))
    print(f"[DGM-K] Using fixed dgm_k={dgm_k} (no dataset-dependent adapt)")

    dgm_distance = config.get('dgm_distance', 'euclidean')
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
        out_dim = num_classes
    else:
        out_dim = 1

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    num_cols = len(feature_cols)

    attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    input_proj = torch.nn.Linear(1, attn_dim).to(device)

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

    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    params = list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters())
    params += list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters())
    params += list(out_proj.parameters()) + list(dgm_module.parameters()) + [column_embed, pool_query]

    optimizer = torch.optim.Adam(params, lr=lr)
    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_states = None

    def forward_pass(x_tensor: Tensor, dgm_module_inst: torch.nn.Module) -> tuple[Tensor, Tensor, Tensor]:
        Ns = x_tensor.shape[0]
        x_in = input_proj(x_tensor.unsqueeze(-1))
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_weights = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)

        row_emb_std = _standardize(row_emb, dim=0)
        row_emb_batched = row_emb_std.unsqueeze(0)
        dgm_k_split = int(min(dgm_k, max(1, Ns - 1)))  # 安全上限，保持與 ExcelFormer 一致的處理方式
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)
        edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)

        gcn_out = gnn(row_emb_dgm, edge_index_dgm)
        logits = pred_head(gcn_out)

        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_with_ctx, tokens_with_ctx, tokens_with_ctx)
        recon = out_proj(tokens_out).squeeze(-1)

        E = edge_index_dgm.shape[1]
        avg_deg = E / max(1, Ns)
        print(f"[START-GNN-DGM] split Ns={Ns}, edges={E}, avg_deg={avg_deg:.2f}")

        return logits, recon, logprobs_dgm

    for epoch in range(gnn_epochs):
        dgm_module.train(); gnn.train(); attn_in.train(); attn_out.train(); input_proj.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()
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

        dgm_module.eval(); gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
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

    dgm_module.eval(); gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
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


def gnn_after_materialize_fn(train_tensor_frame: TensorFrame, val_tensor_frame: TensorFrame, test_tensor_frame: TensorFrame, config: dict, dataset_name: str, task_type: str):
    device = resolve_device(config)
    print(f"[MATERIALIZE-GNN-DGM] Using device: {device}")

    def tensor_frame_to_df(tensor_frame: TensorFrame) -> pd.DataFrame:
        col_names = tensor_frame.col_names_dict[stype.numerical]
        features = tensor_frame.feat_dict[stype.numerical].cpu().numpy()
        df = pd.DataFrame(features, columns=col_names)
        if hasattr(tensor_frame, 'y') and tensor_frame.y is not None:
            df['target'] = tensor_frame.y.cpu().numpy()
        return df

    train_df = tensor_frame_to_df(train_tensor_frame)
    val_df = tensor_frame_to_df(val_tensor_frame)
    test_df = tensor_frame_to_df(test_tensor_frame)

    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    feature_cols = [c for c in all_df.columns if c != 'target']
    num_cols = len(feature_cols)

    if 'dgm_k' in config:
        dgm_k = int(config['dgm_k'])
        print(f"[DGM-K] Using user-specified dgm_k={dgm_k}")
    else:
        dgm_k = 10
    dgm_distance = config.get('dgm_distance', 'euclidean')
    gnn_epochs = config.get('epochs', 200)
    patience = config.get('gnn_patience', 10)
    loss_threshold = config.get('gnn_loss_threshold', 1e-4)
    attn_dim = config.get('gnn_attn_dim', config.get('gnn_hidden', 64))
    gnn_hidden = config.get('gnn_hidden', 64)
    gnn_out_dim = config.get('gnn_out_dim', attn_dim)
    attn_heads = config.get('gnn_num_heads', 4)
    lr = config.get('gnn_lr', 1e-3)

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
        out_dim = num_classes
    else:
        out_dim = 1

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    input_proj = torch.nn.Linear(1, attn_dim).to(device)

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

    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    params = list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters())
    params += list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters())
    params += list(out_proj.parameters()) + list(dgm_module.parameters()) + [column_embed, pool_query]

    optimizer = torch.optim.Adam(params, lr=lr)
    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_states = None

    def forward_pass(x_tensor: Tensor, dgm_module_inst: torch.nn.Module) -> tuple[Tensor, Tensor, Tensor]:
        Ns = x_tensor.shape[0]
        x_in = input_proj(x_tensor.unsqueeze(-1))
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_weights = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)

        row_emb_std = _standardize(row_emb, dim=0)
        row_emb_batched = row_emb_std.unsqueeze(0)
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module_inst(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)
        edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)

        gcn_out = gnn(row_emb_dgm, edge_index_dgm)
        logits = pred_head(gcn_out)

        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_with_ctx, tokens_with_ctx, tokens_with_ctx)
        recon = out_proj(tokens_out).squeeze(-1)

        E = edge_index_dgm.shape[1]
        avg_deg = E / max(1, Ns)
        print(f"[MATERIALIZE-GNN-DGM] split Ns={Ns}, edges={E}, avg_deg={avg_deg:.2f}")

        return logits, recon, logprobs_dgm

    for epoch in range(gnn_epochs):
        dgm_module.train(); gnn.train(); attn_in.train(); attn_out.train(); input_proj.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()
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

        dgm_module.eval(); gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
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

        print(f"[MATERIALIZE-GNN-DGM] Epoch {epoch+1}/{gnn_epochs}, Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}{' ↓ (improved)' if improved else ''}")
        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"[MATERIALIZE-GNN-DGM] Early stopping at epoch {epoch+1} (Val Loss: {val_loss_val:.4f})")
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

    dgm_module.eval(); gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
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

    dataset = Yandex(train_df_gnn, val_df_gnn, test_df_gnn, name=dataset_name, task_type=task_type)
    dataset.materialize()

    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    categorical_transform = CatToNumTransform()
    categorical_transform.fit(train_tensor_frame, dataset.col_stats)
    train_tensor_frame = categorical_transform(train_tensor_frame)
    val_tensor_frame = categorical_transform(val_tensor_frame)
    test_tensor_frame = categorical_transform(test_tensor_frame)
    col_stats = categorical_transform.transformed_stats

    mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
    mutual_info_sort.fit(train_tensor_frame, col_stats)
    train_tensor_frame = mutual_info_sort(train_tensor_frame)
    val_tensor_frame = mutual_info_sort(val_tensor_frame)
    test_tensor_frame = mutual_info_sort(test_tensor_frame)

    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    return (
        train_loader,
        val_loader,
        test_loader,
        col_stats,
        mutual_info_sort,
        dataset,
        train_tensor_frame,
        val_tensor_frame,
        test_tensor_frame,
        gnn_early_stop_epochs,
    )




# -----------------------------------------------------------------------------
# FTTransformer pipeline
# -----------------------------------------------------------------------------

def start_fn(train_df, val_df, test_df):
    return train_df, val_df, test_df


def materialize_fn(train_df, val_df, test_df, dataset_results, config):
    print('Executing materialize_fn')
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    dataset_name = dataset_results['dataset']
    task_type = dataset_results['info']['task_type']
    device = resolve_device(config)
    print(f"[MATERIALIZE] Device: {device}")

    numerical_encoder_type = config.get('numerical_encoder', 'linear')
    channels = config.get('channels', 256)
    num_layers = config.get('num_layers', 4)

    dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    if numerical_encoder_type == 'linear':
        numerical_encoder = LinearEncoder()
    elif numerical_encoder_type == 'linearbucket':
        numerical_encoder = LinearBucketEncoder()
    elif numerical_encoder_type == 'linearperiodic':
        numerical_encoder = LinearPeriodicEncoder()
    else:
        raise ValueError(f'Unsupported encoder type: {numerical_encoder_type}')

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: numerical_encoder,
    }

    if is_classification:
        output_channels = dataset.num_classes
    else:
        output_channels = 1

    is_binary_class = is_classification and output_channels == 2
    if is_binary_class:
        metric_computer = AUROC(task='binary')
        metric = 'AUC'
    elif is_classification:
        metric_computer = Accuracy(task='multiclass', num_classes=output_channels)
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
        'col_stats': dataset.col_stats,
        'metric_computer': metric_computer,
        'metric': metric,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'out_channels': output_channels,
        'stype_encoder_dict': stype_encoder_dict,
        'device': device,
        'channels': channels,
        'num_layers': num_layers,
    }


def fttransformer_core_fn(material_outputs, config, task_type, gnn_stage=None):
    print('Executing fttransformer_core_fn')
    train_tensor_frame = material_outputs['train_tensor_frame']
    val_tensor_frame = material_outputs['val_tensor_frame']
    test_tensor_frame = material_outputs['test_tensor_frame']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    col_stats = material_outputs['col_stats']
    device = material_outputs['device']
    stype_encoder_dict = material_outputs['stype_encoder_dict']
    is_classification = material_outputs['is_classification']
    is_binary_class = material_outputs['is_binary_class']
    out_channels = material_outputs['out_channels']
    metric_computer = material_outputs['metric_computer']
    metric = material_outputs['metric']

    channels = config.get('channels', material_outputs.get('channels', 256))
    num_layers = config.get('num_layers', material_outputs.get('num_layers', 4))
    patience = config.get('patience', 10)
    loss_threshold = config.get('loss_threshold', 1e-4)

    gnn_hidden = config.get('gnn_hidden', 64)
    attn_heads = config.get('gnn_num_heads', 4)
    dgm_k = config.get('dgm_k', config.get('gnn_knn', 5))
    dgm_distance = config.get('dgm_distance', 'euclidean')
    gnn_dropout = config.get('gnn_dropout', 0.1)

    use_gnn = gnn_stage in ['encoding', 'columnwise', 'decoding']
    gnn = None
    dgm_module = None
    pool_query = None
    fusion_alpha_param = None

    if use_gnn:
        self_attn = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
        attn_norm = torch.nn.LayerNorm(channels).to(device)
        column_embed = torch.nn.Parameter(torch.randn(train_tensor_frame.num_cols, channels, device=device))
        pool_query = torch.nn.Parameter(torch.randn(channels, device=device))
        if gnn_stage != 'decoding':
            dgm_embed_f = torch.nn.Module()
            dgm_embed_f.forward = lambda x, A=None: x  # type: ignore
            dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
            gnn = SimpleGCN(channels, gnn_hidden, channels).to(device)
            self_attn_out = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, batch_first=True).to(device)
            gcn_to_attn = torch.nn.Linear(channels, channels).to(device)
            attn_out_norm = torch.nn.LayerNorm(channels).to(device)
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
            fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))
        else:
            dgm_embed_f = torch.nn.Module()
            dgm_embed_f.forward = lambda x, A=None: x  # type: ignore
            dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
            gnn = SimpleGCN(channels, gnn_hidden, out_channels).to(device)

    encoder = StypeWiseFeatureEncoder(
        out_channels=channels,
        col_stats=col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)

    backbone = FTTransformerConvs(
        channels=channels,
        num_layers=num_layers,
    ).to(device)

    decoder = None
    if gnn_stage != 'decoding':
        decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        ).to(device)
        for m in decoder:
            if not isinstance(m, ReLU):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    def model_forward(tf, mixup_encoded: bool = False):
        """
        前向傳播對齐 ExcelFormer：
        1. Encoding GNN（如果有）→ 融合到 x
        2. Backbone 處理 x
        3. Columnwise GNN（在 backbone 之後）→ 融合到 x
        4. Decoding GNN（作為 decoder）→ 直接輸出預測
        """
        x, _ = encoder(tf)
        batch_size, num_cols, channels_ = x.shape

        # ======================== 編碼階段 GNN ========================
        if gnn_stage == 'encoding' and dgm_module is not None:
            # Step 1: Self-Attention 列間交互
            tokens = x + column_embed.unsqueeze(0)
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            ffn_out1 = ffn_pre(attn_norm(tokens_attn))
            tokens_attn = tokens_attn + ffn_out1

            # Step 2: Attention Pooling
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels_)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

            # Step 3: DGM 動態建圖
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)
            if hasattr(dgm_module, 'k'):
                dgm_module.k = int(min(int(dgm_module.k), max(1, x_pooled_batched.shape[1] - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])

            # Step 4: GCN 處理
            x_gnn_out = gnn(x_dgm, edge_index_dgm)

            # Step 5: Self-Attention 解碼
            gcn_ctx = gcn_to_attn(x_gnn_out).unsqueeze(1)
            tokens_with_ctx = tokens_attn + gcn_ctx
            tokens_ctx_norm = attn_out_norm(tokens_with_ctx)
            attn_out2, _ = self_attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
            tokens_mid = tokens_with_ctx + attn_out2
            ffn_out2 = ffn_post(attn_out_norm(tokens_mid))
            tokens_out = tokens_mid + ffn_out2

            # Step 6: 殘差融合
            fusion_alpha = torch.sigmoid(fusion_alpha_param)
            x = x + fusion_alpha * tokens_out

        # ======================== Backbone 處理 ========================
        x, x_cls = backbone(x)

        # ======================== Columnwise 階段 GNN（在 backbone 之後）========================
        if gnn_stage == 'columnwise' and dgm_module is not None:
            # Step 1: Self-Attention 列間交互
            tokens = x + column_embed.unsqueeze(0)
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            ffn_out1 = ffn_pre(attn_norm(tokens_attn))
            tokens_attn = tokens_attn + ffn_out1

            # Step 2: Attention Pooling
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels_)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

            # Step 3: DGM 動態建圖
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)
            if hasattr(dgm_module, 'k'):
                dgm_module.k = int(min(int(dgm_module.k), max(1, x_pooled_batched.shape[1] - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])

            # Step 4: GCN 處理
            x_gnn_out = gnn(x_dgm, edge_index_dgm)

            # Step 5: Self-Attention 解碼
            gcn_ctx = gcn_to_attn(x_gnn_out).unsqueeze(1)
            tokens_with_ctx = tokens_attn + gcn_ctx
            tokens_ctx_norm = attn_out_norm(tokens_with_ctx)
            attn_out2, _ = self_attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
            tokens_mid = tokens_with_ctx + attn_out2
            ffn_out2 = ffn_post(attn_out_norm(tokens_mid))
            tokens_out = tokens_mid + ffn_out2

            # Step 6: 殘差融合
            fusion_alpha = torch.sigmoid(fusion_alpha_param)
            x = x + fusion_alpha * tokens_out

        # ======================== Decoding 階段 GNN（作為 decoder）========================
        if gnn_stage == 'decoding' and dgm_module is not None:
            # Step 1: Self-Attention 列間交互
            tokens = x + column_embed.unsqueeze(0)
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1

            # Step 2: Attention Pooling
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels_)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

            # Step 3: DGM 動態建圖
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)
            if hasattr(dgm_module, 'k'):
                dgm_module.k = int(min(int(dgm_module.k), max(1, x_pooled_batched.shape[1] - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])

            # Step 4: GCN 直接輸出預測
            out = gnn(x_dgm, edge_index_dgm)
            y_mixedup = tf.y
            return out, y_mixedup

        # ======================== 標準輸出（非 GNN 或 none 階段）========================
        out = decoder(x_cls) if decoder is not None else x_cls
        return out, tf.y

    lr = config.get('lr', 0.001)
    gamma = config.get('gamma', 0.95)

    all_params = list(encoder.parameters()) + list(backbone.parameters())
    if decoder is not None:
        all_params += list(decoder.parameters())
    if gnn is not None:
        all_params += list(gnn.parameters())
    if use_gnn:
        all_params += list(self_attn.parameters()) + list(attn_norm.parameters()) + [column_embed]
        if gnn_stage != 'decoding':
            all_params += list(self_attn_out.parameters()) + list(attn_out_norm.parameters())
            all_params += list(gcn_to_attn.parameters()) + list(ffn_pre.parameters()) + list(ffn_post.parameters())
            all_params += list(dgm_module.parameters()) + [pool_query, fusion_alpha_param]
        else:
            all_params += list(dgm_module.parameters()) + [pool_query]

    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

    def train(epoch: int):
        encoder.train(); backbone.train()
        if decoder is not None:
            decoder.train()
        if gnn is not None:
            gnn.train()
        if use_gnn:
            self_attn.train(); attn_norm.train();
            if gnn_stage != 'decoding':
                self_attn_out.train(); attn_out_norm.train(); gcn_to_attn.train(); ffn_pre.train(); ffn_post.train(); dgm_module.train()
            else:
                dgm_module.train()
            if fusion_alpha_param is not None:
                fusion_alpha_param.requires_grad_(True)
            pool_query.requires_grad_(True)
            column_embed.requires_grad_(True)

        loss_accum = total_count = 0
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            pred, y_ref = model_forward(tf)
            if is_classification:
                loss = F.cross_entropy(pred, y_ref)
            else:
                loss = F.mse_loss(pred.view(-1), y_ref.view(-1))
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(y_ref)
            total_count += len(y_ref)
            optimizer.step()
        return loss_accum / total_count

    @torch.no_grad()
    def test(loader):
        encoder.eval(); backbone.eval()
        if decoder is not None:
            decoder.eval()
        if gnn is not None:
            gnn.eval()
        if use_gnn:
            self_attn.eval(); attn_norm.eval();
            if gnn_stage != 'decoding':
                self_attn_out.eval(); attn_out_norm.eval(); gcn_to_attn.eval(); ffn_pre.eval(); ffn_post.eval(); dgm_module.eval()
            else:
                dgm_module.eval()
        metric_computer.reset()
        loss_accum = total_count = 0
        for tf in loader:
            tf = tf.to(device)
            pred, y_ref = model_forward(tf)
            if is_classification:
                loss = F.cross_entropy(pred, y_ref)
            else:
                loss = F.mse_loss(pred.view(-1), y_ref.view(-1))
            loss_accum += float(loss) * len(y_ref)
            total_count += len(y_ref)
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
        return metric_computer.compute().item() ** 0.5, avg_loss

    best_val_loss = float('inf')
    best_val_metric = 0 if is_classification else float('inf')
    best_epoch = 0
    early_stop_counter = 0
    train_losses = []
    train_metrics = []
    val_metrics = []
    val_losses = []
    best_states = None

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

        improved = val_loss < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss
            best_val_metric = val_metric
            best_epoch = epoch
            early_stop_counter = 0
            best_states = {
                'encoder': encoder.state_dict(),
                'backbone': backbone.state_dict(),
                'decoder': decoder.state_dict() if decoder is not None else None,
                'gnn': gnn.state_dict() if gnn is not None else None,
            }
            if use_gnn:
                best_states['self_attn'] = self_attn.state_dict()
                best_states['attn_norm'] = attn_norm.state_dict()
                best_states['column_embed'] = column_embed.detach().clone()
                best_states['pool_query'] = pool_query.detach().clone()
                if gnn_stage != 'decoding':
                    best_states['self_attn_out'] = self_attn_out.state_dict()
                    best_states['attn_out_norm'] = attn_out_norm.state_dict()
                    best_states['gcn_to_attn'] = gcn_to_attn.state_dict()
                    best_states['ffn_pre'] = ffn_pre.state_dict()
                    best_states['ffn_post'] = ffn_post.state_dict()
                    best_states['dgm_module'] = dgm_module.state_dict() if dgm_module is not None else None
                    best_states['fusion_alpha_param'] = fusion_alpha_param.detach().clone() if fusion_alpha_param is not None else None
                else:
                    best_states['dgm_module'] = dgm_module.state_dict() if dgm_module is not None else None
        else:
            early_stop_counter += 1

        print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, Val Loss: {val_loss:.4f}, Val {metric}: {val_metric:.4f}{' *' if improved else ''}")
        lr_scheduler.step()
        if early_stop_counter >= patience:
            early_stop_epochs = epoch
            print(f"Early stopping at epoch {epoch}")
            break

    if best_states is not None:
        encoder.load_state_dict(best_states['encoder'])
        backbone.load_state_dict(best_states['backbone'])
        if decoder is not None and best_states['decoder'] is not None:
            decoder.load_state_dict(best_states['decoder'])
        if gnn is not None and best_states['gnn'] is not None:
            gnn.load_state_dict(best_states['gnn'])
        if use_gnn:
            self_attn.load_state_dict(best_states['self_attn'])
            attn_norm.load_state_dict(best_states['attn_norm'])
            with torch.no_grad():
                column_embed.copy_(best_states['column_embed'])
                pool_query.copy_(best_states['pool_query'])
            if gnn_stage != 'decoding':
                if 'self_attn_out' in best_states and best_states['self_attn_out'] is not None:
                    self_attn_out.load_state_dict(best_states['self_attn_out'])
                if 'attn_out_norm' in best_states and best_states['attn_out_norm'] is not None:
                    attn_out_norm.load_state_dict(best_states['attn_out_norm'])
                if 'gcn_to_attn' in best_states and best_states['gcn_to_attn'] is not None:
                    gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
                if 'ffn_pre' in best_states and best_states['ffn_pre'] is not None:
                    ffn_pre.load_state_dict(best_states['ffn_pre'])
                if 'ffn_post' in best_states and best_states['ffn_post'] is not None:
                    ffn_post.load_state_dict(best_states['ffn_post'])
                if 'dgm_module' in best_states and best_states['dgm_module'] is not None and dgm_module is not None:
                    dgm_module.load_state_dict(best_states['dgm_module'])
                if fusion_alpha_param is not None and 'fusion_alpha_param' in best_states and best_states['fusion_alpha_param'] is not None:
                    with torch.no_grad():
                        fusion_alpha_param.copy_(best_states['fusion_alpha_param'])
            else:
                if 'dgm_module' in best_states and best_states['dgm_module'] is not None and dgm_module is not None:
                    dgm_module.load_state_dict(best_states['dgm_module'])
        print(f"Restored best weights from epoch {best_epoch}")

    test_metric, test_loss = test(test_loader)
    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val {metric}: {best_val_metric:.4f}, Test {metric}: {test_metric:.4f}")

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
        'backbone': backbone,
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


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    print('FTTransformer - pipeline')
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    try:
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        gnn_early_stop_epochs = 0
        if gnn_stage == 'start':
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)

        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        if gnn_stage == 'materialize':
            train_tensor_frame = material_outputs['train_tensor_frame']
            val_tensor_frame = material_outputs['val_tensor_frame']
            test_tensor_frame = material_outputs['test_tensor_frame']
            (
                train_loader,
                val_loader,
                test_loader,
                col_stats,
                mutual_info_sort,
                dataset,
                train_tensor_frame,
                val_tensor_frame,
                test_tensor_frame,
                gnn_early_stop_epochs,
            ) = gnn_after_materialize_fn(
                train_tensor_frame,
                val_tensor_frame,
                test_tensor_frame,
                config,
                dataset_results['dataset'],
                task_type,
            )
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

        results = fttransformer_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)

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


# Example:
# python main.py --dataset kaggle_Audit_Data --models fttransformer --gnn_stages all --epochs 2
# python main.py --dataset eye --models fttransformer --gnn_stages all --epochs 2
# python main.py --dataset house --models fttransformer --gnn_stages all --epochs 2
# python main.py --dataset credit --models fttransformer --gnn_stages all --epochs 2
# python main.py --dataset openml_The_Office_Dataset --models fttransformer --gnn_stages all --epochs 2
