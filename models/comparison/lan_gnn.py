"""
LAN-GNN: Learnable Adaptive Neighborhood GNN for Tabular Prediction

This wrapper adapts the "Learning Adaptive Neighborhoods for GNNs" idea to tabular data:
- Builds a single graph over all rows (transductive)
- Learns adaptive k-NN via a differentiable top-k adjacency generator
- Applies dense graph convolution layers for message passing

Integration: comparison model with main(train_df, val_df, test_df, dataset_results, config, gnn_stage)
Outputs best val/test metrics and early stopping info similar to dgm.py
"""
import os
import sys
import time
import math
import logging
from types import SimpleNamespace
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader

try:
    # sklearn dependency; used to cap BLAS threads for large kNN builds
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

logger = logging.getLogger(__name__)

# Try import the adaptive neighborhood generator from the reference repo
LAN_REPO = '/home/skyler/ModelComparison/learning-adaptive-neighborhoods-for-gnns'
if LAN_REPO not in sys.path:
    sys.path.insert(0, LAN_REPO)

DGG_AVAILABLE = True
try:
    # Use a self-contained generator that doesn't require heavy args
    from dgm import DGG_StraightThrough
except Exception as e:
    DGG_AVAILABLE = False
    logger.warning(f"LAN-GNN: Could not import DGG generator: {e}")

# Optional fallback using PyTorch Geometric for fixed kNN
try:
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    logger.warning(f"LAN-GNN: PyG not available: {e}")


class DenseGraphConvolution(nn.Module):
    """Dense adjacency variant of GCN layer (no biases/norms)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
    def forward(self, x, adj_norm):
        # x: [N, Fin]; adj_norm: [N, N]
        support = x @ self.weight
        out = adj_norm @ support
        return out


def normalize_adj_dense(A: torch.Tensor) -> torch.Tensor:
    # Ensure no self-loops first, then add them
    N = A.size(0)
    A = A.clone()
    A[torch.arange(N), torch.arange(N)] = 0
    A_hat = A + torch.eye(N, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(-1)
    deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ A_hat @ D


class LocalTopKAdj(nn.Module):
    """Fallback adjacency generator: cosine similarity + Gumbel noise + top-k.
    Returns a dense [N,N] row-stochastic adjacency with hard top-k per row.
    """
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
    @staticmethod
    def _cosine_sim(x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)
        return x @ x.t()
    def forward(self, h: torch.Tensor, temperature: float = 0.5, noise: bool = True) -> torch.Tensor:
        # h: [N, F]
        sim = self._cosine_sim(h)  # [N, N], in [-1,1]
        sim = (sim + 1.0) / 2.0  # [0,1]
        if noise:
            # Gumbel noise
            g = -torch.log(-torch.log(torch.rand_like(sim).clamp_min(1e-9)).clamp_min(1e-9))
            sim = sim + g
        probs = F.softmax(sim / max(temperature, 1e-3), dim=-1)
        # Hard top-k one-hot per row
        topv, topi = torch.topk(probs, k=min(self.k, max(1, probs.size(-1)-1)), dim=-1, largest=True)
        adj = torch.zeros_like(probs)
        adj.scatter_(dim=-1, index=topi, src=torch.ones_like(topv))
        # Mask self-loops out of top-k if possible (optional)
        eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        adj = torch.where(eye > 0, torch.zeros_like(adj), adj)
        return adj


def build_knn_edge_index_sklearn(x_np: np.ndarray, k: int, sym: bool = True) -> torch.Tensor:
    """Build a kNN graph edge_index using sklearn on CPU.

    This avoids O(N^2) dense adjacency allocation and is suitable for large N.
    Returns a torch.LongTensor of shape [2, E].
    """
    if not isinstance(x_np, np.ndarray):
        x_np = np.asarray(x_np)
    x_np = x_np.astype(np.float32, copy=False)
    N = int(x_np.shape[0])
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    k_eff = int(min(max(int(k), 1), N - 1))
    # Prevent OpenBLAS / MKL from oversubscribing threads on large N
    if threadpool_limits is None:
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto', n_jobs=1)
    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            nbrs.fit(x_np)
            _, indices = nbrs.kneighbors(x_np)
    else:
        nbrs.fit(x_np)
        _, indices = nbrs.kneighbors(x_np)
    dst = indices[:, 1:].reshape(-1)
    src = np.repeat(np.arange(N, dtype=np.int64), k_eff)
    if sym:
        src_all = np.concatenate([src, dst.astype(np.int64, copy=False)], axis=0)
        dst_all = np.concatenate([dst.astype(np.int64, copy=False), src], axis=0)
    else:
        src_all = src
        dst_all = dst.astype(np.int64, copy=False)
    edge_index = np.stack([src_all, dst_all], axis=0)
    return torch.from_numpy(edge_index)


class LANLayer(nn.Module):
    """One layer of learnable adaptive neighborhood + dense GCN."""
    def __init__(self, dim, hidden, k=10, temperature=0.5, dropout=0.2, use_dgg: bool = True):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.dropout = dropout
        self.proj = nn.Linear(dim, hidden)
        self.use_dgg = bool(use_dgg) and DGG_AVAILABLE
        if self.use_dgg:
            # Generate per-forward adjacency with straight-through top-k
            # Use metric distance for stability on tabular data
            self.gen = DGG_StraightThrough(in_dim=hidden, latent_dim=hidden, k=k, hard=True, self_loops_noise=False, dist_fn='metric')
            self.dense_conv = DenseGraphConvolution(hidden, hidden)
        elif PYG_AVAILABLE:
            self.gen = None
            self.gcn = GCNConv(hidden, hidden)
        else:
            raise ImportError("Neither DGG generator nor PyG available for LAN-GNN")
        # Local fallback generator (used if DGG forward fails at runtime)
        self.local_gen = LocalTopKAdj(k=k)
    def forward(self, x, edge_index: torch.Tensor | None = None):
        # x: [N, Fin]
        h = F.relu(self.proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.use_dgg:
            try:
                # Use batch=1 for generator: expects [B, N, F]
                h_b = h.unsqueeze(0)
                # Use deterministic adjacency (no noise) for stability on small tabular datasets
                adj_soft = self.gen(h_b, temp=self.temperature, noise=False)[0]  # [N, N]
            except Exception as e:
                logger.warning(f"LAN-GNN: DGG generator failed: {e}. Falling back to local top-k generator.")
                self.use_dgg = False
                adj_soft = self.local_gen(h, temperature=self.temperature, noise=self.training)
            adj_norm = normalize_adj_dense(adj_soft)
            out = self.dense_conv(h, adj_norm)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            return out
        # PyG sparse fallback (recommended for large N)
        if not hasattr(self, 'gcn'):
            raise RuntimeError("LAN-GNN sparse fallback requires PyG (GCNConv) available")
        if edge_index is None:
            # Small-N convenience fallback only (O(N^2))
            with torch.no_grad():
                dist = torch.cdist(h, h)
                _, idx = torch.topk(dist, k=self.k + 1, largest=False)
                idx = idx[:, 1:]
                src = torch.arange(h.size(0), device=h.device).unsqueeze(1).expand(-1, self.k).reshape(-1)
                dst = idx.reshape(-1)
                edge_index = torch.stack([src, dst], dim=0)
        out = self.gcn(h, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class LANGNN(nn.Module):
    """Two-layer LAN-GNN for tabular node classification/regression."""
    def __init__(self, in_dim, hidden=128, out_dim=2, k=10, layers=2, dropout=0.2, task='classification', use_dgg: bool = True):
        super().__init__()
        self.task = task
        self.layers = nn.ModuleList()
        dim = in_dim
        for _ in range(layers):
            self.layers.append(LANLayer(dim=dim, hidden=hidden, k=k, temperature=0.5, dropout=dropout, use_dgg=use_dgg))
            dim = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )
    def forward(self, x, edge_index: torch.Tensor | None = None):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index=edge_index)
        logits = self.head(h)
        return logits


class TabularGraphDataset(Dataset):
    """Transductive: packs whole graph in a single item."""
    def __init__(self, X, y_onehot_or_float, mask):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y_onehot_or_float)
        self.mask = torch.BoolTensor(mask)
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return self.X, self.y, self.mask


def evaluate(model, X, masks, device, task, num_classes):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        logits = model(X_t)
        metrics = []
        for mask in masks:
            mask_t = torch.BoolTensor(mask).to(device)
            y = masks[mask]['y'] if isinstance(masks, dict) else None
        return None


def _compute_metric(y_true_np, pred_np, task, num_classes):
    # Sanitize potential NaNs/Infs for robust metric computation
    y_true_np = np.nan_to_num(y_true_np, nan=0.0, posinf=0.0, neginf=0.0)
    pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=0.0, neginf=0.0)
    if task == 'classification':
        y_pred_classes = pred_np.argmax(-1)
        y_true_classes = y_true_np.argmax(-1)
        if num_classes == 2:
            y_scores = torch.softmax(torch.tensor(pred_np), dim=-1)[:, 1]
            y_scores_np = np.nan_to_num(y_scores.numpy(), nan=0.0, posinf=1.0, neginf=0.0)
            return roc_auc_score(y_true_classes, y_scores_np)
        else:
            return accuracy_score(y_true_classes, y_pred_classes)
    else:
        mse = mean_squared_error(y_true_np.flatten(), pred_np.flatten())
        return float(np.sqrt(mse))


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage=None):
    try:
        logger.info("Running LAN-GNN (learnable adaptive kNN) model...")
        dataset_name = dataset_results['dataset']
        task_type = dataset_results['info']['task_type']
        seed = config.get('seed', 42)
        epochs = config.get('epochs', 200)
        lr = config.get('lr', 1e-3)
        patience = config.get('patience', 10)
        hidden_dim = config.get('gnn_hidden_dim', 128)
        num_layers = config.get('gnn_layers', 2)
        k = config.get('lan_k', 10)
        dropout = config.get('gnn_dropout', 0.2)
        device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # Merge splits (transductive)
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        if 'target' in all_df.columns:
            target_col = 'target'
        elif 'label' in all_df.columns:
            target_col = 'label'
        else:
            target_col = all_df.columns[-1]
        X = all_df.drop(columns=[target_col]).values
        y_raw = all_df[target_col].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        is_classification = task_type.lower() in ['binclass', 'multiclass']
        if is_classification:
            le = LabelEncoder()
            y_enc = le.fit_transform(y_raw)
            num_classes = int(len(np.unique(y_enc)))
            y = np.eye(num_classes)[y_enc.astype(int)].astype(np.float32)
        else:
            num_classes = 1
            y = y_raw.reshape(-1, 1).astype(np.float32)
        n_train = len(train_df)
        n_val = len(val_df)
        n_test = len(test_df)
        train_mask = np.zeros(len(all_df), dtype=bool)
        val_mask = np.zeros(len(all_df), dtype=bool)
        test_mask = np.zeros(len(all_df), dtype=bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train+n_val] = True
        test_mask[n_train+n_val:] = True
        # Memory safety: O(N^2) adjacency; downgrade to fixed kNN if too big
        num_samples = X.shape[0]
        estimated_memory_gb = (num_samples ** 2 * 4) / (1024 ** 3)
        memory_threshold_gb = 10.0
        use_learnable_adj = DGG_AVAILABLE and (estimated_memory_gb <= memory_threshold_gb)
        edge_index = None
        if not use_learnable_adj:
            if DGG_AVAILABLE:
                logger.warning(f"LAN-GNN: N={num_samples} would need {estimated_memory_gb:.2f}GB. Using sparse fixed kNN (PyG) fallback.")
            if not PYG_AVAILABLE:
                raise ImportError("LAN-GNN fallback for large N requires PyG (torch_geometric) installed")
            # Build a fixed kNN graph once on CPU; reuse for all epochs/layers
            edge_index = build_knn_edge_index_sklearn(X, k=k, sym=True).to(device)
        model = LANGNN(
            in_dim=X.shape[1],
            hidden=hidden_dim,
            out_dim=(num_classes if is_classification else 1),
            k=k,
            layers=num_layers,
            dropout=dropout,
            task=('classification' if is_classification else 'regression'),
            use_dgg=use_learnable_adj,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val = -float('inf') if is_classification else float('inf')
        best_test = -float('inf') if is_classification else float('inf')
        patience_counter = 0
        stopped_epoch = 0
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y).to(device)
        train_mask_t = torch.BoolTensor(train_mask).to(device)
        val_mask_t = torch.BoolTensor(val_mask).to(device)
        test_mask_t = torch.BoolTensor(test_mask).to(device)
        train_metrics = []
        val_metrics = []
        test_metrics = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_t, edge_index=edge_index)  # [N, C]
            if is_classification:
                train_logits = logits[train_mask_t]
                train_y = y_t[train_mask_t]
                # Use cross-entropy for both binary and multiclass (more stable than BCE with one-hot)
                loss = F.cross_entropy(train_logits, train_y.argmax(-1))
            else:
                train_logits = logits[train_mask_t]
                train_y = y_t[train_mask_t]
                loss = F.mse_loss(train_logits, train_y)
            loss.backward()
            optimizer.step()
            # Evaluate
            model.eval()
            with torch.no_grad():
                logits = model(X_t, edge_index=edge_index)
                def metric_for(mask_t):
                    pred = logits[mask_t].detach().cpu().numpy()
                    yt = y_t[mask_t].detach().cpu().numpy()
                    return _compute_metric(yt, pred, 'classification' if is_classification else 'regression', num_classes)
                tr_m = metric_for(train_mask_t)
                va_m = metric_for(val_mask_t)
                te_m = metric_for(test_mask_t)
            train_metrics.append(tr_m)
            val_metrics.append(va_m)
            test_metrics.append(te_m)
            improved = (va_m > best_val) if is_classification else (va_m < best_val)
            if improved:
                best_val = va_m
                best_test = te_m
                patience_counter = 0
            else:
                patience_counter += 1
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {tr_m:.4f}, Val: {va_m:.4f}, Test: {te_m:.4f}")
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                logger.info(f"Early stopping at epoch {stopped_epoch} (patience={patience})")
                break
        if stopped_epoch == 0:
            stopped_epoch = epoch + 1
        metric_name = 'AUC' if is_classification and (y.shape[1] == 2) else ('Acc' if is_classification else 'RMSE')
        logger.info(f"Best Val {metric_name}: {best_val:.4f}")
        logger.info(f"Best Test {metric_name}: {best_test:.4f}")
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_val_metric': best_val,
            'best_test_metric': best_test,
            'metric': metric_name,
            'is_classification': is_classification,
            'early_stop_epochs': stopped_epoch,
            'patience_counter': patience_counter,
        }
    except Exception as e:
        logger.error(f"LAN-GNN error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        is_classification = dataset_results['info']['task_type'].lower() in ['binclass', 'multiclass']
        return {
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': [],
            'best_val_metric': -float('inf') if is_classification else float('inf'),
            'best_test_metric': -float('inf') if is_classification else float('inf'),
            'error': str(e),
        }


#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models lan_gnn --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models lan_gnn --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models lan_gnn --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models lan_gnn --gnn_stages all --epochs 2
#  python main.py --dataset helena --models lan_gnn --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models lan_gnn --gnn_stages all --epochs 2