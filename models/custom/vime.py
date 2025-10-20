
from __future__ import annotations

import math
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

# Optional: PyG for GNN stages
try:
    from torch_geometric.nn import GCNConv
    _HAS_PYG = True
except Exception:  # pragma: no cover
    GCNConv = None
    _HAS_PYG = False


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
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("PyTorch Geometric is required for GNN stages.")
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
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


def gnn_after_start_fn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict[str, Any]):
    if not _HAS_PYG:
        return train_df, val_df, test_df, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = int(config.get('gnn_knn', 5))
    hidden_dim = int(config.get('gnn_hidden', 64))
    gnn_epochs = int(config.get('gnn_epochs', config.get('epochs', 200)))
    patience = int(config.get('gnn_patience', config.get('patience', 10)))
    min_epochs = patience + 1

    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    feat_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feat_cols].values, dtype=torch.float32, device=device)
    edge_index = knn_graph(x, k)
    in_dim = x.size(1)
    gnn = SimpleGCN(in_dim, hidden_dim, in_dim).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=0.01)

    # masks by split
    n_train, n_val = len(train_df), len(val_df)
    N = x.size(0)
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True

    best = float('inf')
    no_improve = 0
    best_epoch = 0
    total_epochs = gnn_epochs
    for epoch in tqdm(range(gnn_epochs), desc="GNN Start Training"):
        gnn.train()
        opt.zero_grad()
        out = gnn(x, edge_index)
        loss = F.mse_loss(out[train_mask], x[train_mask])
        loss.backward()
        opt.step()

        gnn.eval()
        with torch.no_grad():
            # Recompute outputs for validation after optimizer step
            val_out = gnn(x, edge_index)
            val_loss = F.mse_loss(val_out[val_mask], x[val_mask]).item()
        if val_loss + 1e-8 < best:
            best = val_loss
            no_improve = 0
            best_epoch = epoch + 1
        else:
            no_improve += 1
            if no_improve >= patience and (epoch + 1) >= min_epochs:
                total_epochs = epoch + 1
                break

    gnn.eval()
    with torch.no_grad():
        final = gnn(x, edge_index).detach().cpu().numpy()

    train_emb = final[:n_train]
    val_emb = final[n_train:n_train + n_val]
    test_emb = final[n_train + n_val:]
    emb_cols = [f'N_feature_{i+1}' for i in range(in_dim)]
    train_g = pd.DataFrame(train_emb, columns=emb_cols, index=train_df.index)
    val_g = pd.DataFrame(val_emb, columns=emb_cols, index=val_df.index)
    test_g = pd.DataFrame(test_emb, columns=emb_cols, index=test_df.index)
    for df_g, df_src in [(train_g, train_df), (val_g, val_df), (test_g, test_df)]:
        df_g['target'] = df_src['target'].values
    return train_g, val_g, test_g, total_epochs


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


def gnn_after_materialize_arrays(mat: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    if not _HAS_PYG:
        return mat, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = int(config.get('gnn_knn', 5))
    hidden_dim = int(config.get('gnn_hidden', 64))
    gnn_epochs = int(config.get('gnn_epochs', config.get('epochs', 200)))
    patience = int(config.get('gnn_patience', config.get('patience', 10)))
    min_epochs = patience + 1

    X_train = torch.tensor(mat['X_train'], dtype=torch.float32, device=device)
    X_val = torch.tensor(mat['X_val'], dtype=torch.float32, device=device)
    X_test = torch.tensor(mat['X_test'], dtype=torch.float32, device=device)
    x = torch.cat([X_train, X_val, X_test], dim=0)
    edge_index = knn_graph(x, k)
    in_dim = x.size(1)
    gnn = SimpleGCN(in_dim, hidden_dim, in_dim).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=0.01)

    n_train, n_val = X_train.size(0), X_val.size(0)
    N = x.size(0)
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True

    best = float('inf'); no_improve = 0; best_epoch = 0; total_epochs = gnn_epochs
    for epoch in tqdm(range(gnn_epochs), desc="GNN Materialize Training"):
        gnn.train(); opt.zero_grad()
        out = gnn(x, edge_index)
        loss = F.mse_loss(out[train_mask], x[train_mask])
        loss.backward(); opt.step()
        gnn.eval();
        with torch.no_grad():
            # Recompute outputs for validation after optimizer step
            val_out = gnn(x, edge_index)
            val_loss = F.mse_loss(val_out[val_mask], x[val_mask]).item()
        if val_loss + 1e-8 < best:
            best = val_loss; no_improve = 0; best_epoch = epoch + 1
        else:
            no_improve += 1
            if no_improve >= patience and (epoch + 1) >= min_epochs:
                total_epochs = epoch + 1
                break

    gnn.eval();
    with torch.no_grad():
        final = gnn(x, edge_index)
    final = final.detach().cpu().numpy()

    mat['X_train'] = final[:n_train]
    mat['X_val'] = final[n_train:n_train + n_val]
    mat['X_test'] = final[n_train + n_val:]
    return mat, total_epochs


def vime_core_fn(mat: Dict[str, Any], config: Dict[str, Any], task_type: str, gnn_stage: str = 'none'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=float(config.get('weight_decay', 1e-5)))

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
            h, mask_logits, recon = enc(x_tilde)

            # GNN at encoding/columnwise: forward-only graph over batch embeddings
            if gnn_stage in ['encoding', 'columnwise'] and _HAS_PYG and h.size(0) >= 3:
                k = int(config.get('gnn_knn', 5))
                edge_index = knn_graph(h, min(k, max(1, h.size(0) - 1)))
                gnn = SimpleGCN(h.size(1), int(config.get('gnn_hidden', 64)), h.size(1)).to(device)
                h = gnn(h, edge_index)

            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_label)
            recon_loss = F.mse_loss(recon, xb)
            loss = mask_loss + alpha * recon_loss
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
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience and (ep + 1) >= min_epochs:
                total_epochs = ep + 1
                break

    # Restore best
    enc.load_state_dict(best_state)
    enc.eval()
    with torch.no_grad():
        Z_train = enc.net(X_train)
        Z_val = enc.net(X_val)
        Z_test = enc.net(X_test)

    return enc, Z_train.detach().cpu(), Z_val.detach().cpu(), Z_test.detach().cpu(), total_epochs


def gnn_decoding_eval(Z_train: torch.Tensor, y_train: np.ndarray,
                      Z_val: torch.Tensor, y_val: np.ndarray,
                      Z_test: torch.Tensor, y_test: np.ndarray,
                      config: Dict[str, Any], task_type: str,
                      y_mean: float | None = None, y_std: float | None = None):
    if not _HAS_PYG:
        # Keep return arity consistent with normal path (val_metric, test_metric, total_epochs)
        return None, None, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.cat([Z_train, Z_val, Z_test], dim=0).to(device)
    n_train, n_val = Z_train.size(0), Z_val.size(0)
    N = X.size(0)
    k = int(config.get('gnn_knn', 5))
    edge_index = knn_graph(X, min(k, max(1, X.size(0) - 1)))

    hidden = int(config.get('gnn_hidden', 64))
    patience = int(config.get('gnn_patience', config.get('patience', 10)))
    epochs = int(config.get('gnn_epochs', config.get('epochs', 200)))
    min_epochs = patience + 1

    is_class = task_type in ['binclass', 'multiclass']
    if is_class:
        y_all = np.concatenate([y_train, y_val, y_test], axis=0)
        classes = np.unique(y_all)
        num_classes = len(classes)
        if task_type == 'binclass':
            out_dim = 1
        else:
            out_dim = num_classes
    else:
        out_dim = 1

    gnn = SimpleGCN(X.size(1), hidden, out_dim).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=0.01)

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    y_concat_np = np.concatenate([y_train, y_val, y_test], axis=0)
    if is_class:
        y_all = torch.tensor(y_concat_np, dtype=torch.long, device=device)
        y_all_loss = y_all
    else:
        # For regression, optionally standardize y for loss/early stopping
        if (y_mean is not None) and (y_std is not None):
            y_all_std_np = (y_concat_np.astype(np.float32) - y_mean) / y_std
            y_all_loss = torch.tensor(y_all_std_np, dtype=torch.float32, device=device).view(-1, 1)
        else:
            y_all_loss = torch.tensor(y_concat_np, dtype=torch.float32, device=device).view(-1, 1)

    # Initialize tracking
    best_val = None
    best_test = None
    no_improve = 0
    best_epoch = 0
    total_epochs = epochs
    # We early-stop by validation loss only
    best_val_loss = float('inf')
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for ep in tqdm(range(epochs), desc="GNN Decoding Training"):
        gnn.train(); opt.zero_grad()
        logits = gnn(X, edge_index)
        if is_class:
            if task_type == 'binclass':
                loss = bce(logits[train_mask].view(-1, 1), y_all_loss[train_mask].float().view(-1, 1))
            else:
                loss = ce(logits[train_mask], y_all_loss[train_mask])
        else:
            loss = mse(logits[train_mask], y_all_loss[train_mask])
        loss.backward(); opt.step()

        gnn.eval()
        with torch.no_grad():
            logits = gnn(X, edge_index)
            # Compute validation loss for early stopping
            if is_class:
                if task_type == 'binclass':
                    val_loss = bce(logits[val_mask].view(-1, 1), y_all_loss[val_mask].float().view(-1, 1)).item()
                else:
                    val_loss = ce(logits[val_mask], y_all_loss[val_mask]).item()
                # Compute metrics for reporting (not for early stop)
                if task_type == 'binclass':
                    val_prob = torch.sigmoid(logits[val_mask]).view(-1).detach().cpu().numpy()
                    test_prob = torch.sigmoid(logits[test_mask]).view(-1).detach().cpu().numpy()
                    val_metric = roc_auc_score(y_concat_np[n_train:n_train+n_val], val_prob)
                    test_metric = roc_auc_score(y_concat_np[n_train+n_val:], test_prob)
                else:
                    val_pred = logits[val_mask].argmax(dim=1).detach().cpu().numpy()
                    test_pred = logits[test_mask].argmax(dim=1).detach().cpu().numpy()
                    val_metric = accuracy_score(y_concat_np[n_train:n_train+n_val], val_pred)
                    test_metric = accuracy_score(y_concat_np[n_train+n_val:], test_pred)
            else:
                val_loss = mse(logits[val_mask], y_all_loss[val_mask]).item()
                # Metrics for reporting: RMSE in original units
                val_pred = logits[val_mask].detach().cpu().numpy()
                test_pred = logits[test_mask].detach().cpu().numpy()
                if (y_mean is not None) and (y_std is not None):
                    val_pred = val_pred * y_std + y_mean
                    test_pred = test_pred * y_std + y_mean
                val_true = y_concat_np[n_train:n_train+n_val].astype(np.float32).reshape(-1, 1)
                test_true = y_concat_np[n_train+n_val:].astype(np.float32).reshape(-1, 1)
                val_metric = math.sqrt(mean_squared_error(val_true, val_pred))
                test_metric = math.sqrt(mean_squared_error(test_true, test_pred))

        # Early stopping based on validation loss improvement
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_val = val_metric
            best_test = test_metric
            no_improve = 0
            best_epoch = ep + 1
        else:
            no_improve += 1
            if no_improve >= patience and (ep + 1) >= min_epochs:
                total_epochs = ep + 1
                break

    return best_val, best_test, total_epochs


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
    # Placeholder to satisfy injector naming; actual columnwise handled in vime_core_fn when selected
        # VIME 無columnwise conv層，但可於此直接forward GNN於encoder output
        mat = args[0] if len(args) > 0 else None
        config = kwargs.get('config', {})
        if mat is None:
            return None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train = torch.tensor(mat['X_train'], dtype=torch.float32, device=device)
        X_val = torch.tensor(mat['X_val'], dtype=torch.float32, device=device)
        X_test = torch.tensor(mat['X_test'], dtype=torch.float32, device=device)
        in_dim = X_train.size(1)
        hidden_dim = int(config.get('gnn_hidden', 64))
        k = int(config.get('gnn_knn', 5))
        # 合併所有資料
        x = torch.cat([X_train, X_val, X_test], dim=0)
        edge_index = knn_graph(x, min(k, max(1, x.size(0) - 1)))
        gnn = SimpleGCN(in_dim, hidden_dim, in_dim).to(device)
        with torch.no_grad():
            final = gnn(x, edge_index).detach().cpu().numpy()
        n_train, n_val = X_train.size(0), X_val.size(0)
        mat['X_train'] = final[:n_train]
        mat['X_val'] = final[n_train:n_train + n_val]
        mat['X_test'] = final[n_train + n_val:]
        return mat


def decoding_fn(*args, **kwargs):
    # Placeholder
    return args[0]


def main(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
         dataset_results: Dict[str, Any], config: Dict[str, Any], gnn_stage: str):
    # Stage: start
    gnn_early_stop_epochs = 0
    if gnn_stage == 'start':
        train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config)

    # Stage: materialize
    mat = materialize_fn(train_df, val_df, test_df, dataset_results, config)
    if gnn_stage == 'materialize':
        mat, gnn_early_stop_epochs = gnn_after_materialize_arrays(mat, config)

    # Core VIME (self-supervised) with possible encoding/columnwise GNN forward
    task_type = _task_type_from_info(dataset_results)
    enc, Z_train, Z_val, Z_test, early_stop_epochs = vime_core_fn(mat, config, task_type, gnn_stage)

    # Decoding stage: train GNN as downstream classifier/regressor on Z
    best_val_metric = None
    best_test_metric = None
    gnn_dec_early = 0
    if gnn_stage == 'decoding':
        y_train = mat['y_train']; y_val = mat['y_val']; y_test = mat['y_test']
        # Ensure 1D labels for sklearn metrics
        y_train = y_train.astype(int) if task_type != 'regression' else y_train.astype(float)
        y_val = y_val.astype(int) if task_type != 'regression' else y_val.astype(float)
        y_test = y_test.astype(int) if task_type != 'regression' else y_test.astype(float)
        # Pass y_mean/y_std for regression so RMSE is in original units but loss uses standardized y
        val_m, test_m, gnn_dec_early = gnn_decoding_eval(
            Z_train, y_train, Z_val, y_val, Z_test, y_test,
            config, task_type,
            y_mean=mat.get('y_mean', None), y_std=mat.get('y_std', None)
        )
        best_val_metric = val_m
        best_test_metric = test_m
        gnn_early_stop_epochs = gnn_dec_early
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


