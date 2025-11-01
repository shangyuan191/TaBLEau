import time
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

# Try to import DGL and related graph conv layers. If unavailable, set a flag so
# code can either raise a clear error later or handle a non-DGL path.
try:
    import dgl
    from dgl.nn import GraphConv, GATConv
    DGL_AVAILABLE = True
except Exception:
    DGL_AVAILABLE = False
    # Provide placeholders so module-level class definitions don't fail at import time.
    GraphConv = None
    GATConv = None
# Use PyG (torch_geometric) for a simple GCN implementation similar to excelformer
try:
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except Exception:
    GCNConv = None
    PYG_AVAILABLE = False




def build_knn_graph(X: np.ndarray, k: int = 5):
    """Build a kNN graph (k neighbors) and return PyG-style edge_index tensor.

    Returns edge_index as torch.LongTensor of shape [2, num_edges].
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    src = []
    dst = []
    for i in range(n):
        for j in indices[i]:
            if j == i:
                continue
            src.append(i)
            dst.append(j)
    # add reverse edges
    src2 = src + dst
    dst2 = dst + src
    # create edge_index tensor
    edge_index = torch.tensor([src2, dst2], dtype=torch.long)
    return edge_index


class NodeInitializer(nn.Module):
    def __init__(self, in_dim, out_dim, p_dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.do = nn.Dropout(p_dropout)

    def forward(self, x):
        return self.do(F.relu(self.lin(x)))


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_layers=2, p_dropout=0.2, n_out=2):
        super().__init__()
        self.init = NodeInitializer(in_dim, hidden_dim, p_dropout=p_dropout)
        self.convs = nn.ModuleList()
        if PYG_AVAILABLE:
            for _ in range(max(1, n_layers)):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        else:
            # fallback to linear layers if PyG not available
            for _ in range(max(1, n_layers)):
                self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(p_dropout)
        self.head = nn.Linear(hidden_dim, n_out)

    def forward(self, edge_index, x):
        h = self.init(x)
        for conv in self.convs:
            if PYG_AVAILABLE and isinstance(conv, GCNConv):
                h = conv(h, edge_index)
            else:
                h = F.relu(conv(h))
            h = self.dropout(h)
        out = self.head(h)
        return out


class SimpleGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_layers=2, n_heads=4, p_dropout=0.2, n_out=2):
        super().__init__()
        self.init = NodeInitializer(in_dim, hidden_dim, p_dropout=p_dropout)
        self.gat_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // n_heads, num_heads=n_heads, feat_drop=p_dropout, attn_drop=p_dropout, activation=F.elu))
        self.head = nn.Linear(hidden_dim, n_out)

    def forward(self, g, x):
        h = self.init(x)
        for layer in self.gat_layers:
            h = layer(g, h)
            h = h.flatten(1)
        out = self.head(h)
        return out


def train_node_classification(edge_index, features, labels, train_idx, val_idx, test_idx, is_classification=True, model_type='gcn', hidden_dim=128, epochs=200, lr=1e-3, patience=10, device='cpu'):
    """Train a small GCN using PyG-style edge_index. Returns metrics and metadata."""
    device = torch.device(device)
    X = torch.tensor(features, dtype=torch.float32).to(device)
    # labels will be prepared by caller
    y = torch.tensor(labels).to(device)

    # determine output dim
    if is_classification:
        if y.dtype in (torch.float32, torch.float64):
            # binary classification with float labels 0/1 -> output dim 1
            n_out = 1
        else:
            n_out = int(torch.max(y).item()) + 1
    else:
        n_out = 1

    model = SimpleGCN(in_dim=features.shape[1], hidden_dim=hidden_dim, n_layers=2, p_dropout=0.2, n_out=n_out)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -np.inf if is_classification else np.inf
    best_epoch = 0
    best_state = None
    early_stop_epoch = epochs

    # convert indices to torch tensors
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)

    for epoch in range(epochs):
        model.train()
        logits = model(edge_index.to(device), X)
        if is_classification:
            if n_out == 1:
                logits_train = logits[train_idx_t][:, 0]
                loss = F.binary_cross_entropy_with_logits(logits_train, y[train_idx_t].float())
            else:
                loss = F.cross_entropy(logits[train_idx_t], y[train_idx_t].long())
        else:
            loss = F.mse_loss(logits[train_idx_t].squeeze(), y[train_idx_t].float())

        opt.zero_grad()
        loss.backward()
        opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            logits_eval = model(edge_index.to(device), X)
            if is_classification:
                if n_out == 1:
                    probs = torch.sigmoid(logits_eval[:, 0]).cpu().numpy()
                    try:
                        val_metric = roc_auc_score(y[val_idx_t].cpu().numpy(), probs[val_idx])
                    except Exception:
                        val_metric = accuracy_score((probs[val_idx] > 0.5).astype(int), y[val_idx_t].cpu().numpy())
                else:
                    probs = F.softmax(logits_eval, dim=1).cpu().numpy()
                    val_metric = accuracy_score(y[val_idx_t].cpu().numpy(), probs[val_idx].argmax(axis=1))
            else:
                preds = logits_eval[val_idx_t].cpu().numpy().squeeze()
                val_metric = -mean_squared_error(y[val_idx_t].cpu().numpy(), preds, squared=False)

        improved = (val_metric > best_val) if is_classification else (val_metric < best_val)
        if improved:
            best_val = val_metric
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            early_stop_epoch = epoch
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_final = model(edge_index.to(device), X)
        if is_classification:
            if n_out == 1:
                probs = torch.sigmoid(logits_final[:, 0]).cpu().numpy()
                try:
                    val_score = roc_auc_score(y[val_idx_t].cpu().numpy(), probs[val_idx])
                    test_score = roc_auc_score(y[test_idx_t].cpu().numpy(), probs[test_idx])
                except Exception:
                    val_score = accuracy_score((probs[val_idx] > 0.5).astype(int), y[val_idx_t].cpu().numpy())
                    test_score = accuracy_score((probs[test_idx] > 0.5).astype(int), y[test_idx_t].cpu().numpy())
            else:
                probs = F.softmax(logits_final, dim=1).cpu().numpy()
                val_score = accuracy_score(y[val_idx_t].cpu().numpy(), probs[val_idx].argmax(axis=1))
                test_score = accuracy_score(y[test_idx_t].cpu().numpy(), probs[test_idx].argmax(axis=1))
        else:
            val_score = mean_squared_error(y[val_idx_t].cpu().numpy(), logits_final[val_idx_t].cpu().numpy().squeeze(), squared=False)
            test_score = mean_squared_error(y[test_idx_t].cpu().numpy(), logits_final[test_idx_t].cpu().numpy().squeeze(), squared=False)

    return {
        'best_val_metric': float(val_score),
        'best_test_metric': float(test_score),
        'best_epoch': int(best_epoch),
        'gnn_early_stop_epochs': int(early_stop_epoch)
    }


def _preprocess_tables(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Preprocess three DataFrame splits into numpy feature matrices and label arrays.

    Returns: X_train, y_train, X_val, y_val, X_test, y_test (all numpy arrays).

    Behavior / assumptions:
    - If a column named 'target' exists it will be used as label; otherwise the last column is used.
    - Categorical/object columns are converted via one-hot encoding (pd.get_dummies).
    - Missing values are filled with 0 after one-hot encoding.
    - Label columns that are non-numeric are factorized to integer classes.
    """
    # Defensive copies
    dfs = [train_df.copy(), val_df.copy(), test_df.copy()]

    # determine target column
    if 'target' in dfs[0].columns:
        target_col = 'target'
    else:
        target_col = dfs[0].columns[-1]

    X_parts = []
    y_parts = []

    for df in dfs:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()
        X_parts.append(X)
        y_parts.append(y)

    # Concatenate features so get_dummies will produce same columns for all splits
    X_concat = pd.concat(X_parts, ignore_index=True)
    # One-hot encode categorical/object columns; keep NA indicator to avoid dropping missingness
    X_encoded = pd.get_dummies(X_concat, dummy_na=True)
    X_encoded = X_encoded.fillna(0)

    # Split back to individual numpy arrays
    n0 = len(X_parts[0])
    n1 = len(X_parts[1])
    # n2 = len(X_parts[2])
    X_all_np = X_encoded.to_numpy(dtype=np.float32)
    X_train = X_all_np[:n0]
    X_val = X_all_np[n0:n0 + n1]
    X_test = X_all_np[n0 + n1:]

    # Process labels: factorize non-numeric labels
    y_train = y_parts[0]
    y_val = y_parts[1]
    y_test = y_parts[2]

    def _proc_y(y_series):
        if pd.api.types.is_numeric_dtype(y_series):
            return y_series.to_numpy()
        # factorize strings/categories to int labels
        vals, _ = pd.factorize(y_series)
        return vals

    y_train_np = np.asarray(_proc_y(y_train))
    y_val_np = np.asarray(_proc_y(y_val))
    y_test_np = np.asarray(_proc_y(y_test))

    return X_train, y_train_np, X_val, y_val_np, X_test, y_test_np


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage=None):
    """Entry point used by ModelRunner. Accepts split DataFrames and trains a GNN.

    Returns a dict with keys `best_val_metric`, `best_test_metric`, `elapsed_time`.
    """
    start = time.time()
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = _preprocess_tables(train_df, val_df, test_df)
        # build combined feature matrix and labels
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        n_train = len(X_train)
        n_val = len(X_val)
        n_test = len(X_test)

        # build kNN graph across all samples (default k=5)
        g = build_knn_graph(X_all, k=config.get('knn_k', 5) if config else 5)

        # indices
        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

        is_classification = 'class' in dataset_results.get('info', {}).get('task_type', 'binclass') or dataset_results.get('info', {}).get('task_type') == 'binclass'

        device = str(config.get('device', 'cpu')) if config else 'cpu'

        model_type = config.get('model_type', 'gcn') if config else 'gcn'
        hidden_dim = config.get('hidden_dim', 128) if config else 128
        epochs = config.get('epochs', 200) if config else 200
        lr = config.get('lr', 1e-3) if config else 1e-3
        patience = config.get('patience', 10) if config else 10

        res = train_node_classification(g, X_all, y_all, train_idx, val_idx, test_idx, is_classification=is_classification, model_type=model_type, hidden_dim=hidden_dim, epochs=epochs, lr=lr, patience=patience, device=device)

        elapsed = time.time() - start
        res['elapsed_time'] = elapsed
        res['model'] = 'tabgnn_refactor'
        return res
    except Exception as e:
        return {'error': str(e), 'model': 'tabgnn_refactor'}


#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models tabgnn --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models tabgnn --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models tabgnn --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models tabgnn --gnn_stages all --epochs 2
#  python main.py --dataset helena --models tabgnn --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models tabgnn --gnn_stages all --epochs 2