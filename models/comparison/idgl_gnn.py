"""
IDGL-GNN: Iterative Deep Graph Learning style baseline for Tabular Prediction

This comparison baseline adapts the NeurIPS'20 IDGL idea to TaBLEau tabular splits:
- Build a single transductive graph over all rows (train+val+test)
- Small-N: learn an adaptive adjacency with IDGL's GraphLearner, then apply dense GCN
- Large-N: avoid dense NxN by falling back to a fixed kNN sparse graph (PyG)

Integration: comparison model with main(train_df, val_df, test_df, dataset_results, config, gnn_stage)
"""

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

logger = logging.getLogger(__name__)

# Optional PyG fallback for large N
try:
    from torch_geometric.nn import GCNConv

    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    logger.warning(f"IDGL-GNN: PyG not available: {e}")


def _ensure_idgl_on_path() -> None:
    this_file = Path(__file__).resolve()
    modelcomparison_root = this_file.parents[3]  # .../ModelComparison
    idgl_src = modelcomparison_root / "IDGL" / "src"
    if idgl_src.exists() and str(idgl_src) not in sys.path:
        sys.path.insert(0, str(idgl_src))


_IDGL_AVAILABLE = True
try:
    _ensure_idgl_on_path()
    from core.layers.graphlearn import GraphLearner
    from core.utils.generic_utils import to_cuda
    from core.utils.constants import VERY_SMALL_NUMBER
except Exception as e:
    _IDGL_AVAILABLE = False
    logger.warning(f"IDGL-GNN: Could not import IDGL modules: {e}")


def build_knn_edge_index_sklearn(x_np: np.ndarray, k: int, sym: bool = True) -> torch.Tensor:
    """Build a kNN edge_index using sklearn on CPU."""
    k_eff = int(max(1, min(k, x_np.shape[0] - 1)))
    nn_model = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean", n_jobs=1)

    if threadpool_limits is None:
        nn_model.fit(x_np)
        neighbors = nn_model.kneighbors(x_np, return_distance=False)
    else:
        with threadpool_limits(limits=1):
            nn_model.fit(x_np)
            neighbors = nn_model.kneighbors(x_np, return_distance=False)

    neighbors = neighbors[:, 1:]
    src = np.repeat(np.arange(x_np.shape[0]), k_eff)
    dst = neighbors.reshape(-1)

    if sym:
        src2 = np.concatenate([src, dst])
        dst2 = np.concatenate([dst, src])
        src, dst = src2, dst2

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    return edge_index


def _compute_metric(y_true_np: np.ndarray, pred_np: np.ndarray, task: str, num_classes: int) -> float:
    y_true_np = np.nan_to_num(y_true_np, nan=0.0, posinf=0.0, neginf=0.0)
    pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=0.0, neginf=0.0)

    if task == "classification":
        y_pred_classes = pred_np.argmax(-1)
        y_true_classes = y_true_np.argmax(-1)
        if num_classes == 2:
            probs_pos = torch.softmax(torch.tensor(pred_np), dim=-1)[:, 1].numpy()
            probs_pos = np.nan_to_num(probs_pos, nan=0.0, posinf=1.0, neginf=0.0)
            return float(roc_auc_score(y_true_classes, probs_pos))
        return float(accuracy_score(y_true_classes, y_pred_classes))

    mse = mean_squared_error(y_true_np.flatten(), pred_np.flatten())
    return float(np.sqrt(mse))


def _normalize_adj_dense(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization used by common dense GCN baselines."""
    n = A.size(0)
    A = A.clone()
    A[torch.arange(n), torch.arange(n)] = 0
    A_hat = A + torch.eye(n, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(-1)
    deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ A_hat @ D


class DenseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        layers = int(max(1, layers))
        self.dropout = float(dropout)
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
        )
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = x
        for i, w in enumerate(self.weights):
            h = adj_norm @ (h @ w)
            if i != len(self.weights) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class SparseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("Sparse fallback requires torch_geometric")
        layers = int(max(1, layers))
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        if layers == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class IDGLTabularNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        dropout: float,
        use_learned_adj: bool,
        knn_k: int,
        device: torch.device,
        graph_learn_hidden: Optional[int] = None,
        graph_metric_type: str = "attention",
    ):
        super().__init__()
        self.use_learned_adj = bool(use_learned_adj)
        self.knn_k = int(knn_k)
        self.device = device

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.layers = int(layers)
        self.dropout = float(dropout)

        self.dense_gcn = DenseGCN(self.in_dim, self.hidden_dim, self.out_dim, layers=self.layers, dropout=self.dropout)

        self.graph_learner: Optional[nn.Module] = None
        if self.use_learned_adj:
            if not _IDGL_AVAILABLE:
                raise ImportError("IDGL modules not available; cannot use learned adjacency")
            gl_hidden = int(graph_learn_hidden or hidden_dim)
            # IDGL GraphLearner expects 2D node feature input for cosine/attention; we use attention by default
            self.graph_learner = GraphLearner(
                input_size=in_dim,
                hidden_size=gl_hidden,
                topk=self.knn_k,
                epsilon=None,
                num_pers=16,
                metric_type=graph_metric_type,
                device=device,
            )

        self.cached_edge_index: Optional[torch.Tensor] = None
        self.sparse_gcn: Optional[SparseGCN] = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        adj_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_learned_adj:
            assert self.graph_learner is not None
            if adj_norm is None:
                raw_adj = self.graph_learner(x)  # [N,N] dense
                # Convert to row-stochastic adjacency
                adj = torch.softmax(raw_adj, dim=-1)
                adj_norm = _normalize_adj_dense(adj)
            return self.dense_gcn(x, adj_norm)

        # Fixed sparse fallback
        if edge_index is None:
            raise ValueError("edge_index is required when use_learned_adj=False")
        if self.sparse_gcn is None:
            # Lazily create sparse module to avoid requiring PyG when not used
            self.sparse_gcn = SparseGCN(
                in_dim=self.in_dim,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                layers=self.layers,
                dropout=self.dropout,
            ).to(self.device)
        return self.sparse_gcn(x, edge_index)


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage=None):
    try:
        logger.info("Running IDGL-GNN baseline...")

        task_type = dataset_results["info"]["task_type"].lower()
        seed = int(config.get("seed", 42))
        epochs = int(config.get("epochs", 200))
        lr = float(config.get("lr", 1e-3))
        patience = int(config.get("patience", 10))
        hidden_dim = int(config.get("gnn_hidden_dim", 128))
        num_layers = int(config.get("gnn_layers", 2))
        dropout = float(config.get("gnn_dropout", 0.2))
        k = int(config.get("idgl_k", config.get("lan_k", 10)))

        device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        if "target" in all_df.columns:
            target_col = "target"
        elif "label" in all_df.columns:
            target_col = "label"
        else:
            target_col = all_df.columns[-1]

        X = all_df.drop(columns=[target_col]).values
        y_raw = all_df[target_col].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        is_classification = task_type in ["binclass", "multiclass"]
        if is_classification:
            le = LabelEncoder()
            y_enc = le.fit_transform(y_raw)
            num_classes = int(len(np.unique(y_enc)))
            y = np.eye(num_classes)[y_enc.astype(int)].astype(np.float32)
            out_dim = num_classes
            task = "classification"
        else:
            num_classes = 1
            y = y_raw.reshape(-1, 1).astype(np.float32)
            out_dim = 1
            task = "regression"

        n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
        train_mask = np.zeros(len(all_df), dtype=bool)
        val_mask = np.zeros(len(all_df), dtype=bool)
        test_mask = np.zeros(len(all_df), dtype=bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True

        num_samples = X.shape[0]
        estimated_memory_gb = (num_samples**2 * 4) / (1024**3)
        memory_threshold_gb = float(config.get("idgl_dense_memory_gb", 10.0))
        use_learned_adj = _IDGL_AVAILABLE and (estimated_memory_gb <= memory_threshold_gb)

        edge_index = None
        if not use_learned_adj:
            if not PYG_AVAILABLE:
                raise ImportError("IDGL-GNN large-N fallback requires PyG (torch_geometric) installed")
            logger.warning(
                f"IDGL-GNN: N={num_samples} would need {estimated_memory_gb:.2f}GB for dense adj. "
                "Using sparse fixed kNN fallback."
            )
            edge_index = build_knn_edge_index_sklearn(X, k=k, sym=True).to(device)

        model = IDGLTabularNet(
            in_dim=X.shape[1],
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=num_layers,
            dropout=dropout,
            use_learned_adj=use_learned_adj,
            knn_k=k,
            device=device,
            graph_learn_hidden=int(config.get("idgl_graph_learn_hidden", hidden_dim)),
            graph_metric_type=str(config.get("idgl_graph_metric", "attention")),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y).to(device)
        train_mask_t = torch.BoolTensor(train_mask).to(device)
        val_mask_t = torch.BoolTensor(val_mask).to(device)
        test_mask_t = torch.BoolTensor(test_mask).to(device)

        best_val = -float("inf") if is_classification else float("inf")
        best_test = -float("inf") if is_classification else float("inf")
        patience_counter = 0
        stopped_epoch = 0

        train_metrics, val_metrics, test_metrics = [], [], []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            logits = model(X_t, edge_index=edge_index)

            if is_classification:
                loss = F.cross_entropy(logits[train_mask_t], y_t[train_mask_t].argmax(-1))
            else:
                loss = F.mse_loss(logits[train_mask_t], y_t[train_mask_t])

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_t, edge_index=edge_index)

                def metric_for(mask_t: torch.Tensor) -> float:
                    pred = logits[mask_t].detach().cpu().numpy()
                    yt = y_t[mask_t].detach().cpu().numpy()
                    return _compute_metric(yt, pred, task, num_classes)

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
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Train: {tr_m:.4f}, Val: {va_m:.4f}, Test: {te_m:.4f}"
                )

            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                logger.info(f"Early stopping at epoch {stopped_epoch} (patience={patience})")
                break

        if stopped_epoch == 0:
            stopped_epoch = epoch + 1

        metric_name = "AUC" if is_classification and num_classes == 2 else ("Acc" if is_classification else "RMSE")

        logger.info(f"Best Val {metric_name}: {best_val:.4f}")
        logger.info(f"Best Test {metric_name}: {best_test:.4f}")

        return {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_val_metric": best_val,
            "best_test_metric": best_test,
            "metric": metric_name,
            "is_classification": is_classification,
            "early_stop_epochs": stopped_epoch,
            "patience_counter": patience_counter,
            "used_learned_adj": use_learned_adj,
        }

    except Exception as e:
        logger.error(f"IDGL-GNN error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        is_classification = dataset_results["info"]["task_type"].lower() in ["binclass", "multiclass"]
        return {
            "train_metrics": [],
            "val_metrics": [],
            "test_metrics": [],
            "best_val_metric": None,
            "best_test_metric": None,
            "metric": "AUC" if is_classification else "RMSE",
            "is_classification": is_classification,
            "early_stop_epochs": 0,
            "patience_counter": 0,
            "error": str(e),
        }


#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models idgl --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models idgl --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models idgl --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models idgl --gnn_stages all --epochs 2
#  python main.py --dataset helena --models idgl --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models idgl --gnn_stages all --epochs 2