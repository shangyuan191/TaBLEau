"""GLCN baseline wrapper for TaBLEau (comparison model).

This integrates the TensorFlow 1.x implementation from the local repo:
- /home/skyler/ModelComparison/GLCN-tf

GLCN (Graph Learning Convolutional Network) is a *self-contained* row-graph
baseline:
- It first learns edge weights over a candidate edge set (SparseGraphLearn).
- It then performs GCN message passing for node classification.

TaBLEau integration contract:
- main(train_df, val_df, test_df, dataset_results, config, gnn_stage) -> dict

Runtime note:
- The upstream GLCN code depends on TensorFlow 1.x.
- TaBLEau's default environment in this workspace typically does NOT include TF1.

This wrapper therefore executes GLCN in a subprocess. Set:
- env var `GLCN_PYTHON` to a python executable that can import TensorFlow 1.x

Stage mapping (PyTorch-Frame five-stage):
- Primary: `columnwise` (row-row interactions via learned graph + message passing)
- Also: `encoding` (feature projection / normalization), `decoding` (classifier head)

This is not a stage-injection model; `gnn_stage` is ignored.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

GLCN_TF_REPO = "/home/skyler/ModelComparison/GLCN-tf"


def _detect_target_column(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    return df.columns[-1]


def _preprocess_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-hot encode categoricals and standardize features; returns X/y per split."""
    dfs = [train_df.copy(), val_df.copy(), test_df.copy()]
    target_col = _detect_target_column(dfs[0])

    y_parts = []
    X_parts = []
    for df in dfs:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        y_parts.append(df[target_col].copy())
        X_parts.append(df.drop(columns=[target_col]).copy())

    X_concat = pd.concat(X_parts, ignore_index=True)
    X_encoded = pd.get_dummies(X_concat, dummy_na=True)
    X_encoded = X_encoded.fillna(0)

    n_train = len(X_parts[0])
    n_val = len(X_parts[1])

    X_train = X_encoded.iloc[:n_train].to_numpy(dtype=np.float32)
    X_val = X_encoded.iloc[n_train : n_train + n_val].to_numpy(dtype=np.float32)
    X_test = X_encoded.iloc[n_train + n_val :].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def _y_to_numpy(y: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(y):
            # Many TaBLEau datasets already encode classes as ints.
            return y.to_numpy()
        le = LabelEncoder()
        return le.fit_transform(y.astype(str)).astype(np.int64)

    y_train = _y_to_numpy(y_parts[0])
    y_val = _y_to_numpy(y_parts[1])
    y_test = _y_to_numpy(y_parts[2])

    return X_train, y_train, X_val, y_val, X_test, y_test


def _build_masks(n_train: int, n_val: int, n_test: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = n_train + n_val + n_test
    train_mask = np.zeros(n, dtype=np.int32)
    val_mask = np.zeros(n, dtype=np.int32)
    test_mask = np.zeros(n, dtype=np.int32)

    train_mask[:n_train] = 1
    val_mask[n_train : n_train + n_val] = 1
    test_mask[n_train + n_val :] = 1
    return train_mask, val_mask, test_mask


def _onehot_labels(y_all_int: np.ndarray) -> Tuple[np.ndarray, int]:
    y_all_int = np.asarray(y_all_int).astype(np.int64)
    classes = np.unique(y_all_int)
    if classes.size <= 1:
        # Degenerate case; keep 2 classes to satisfy TF placeholders.
        num_classes = 2
    else:
        num_classes = int(classes.max()) + 1

    ys = np.eye(num_classes, dtype=np.float32)[y_all_int]

    # Ensure at least 2 columns for binary datasets that might have been encoded as {0} only.
    if ys.ndim == 1:
        ys = ys.reshape(-1, 1)
    if ys.shape[1] == 1:
        ys = np.hstack([ys, 1.0 - ys])
        num_classes = 2

    return ys, int(num_classes)


def _build_knn_adj(features: np.ndarray, k: int, metric: str = "euclidean") -> "scipy.sparse.csr_matrix":
    """Build an undirected kNN adjacency (unweighted, without self-loops)."""
    import scipy.sparse as sp

    x = np.asarray(features, dtype=np.float32)
    n = int(x.shape[0])
    if n <= 1:
        return sp.csr_matrix((n, n), dtype=np.float32)

    k_eff = int(min(max(int(k), 1), n - 1))

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=str(metric), n_jobs=1)
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]  # drop self

    row = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    col = indices.reshape(-1).astype(np.int64, copy=False)
    data = np.ones_like(row, dtype=np.float32)

    adj = sp.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)
    # Symmetrize
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.tocsr()


class _GlcnRunResult:
    def __init__(
        self,
        best_val_metric: float,
        best_test_metric: float,
        metric_name: str,
        best_epoch: int | None = None,
        stopped_epoch: int | None = None,
        elapsed_sec: float | None = None,
    ):
        self.best_val_metric = float(best_val_metric)
        self.best_test_metric = float(best_test_metric)
        self.metric_name = str(metric_name)
        self.best_epoch = None if best_epoch is None else int(best_epoch)
        self.stopped_epoch = None if stopped_epoch is None else int(stopped_epoch)
        self.elapsed_sec = elapsed_sec


def _run_glcn_subprocess(payload: Dict[str, Any]) -> _GlcnRunResult:
    python_exe = os.environ.get("GLCN_PYTHON")
    if not python_exe:
        # Convenience: many setups already have a TF1 env for LDS-GNN.
        python_exe = os.environ.get("LDS_GNN_PYTHON")
        if python_exe:
            logger.warning("GLCN_PYTHON is not set; falling back to LDS_GNN_PYTHON=%s", python_exe)
        else:
            raise ImportError(
                "GLCN requires TensorFlow 1.x. Set env var GLCN_PYTHON to a python executable "
                "in an environment where TensorFlow 1.x is importable. "
                "Tip: you can run `source scripts/glcn_env.sh` before launching TaBLEau."
            )

    with tempfile.TemporaryDirectory(prefix="glcn_") as td:
        data_path = os.path.join(td, "data.npz")
        out_path = os.path.join(td, "out.json")
        driver_path = os.path.join(td, "driver.py")

        np.savez_compressed(data_path, **payload)

        driver_code = r'''
import json
import os
import sys
import time

import numpy as np

# Limit BLAS/OMP threads to avoid OpenBLAS "too many memory regions" / segfault on large datasets.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Force CPU for stability (parent process may be using CUDA).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import scipy.sparse as sp

# Ensure TF1-style API
import tensorflow as tf


def _define_flag(definer, name, default, help_str):
    """Define a tf.app.flags flag if it's not already defined."""
    try:
        definer(name, default, help_str)
    except Exception:
        # Already defined or flags system not available; ignore.
        pass


def _auc_roc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn; returns NaN if undefined."""
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank-based AUC (Mann-Whitney U)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _metric_from_probs(y_onehot: np.ndarray, probs: np.ndarray, mask: np.ndarray, metric_name: str) -> float:
    idx = np.where(mask.astype(bool))[0]
    if idx.size == 0:
        return float("nan")

    y_true = y_onehot[idx]
    if metric_name == "AUC":
        # Binary only
        y_true_cls = np.argmax(y_true, axis=1)
        scores = probs[idx, 1]
        auc = _auc_roc_binary(y_true_cls, scores)
        if np.isnan(auc):
            # Fallback to accuracy when AUC undefined
            pred = np.argmax(probs[idx], axis=1)
            return float(np.mean(pred == y_true_cls))
        return float(auc)

    pred = np.argmax(probs[idx], axis=1)
    y_true_cls = np.argmax(y_true, axis=1)
    return float(np.mean(pred == y_true_cls))


def _preprocess_adj_sparse(adj: sp.spmatrix):
    """Normalize adjacency and produce (tuple, edge) without densifying."""
    adj = sp.coo_matrix(adj)
    # SciPy may return CSR/CSC for sparse addition; force COO for .row/.col.
    adj = (adj + sp.eye(adj.shape[0], dtype=adj.dtype, format="coo")).tocoo()

    # TF1 SparseTensor ops (used inside upstream GLCN) require lexicographically sorted indices.
    # Ensure COO entries are sorted (row-major) and merged.
    adj.sum_duplicates()
    order0 = np.lexsort((adj.col, adj.row))
    adj = sp.coo_matrix((adj.data[order0], (adj.row[order0], adj.col[order0])), shape=adj.shape)

    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    # Sort normalized adjacency indices as well.
    adj_norm.sum_duplicates()
    order1 = np.lexsort((adj_norm.col, adj_norm.row))
    adj_norm = sp.coo_matrix(
        (adj_norm.data[order1], (adj_norm.row[order1], adj_norm.col[order1])),
        shape=adj_norm.shape,
    )

    coords = np.vstack((adj_norm.row, adj_norm.col)).transpose().astype(np.int64)
    values = adj_norm.data.astype(np.float32)
    shape = adj_norm.shape

    # Edge list shape: [2, num_edges]
    edge = np.vstack([adj_norm.row, adj_norm.col]).astype(np.int64)
    return (coords, values, shape), edge


def run(data_path: str, out_path: str):
    arr = np.load(data_path)

    features_dense = arr["features"].astype(np.float32)
    task_type = str(arr.get("task_type", "binclass")).lower()
    is_classification = task_type in ["binclass", "multiclass"]
    is_regression = task_type == "regression"
    if not (is_classification or is_regression):
        raise ValueError("Unsupported task_type: %r" % task_type)

    y_onehot = None
    y_reg = None
    y_mean = None
    y_std = None
    if is_classification:
        y_onehot = arr["y_onehot"].astype(np.float32)
    else:
        y_reg = arr["y_reg"].astype(np.float32)
        y_mean = float(arr.get("y_mean", 0.0))
        y_std = float(arr.get("y_std", 1.0))

    train_mask = arr["train_mask"].astype(np.int32)
    val_mask = arr["val_mask"].astype(np.int32)
    test_mask = arr["test_mask"].astype(np.int32)

    adj_row = arr["adj_row"].astype(np.int64)
    adj_col = arr["adj_col"].astype(np.int64)
    adj_data = arr["adj_data"].astype(np.float32)
    adj_shape = tuple(arr["adj_shape"].astype(np.int64))

    seed = int(arr["seed"])
    epochs = int(arr["epochs"])
    early_stopping = int(arr["early_stopping"])

    metric_name = str(arr["metric_name"])

    # Define flags expected by upstream code (models.py reads tf.app.flags.FLAGS)
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    _define_flag(flags.DEFINE_float, "lr1", float(arr["lr1"]), "Graph learning LR")
    _define_flag(flags.DEFINE_float, "lr2", float(arr["lr2"]), "GCN LR")
    _define_flag(flags.DEFINE_integer, "epochs", epochs, "epochs")
    _define_flag(flags.DEFINE_integer, "hidden_gcn", int(arr["hidden_gcn"]), "hidden_gcn")
    _define_flag(flags.DEFINE_integer, "hidden_gl", int(arr["hidden_gl"]), "hidden_gl")
    _define_flag(flags.DEFINE_float, "dropout", float(arr["dropout"]), "dropout")
    _define_flag(flags.DEFINE_float, "weight_decay", float(arr["weight_decay"]), "weight_decay")
    _define_flag(flags.DEFINE_integer, "early_stopping", early_stopping, "early_stopping")
    _define_flag(flags.DEFINE_float, "losslr1", float(arr["losslr1"]), "losslr1")
    _define_flag(flags.DEFINE_float, "losslr2", float(arr["losslr2"]), "losslr2")
    _define_flag(flags.DEFINE_integer, "seed", seed, "seed")

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Put upstream repo on sys.path
    glcn_repo = os.environ.get("GLCN_TF_REPO", "")
    if glcn_repo and glcn_repo not in sys.path:
        sys.path.insert(0, glcn_repo)
    glcn_pkg = os.path.join(glcn_repo, "glcn")
    if glcn_pkg and glcn_pkg not in sys.path:
        sys.path.insert(0, glcn_pkg)

    from utils import preprocess_features, sparse_to_tuple, construct_feed_dict  # noqa
    from models import SGLCN  # noqa
    from layers import SparseGraphLearn, GraphConvolution  # noqa

    # --- Scalability hotfix for large N ---
    # Upstream GLCN loss uses tf.matrix_diag(...) and tf.sparse_tensor_to_dense(S),
    # which densifies an NxN matrix and will OOM / segfault on large datasets (e.g., helena).
    # Replace graph-regularizer terms with sparse-safe equivalents:
    # - trace(S^T S) == sum(S.values^2)
    # - trace(X^T (D - S) X) == sum(X * ((D - S)X)) with D = diag(sum_j S_ij)
    try:
        import models as _glcn_models  # noqa

        def _sglcn_loss_sparse(self):
            self.loss1 = 0.0
            self.loss2 = 0.0

            # Weight decay
            for var in self.layers0.vars.values():
                self.loss1 += FLAGS.weight_decay * tf.nn.l2_loss(var)
            for var in self.layers1.vars.values():
                self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)

            # Graph Learning loss (sparse-safe)
            deg = tf.sparse_reduce_sum(self.S, axis=1)  # [N]
            Dx = tf.expand_dims(deg, 1) * self.x
            Sx = tf.sparse_tensor_dense_matmul(self.S, self.x)
            Lx = Dx - Sx
            self.loss1 += tf.reduce_sum(self.x * Lx) * FLAGS.losslr1
            self.loss1 -= tf.reduce_sum(tf.square(self.S.values)) * FLAGS.losslr2

            # Cross entropy error
            self.loss2 += _glcn_models.masked_softmax_cross_entropy(
                self.outputs, self.placeholders["labels"], self.placeholders["labels_mask"]
            )
            self.loss = self.loss1 + self.loss2

        _glcn_models.SGLCN._loss = _sglcn_loss_sparse
    except Exception:
        pass

    def masked_mse(preds, labels, mask):
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.maximum(tf.reduce_mean(mask), 1e-12)
        mask = tf.expand_dims(mask, axis=1)
        return tf.reduce_mean(tf.square(preds - labels) * mask)

    class SGLCNReg(object):
        def __init__(self, placeholders, edge, input_dim, **kwargs):
            self.inputs = placeholders["features"]
            self.edge = edge
            self.input_dim = int(input_dim)
            self.output_dim = 1
            self.placeholders = placeholders
            self.loss1 = 0.0
            self.loss2 = 0.0

            learning_rate1 = tf.train.exponential_decay(
                learning_rate=FLAGS.lr1,
                global_step=placeholders["step"],
                decay_steps=100,
                decay_rate=0.9,
                staircase=True,
            )
            learning_rate2 = tf.train.exponential_decay(
                learning_rate=FLAGS.lr2,
                global_step=placeholders["step"],
                decay_steps=100,
                decay_rate=0.9,
                staircase=True,
            )
            self.optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate1)
            self.optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)

            self.layers0 = SparseGraphLearn(
                input_dim=self.input_dim,
                output_dim=FLAGS.hidden_gl,
                edge=self.edge,
                placeholders=self.placeholders,
                act=tf.nn.relu,
                dropout=True,
                sparse_inputs=True,
            )
            self.layers1 = GraphConvolution(
                input_dim=self.input_dim,
                output_dim=FLAGS.hidden_gcn,
                placeholders=self.placeholders,
                act=tf.nn.relu,
                dropout=True,
                sparse_inputs=True,
                logging=False,
            )
            self.layers2 = GraphConvolution(
                input_dim=FLAGS.hidden_gcn,
                output_dim=self.output_dim,
                placeholders=self.placeholders,
                act=lambda x: x,
                dropout=True,
                logging=False,
            )
            self.build()

        def _loss(self):
            # Weight decay
            for var in self.layers0.vars.values():
                self.loss1 += FLAGS.weight_decay * tf.nn.l2_loss(var)
            for var in self.layers1.vars.values():
                self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)

            # Graph learning loss (sparse-safe, same objective as upstream)
            deg = tf.sparse_reduce_sum(self.S, axis=1)  # [N]
            Dx = tf.expand_dims(deg, 1) * self.x
            Sx = tf.sparse_tensor_dense_matmul(self.S, self.x)
            Lx = Dx - Sx
            self.loss1 += tf.reduce_sum(self.x * Lx) * FLAGS.losslr1
            self.loss1 -= tf.reduce_sum(tf.square(self.S.values)) * FLAGS.losslr2

            # Regression loss (masked MSE)
            self.loss2 += masked_mse(self.outputs, self.placeholders["labels"], self.placeholders["labels_mask"])
            self.loss = self.loss1 + self.loss2

        def build(self):
            self.x, self.S = self.layers0(self.inputs)
            x1 = self.layers1(self.inputs, self.S)
            self.outputs = self.layers2(x1, self.S)

            # Variable split (matches upstream assumption)
            self.vars1 = tf.trainable_variables()[0:2]
            self.vars2 = tf.trainable_variables()[2:]

            self._loss()
            self.opt_op1 = self.optimizer1.minimize(self.loss1, var_list=self.vars1)
            self.opt_op2 = self.optimizer2.minimize(self.loss2, var_list=self.vars2)
            self.opt_op = tf.group(self.opt_op1, self.opt_op2)

    # --- Hotfix for TF1 SparseTensor ordering ---
    # The upstream implementation builds a SparseTensor for the learned graph `S`.
    # Depending on the edge list order, TF1 ops like SparseToDense can error with:
    #   "indices[...] is out of order"
    # Reordering fixes this and is a no-op for already-sorted indices.
    try:
        import layers as _glcn_layers  # noqa

        _old_call = _glcn_layers.SparseGraphLearn.__call__

        def _reordered_call(self, inputs):
            h, sgraph = _old_call(self, inputs)
            try:
                sgraph = tf.sparse_reorder(sgraph)
            except Exception:
                try:
                    sgraph = tf.sparse.reorder(sgraph)
                except Exception:
                    pass
            return h, sgraph

        _glcn_layers.SparseGraphLearn.__call__ = _reordered_call
    except Exception:
        pass

    # Build sparse features
    X_sp = sp.csr_matrix(features_dense)
    features = preprocess_features(X_sp)

    # Build sparse adjacency
    adj = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=adj_shape, dtype=np.float32)
    adj = adj.maximum(adj.T)
    adj, edge = _preprocess_adj_sparse(adj)

    # Build split label matrices (zeros for non-masked rows)
    if is_classification:
        y_train = np.zeros_like(y_onehot)
        y_val = np.zeros_like(y_onehot)
        y_test = np.zeros_like(y_onehot)
        y_train[train_mask.astype(bool)] = y_onehot[train_mask.astype(bool)]
        y_val[val_mask.astype(bool)] = y_onehot[val_mask.astype(bool)]
        y_test[test_mask.astype(bool)] = y_onehot[test_mask.astype(bool)]
        label_dim = int(y_onehot.shape[1])
        y_true_full = y_onehot
    else:
        # Train on standardized y, but evaluate RMSE in original units.
        y_scaled = (y_reg - float(y_mean)) / float(y_std)
        y_train = np.zeros_like(y_scaled)
        y_val = np.zeros_like(y_scaled)
        y_test = np.zeros_like(y_scaled)
        y_train[train_mask.astype(bool)] = y_scaled[train_mask.astype(bool)]
        y_val[val_mask.astype(bool)] = y_scaled[val_mask.astype(bool)]
        y_test[test_mask.astype(bool)] = y_scaled[test_mask.astype(bool)]
        label_dim = 1
        y_true_full = y_reg

    tf.reset_default_graph()

    placeholders = {
        "adj": tf.sparse_placeholder(tf.float32),
        "features": tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        "labels": tf.placeholder(tf.float32, shape=(None, label_dim)),
        "labels_mask": tf.placeholder(tf.int32),
        "dropout": tf.placeholder_with_default(0.0, shape=()),
        "num_nodes": tf.placeholder(tf.int32),
        "step": tf.placeholder(tf.int32),
        "num_features_nonzero": tf.placeholder(tf.int32),
    }

    if is_classification:
        model = SGLCN(placeholders, edge, input_dim=features[2][1], logging=False)
    else:
        model = SGLCNReg(placeholders, edge, input_dim=features[2][1])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    def _eval_probs(labels_mat, mask_vec, epoch):
        feed = construct_feed_dict(features, adj, labels_mat, mask_vec, epoch, placeholders)
        feed.update({placeholders["dropout"]: 0.0})
        probs = sess.run(model.pro, feed_dict=feed)
        loss_val, acc_val = sess.run([model.loss, model.accuracy], feed_dict=feed)
        return probs, float(loss_val), float(acc_val)

    def _eval_reg(labels_mat, mask_vec, epoch):
        feed = construct_feed_dict(features, adj, labels_mat, mask_vec, epoch, placeholders)
        feed.update({placeholders["dropout"]: 0.0})
        preds_scaled = sess.run(model.outputs, feed_dict=feed).reshape(-1)
        loss_val = sess.run(model.loss, feed_dict=feed)
        preds = preds_scaled * float(y_std) + float(y_mean)
        idx = np.where(mask_vec.astype(bool))[0]
        if idx.size == 0:
            return float("nan"), float(loss_val)
        y_true = y_true_full[idx].reshape(-1)
        y_pred = preds[idx]
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return rmse, float(loss_val)

    start = time.time()
    sess.run(tf.global_variables_initializer())

    # For classification: select best by val_loss (stable).
    # For regression: select best by val_rmse (what TaBLEau reports), and use that for patience.
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_epoch = 0
    best_val_metric = float("nan")
    best_test_metric = float("nan")

    patience = 0
    stopped_epoch = 0

    for epoch in range(epochs):
        # Train
        feed = construct_feed_dict(features, adj, y_train, train_mask, epoch, placeholders)
        feed.update({placeholders["dropout"]: float(arr["dropout"])})
        _ = sess.run(model.opt_op, feed_dict=feed)

        # Validate and test (dropout disabled)
        if is_classification:
            probs_val, val_loss, _val_acc = _eval_probs(y_val, val_mask, epoch)
            probs_test, _test_loss, _test_acc = _eval_probs(y_test, test_mask, epoch)

            val_metric = _metric_from_probs(y_onehot, probs_val, val_mask, metric_name)
            test_metric = _metric_from_probs(y_onehot, probs_test, test_mask, metric_name)
        else:
            val_metric, val_loss = _eval_reg(y_val, val_mask, epoch)
            test_metric, _ = _eval_reg(y_test, test_mask, epoch)

        if is_classification:
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = val_loss
                best_epoch = int(epoch)
                best_val_metric = float(val_metric)
                best_test_metric = float(test_metric)
                patience = 0
            else:
                patience += 1
        else:
            # RMSE: smaller is better
            if float(val_metric) < best_val_rmse - 1e-12:
                best_val_rmse = float(val_metric)
                best_epoch = int(epoch)
                best_val_metric = float(val_metric)
                best_test_metric = float(test_metric)
                patience = 0
            else:
                patience += 1

        stopped_epoch = int(epoch)
        if patience >= early_stopping:
            break

    elapsed = time.time() - start
    out = {
        "metric_name": metric_name,
        "best_val_metric": float(best_val_metric),
        "best_test_metric": float(best_test_metric),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "elapsed_sec": float(elapsed),
    }

    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
'''

        with open(driver_path, "w") as f:
            f.write(driver_code)

        env = os.environ.copy()
        env.setdefault("GLCN_TF_REPO", GLCN_TF_REPO)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

        cmd = [python_exe, driver_path, data_path, out_path]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "GLCN subprocess failed. "
                f"returncode={proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n\n"
                f"stderr:\n{proc.stderr}\n"
            )

        with open(out_path, "r") as f:
            out = json.load(f)

        return _GlcnRunResult(
            best_val_metric=float(out.get("best_val_metric")),
            best_test_metric=float(out.get("best_test_metric")),
            metric_name=str(out.get("metric_name", "unknown")),
            best_epoch=out.get("best_epoch"),
            stopped_epoch=out.get("stopped_epoch"),
            elapsed_sec=out.get("elapsed_sec"),
        )


def main(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_results: Dict[str, Any],
    config: Dict[str, Any],
    gnn_stage: str,
) -> Dict[str, Any]:
    """Entry point used by TaBLEau ModelRunner."""
    start = time.time()

    task_type = str(dataset_results.get("info", {}).get("task_type", "binclass")).lower()
    is_classification = task_type in ["binclass", "multiclass"]
    is_regression = task_type == "regression"
    if not (is_classification or is_regression):
        raise ValueError(f"Unsupported task_type={task_type!r} for GLCN wrapper")

    X_train, y_train, X_val, y_val, X_test, y_test = _preprocess_splits(train_df, val_df, test_df)

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    train_mask, val_mask, test_mask = _build_masks(n_train, n_val, n_test)

    features = np.concatenate([X_train, X_val, X_test], axis=0).astype(np.float32)

    metric_name = "RMSE" if is_regression else "Acc"
    num_classes = None
    y_onehot = None
    y_reg = None
    y_mean = None
    y_std = None

    if is_classification:
        y_all = np.concatenate([y_train, y_val, y_test], axis=0)
        y_all_int = y_all.astype(np.int64)
        y_onehot, num_classes = _onehot_labels(y_all_int)
        metric_name = "AUC" if int(num_classes) == 2 else "Acc"
    else:
        # Regression: train on standardized y (train split stats), report RMSE in original units.
        y_all = np.concatenate([y_train, y_val, y_test], axis=0).astype(np.float32).reshape(-1, 1)
        tr = np.zeros((y_all.shape[0],), dtype=bool)
        tr[: len(y_train)] = True
        y_mean = float(np.mean(y_all[tr]))
        y_std = float(np.std(y_all[tr]) + 1e-12)
        y_reg = y_all

    # Build initial kNN adjacency: this defines the candidate edge set for GLCN graph learning.
    knn_k = int(config.get("glcn_k", config.get("knn_k", 10)))
    knn_metric = str(config.get("glcn_knn_metric", "euclidean"))

    adj = _build_knn_adj(features, k=knn_k, metric=knn_metric)
    adj_coo = adj.tocoo()

    # LDS-style note: gnn_stage is ignored because this is self-contained.
    _ = gnn_stage

    payload: Dict[str, Any] = {
        "task_type": task_type,
        "features": features,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "adj_row": adj_coo.row.astype(np.int64),
        "adj_col": adj_coo.col.astype(np.int64),
        "adj_data": adj_coo.data.astype(np.float32),
        "adj_shape": np.asarray(adj_coo.shape, dtype=np.int64),
        "seed": int(config.get("seed", 1)),
        "epochs": int(config.get("epochs", 2000)),
        "early_stopping": int(config.get("patience", 100)),
        "hidden_gcn": int(config.get("glcn_hidden_gcn", 30)),
        "hidden_gl": int(config.get("glcn_hidden_gl", 70)),
        "dropout": float(config.get("glcn_dropout", 0.6)),
        "weight_decay": float(config.get("glcn_weight_decay", 1e-4)),
        "lr1": float(config.get("glcn_lr1", 5e-3)),
        "lr2": float(config.get("glcn_lr2", 5e-3)),
        "losslr1": float(config.get("glcn_losslr1", 1e-2)),
        "losslr2": float(config.get("glcn_losslr2", 1e-4)),
        "metric_name": metric_name,
    }

    if is_classification:
        assert y_onehot is not None
        payload["y_onehot"] = y_onehot
        payload["num_classes"] = int(num_classes) if num_classes is not None else 0
    else:
        assert y_reg is not None
        payload["y_reg"] = y_reg
        payload["y_mean"] = float(y_mean)
        payload["y_std"] = float(y_std)

    result = _run_glcn_subprocess(payload)

    elapsed = time.time() - start

    # Regression sanity check: naive mean predictor RMSE (to interpret absolute RMSE scale)
    notes_extra: Dict[str, Any] = {}
    if is_regression:
        y_train_f = np.asarray(y_train, dtype=np.float32)
        y_val_f = np.asarray(y_val, dtype=np.float32)
        y_test_f = np.asarray(y_test, dtype=np.float32)
        y_mean_tr = float(np.mean(y_train_f))
        notes_extra.update(
            {
                "naive_mean_rmse_val": float(np.sqrt(np.mean((y_val_f - y_mean_tr) ** 2))),
                "naive_mean_rmse_test": float(np.sqrt(np.mean((y_test_f - y_mean_tr) ** 2))),
                "target_mean_train": y_mean_tr,
                "target_std_train": float(np.std(y_train_f) + 1e-12),
            }
        )

    return {
        "best_val_metric": float(result.best_val_metric),
        "best_test_metric": float(result.best_test_metric),
        "metric_name": str(result.metric_name),
        "glcn_best_epoch": result.best_epoch,
        "glcn_stopped_epoch": result.stopped_epoch,
        "glcn_elapsed_sec": result.elapsed_sec,
        "elapsed_time": float(elapsed),
        "stage_mapping": {
            "pytorch_frame_equivalent": ["encoding", "columnwise", "decoding"],
            "primary": "columnwise",
            "notes": "Graph learning + message passing over a row-graph; not a stage-injection model.",
        },
        "notes": {
            "gnn_stage_ignored": True,
            "env_required": "Set GLCN_PYTHON to a TF1-compatible interpreter; optionally set GLCN_TF_REPO.",
            "glcn_repo": GLCN_TF_REPO,
            **notes_extra,
        },
    }

#  small+binclass
#  source scripts/glcn_env.sh && conda run -n tableau python main.py --dataset kaggle_Audit_Data --models glcn --gnn_stages all --epochs 2
#  small+regression
#  source scripts/glcn_env.sh && conda run -n tableau python main.py --dataset openml_The_Office_Dataset --models glcn --gnn_stages all --epochs 2
#  large+binclsas
#  source scripts/glcn_env.sh && conda run -n tableau python main.py --dataset credit --models glcn --gnn_stages all --epochs 2
#  large+multiclass
#  source scripts/glcn_env.sh && conda run -n tableau python main.py --dataset eye --models glcn --gnn_stages all --epochs 2
#  source scripts/glcn_env.sh && conda run -n tableau python main.py --dataset helena --models glcn --gnn_stages all --epochs 2
#  large+regression
#  source scripts/glcn_env.sh && conda run -n tableau python main.py --dataset house --models glcn --gnn_stages all --epochs 2

