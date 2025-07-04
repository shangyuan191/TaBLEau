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



class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def knn_graph(x, k):
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_index = []
    N = x_np.shape[0]
    for i in range(N):
        for j in indices[i][1:]:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    print("Executing GNN between start_fn and materialize_fn")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    # 合併三個df
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    print(f"all_df.head():\n{all_df.head()}")
    feature_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
    y = all_df['target'].values

    # 自動計算 num_classes
    if task_type in ['binclass', 'multiclass']:
        num_classes = len(pd.unique(y))
        print(f"Detected num_classes: {num_classes}")
    else:
        num_classes = 1
    # label 處理
    if task_type == 'binclass':
        y = torch.tensor(y, dtype=torch.float32, device=device)
    elif task_type == 'multiclass':
        y = torch.tensor(y, dtype=torch.long, device=device)
    else:  # regression
        y = torch.tensor(y, dtype=torch.float32, device=device)

    # 建圖
    edge_index = knn_graph(x, k).to(device)
    in_dim = x.shape[1]
    out_dim = in_dim
    # mask
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    N = n_train + n_val + n_test
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True

    patience = config.get('gnn_patience', 10)
    best_loss = float('inf')
    early_stop_counter = 0
    # 建立並訓練GNN
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    gnn.train()
    gnn_early_stop_epochs = 0
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(x, edge_index)
        loss = torch.nn.functional.mse_loss(out, x)
        loss.backward()
        optimizer.step()
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if (epoch+1) % 10 == 0:
            print(f'GNN Epoch {epoch+1}/{gnn_epochs}, Loss: {loss.item():.4f}')
        if early_stop_counter >= patience:
            print(f"GNN Early stopping at epoch {epoch+1}")
            gnn_early_stop_epochs = epoch + 1
            break
    gnn.eval()
    with torch.no_grad():
        final_emb = gnn(x, edge_index).cpu().numpy()
    # print(f"Final embedding shape: {final_emb.shape}")
    # print(f"final embedding type: {type(final_emb)}")
    # print(f"final embedding head:\n{final_emb[:5]}")
    # 將final_emb分回三個df
    train_emb = final_emb[:n_train]
    val_emb = final_emb[n_train:n_train+n_val]
    test_emb = final_emb[n_train+n_val:]

    emb_cols = [f'N_feature_{i}' for i in range(1,out_dim+1)]
    train_df_gnn = pd.DataFrame(train_emb, columns=emb_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(val_emb, columns=emb_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(test_emb, columns=emb_cols, index=test_df.index)
    # print(f"train_df_gnn.shape: {train_df_gnn.shape}")
    # print(f"val_df_gnn.shape: {val_df_gnn.shape}")
    # print(f"test_df_gnn.shape: {test_df_gnn.shape}")
    # print(f"train_df_gnn.head():\n{train_df_gnn.head()}")
    # print(f"val_df_gnn.head():\n{val_df_gnn.head()}")
    # print(f"test_df_gnn.head():\n{test_df_gnn.head()}")

    # 保留原標籤
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values
    # print(f"train_df_gnn.shape: {train_df_gnn.shape}")
    # print(f"val_df_gnn.shape: {val_df_gnn.shape}")
    # print(f"test_df_gnn.shape: {test_df_gnn.shape}")
    # print(f"train_df.head():\n{train_df.head()}")
    # print(f"train_df_gnn.head():\n{train_df_gnn.head()}")
    # print(f"val_df.head():\n{val_df.head()}")
    # print(f"val_df_gnn.head():\n{val_df_gnn.head()}")
    # print(f"test_df.head():\n{test_df.head()}")
    # print(f"test_df_gnn.head():\n{test_df_gnn.head()}")
    

    
    
    

    # 若需要將 num_classes 傳遞到下游，可 return
    return train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs







def gnn_after_materialize_fn(train_tensor_frame, val_tensor_frame, test_tensor_frame, config, dataset_name, task_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)


    # print(f"train_tensor_frame.shape: {train_tensor_frame.feat_dict['numerical'].shape}")
    # print(f"val_tensor_frame.shape: {val_tensor_frame.feat_dict['numerical'].shape}")
    # print(f"test_tensor_frame.shape: {test_tensor_frame.feat_dict['numerical'].shape}")
    # print(f"train_tensor_frame.col_names_dict: {train_tensor_frame.col_names_dict}")
    # print(f"val_tensor_frame.col_names_dict: {val_tensor_frame.col_names_dict}")
    # print(f"test_tensor_frame.col_names_dict: {test_tensor_frame.col_names_dict}")
    # print(f"train_tensor_frame.feat_dict: {train_tensor_frame.feat_dict}")
    # print(f"val_tensor_frame.feat_dict: {val_tensor_frame.feat_dict}")
    # print(f"test_tensor_frame.feat_dict: {test_tensor_frame.feat_dict}")
    # print(f"train_tensor_frame.y: {train_tensor_frame.y}")
    # print(f"val_tensor_frame.y: {val_tensor_frame.y}")
    # print(f"test_tensor_frame.y: {test_tensor_frame.y}")
    # print(f"train_tensor_frame: {train_tensor_frame}")
    # print(f"val_tensor_frame: {val_tensor_frame}")
    # print(f"test_tensor_frame: {test_tensor_frame}")
    # print(f"type of train_tensor_frame.col_names_dict: {type(train_tensor_frame.col_names_dict)}")
    # for key in train_tensor_frame.col_names_dict:
    #     print(f"key: {key}, value: {train_tensor_frame.col_names_dict[key]}")

    # print(f"type of train_tensor_frame.feat_dict: {type(train_tensor_frame.feat_dict)}")
    # for key in train_tensor_frame.feat_dict:
    #     print(f"key: {key}, value shape: {train_tensor_frame.feat_dict[key].shape}")
    # print(f"type of train_tensor_frame.y: {type(train_tensor_frame.y)}")

    # print(train_tensor_frame.col_names_dict.keys())
    # print(val_tensor_frame.col_names_dict.keys())
    def tensor_frame_to_df(tensor_frame):
        # 取得 feature 名稱與 tensor
        col_names = tensor_frame.col_names_dict[stype.numerical]
        features = tensor_frame.feat_dict[stype.numerical]
        df = pd.DataFrame(features, columns=col_names)
        # 加入 target
        if hasattr(tensor_frame, 'y') and tensor_frame.y is not None:
            df['target'] = tensor_frame.y.cpu().numpy()
        return df
    
    
    train_df = tensor_frame_to_df(train_tensor_frame)
    val_df = tensor_frame_to_df(val_tensor_frame)
    test_df = tensor_frame_to_df(test_tensor_frame)

    # 合併三個df
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    # print(f"all_df.head():\n{all_df.head()}")
    feature_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
    y = all_df['target'].values

    # 自動計算 num_classes
    if task_type in ['binclass', 'multiclass']:
        num_classes = len(pd.unique(y))
        print(f"Detected num_classes: {num_classes}")
    else:
        num_classes = 1
# label 處理
    if task_type == 'binclass':
        y = torch.tensor(y, dtype=torch.float32, device=device)
    elif task_type == 'multiclass':
        y = torch.tensor(y, dtype=torch.long, device=device)
    else:  # regression
        y = torch.tensor(y, dtype=torch.float32, device=device)

    # 建圖
    edge_index = knn_graph(x, k).to(device)
    in_dim = x.shape[1]
    out_dim = in_dim
    # mask
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    N = n_train + n_val + n_test
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    patience = config.get('gnn_patience', 10)
    best_loss = float('inf')
    early_stop_counter = 0

    # 建立並訓練GNN
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    gnn_early_stop_epochs = 0
    gnn.train()
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(x, edge_index)
        loss = torch.nn.functional.mse_loss(out, x)
        loss.backward()
        optimizer.step()
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if (epoch+1) % 10 == 0:
            print(f'GNN Epoch {epoch+1}/{gnn_epochs}, Loss: {loss.item():.4f}')
        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"GNN Early stopping at epoch {epoch+1}")
            break
    gnn.eval()
    with torch.no_grad():
        final_emb = gnn(x, edge_index).cpu().numpy()
    # 將final_emb分回三個df
    train_emb = final_emb[:n_train]
    val_emb = final_emb[n_train:n_train+n_val]
    test_emb = final_emb[n_train+n_val:]

    emb_cols = [f'N_feature_{i}' for i in range(1,out_dim+1)]
    train_df_gnn = pd.DataFrame(train_emb, columns=emb_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(val_emb, columns=emb_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(test_emb, columns=emb_cols, index=test_df.index)

    # 保留原標籤
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values

    # 7. Yandex 數據集包裝
    dataset = Yandex(train_df_gnn, val_df_gnn, test_df_gnn, name=dataset_name, task_type=task_type)
    dataset.materialize()

    # 8. split tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    # 9. 類別特徵轉換
    categorical_transform = CatToNumTransform()
    categorical_transform.fit(train_tensor_frame, dataset.col_stats)
    train_tensor_frame = categorical_transform(train_tensor_frame)
    val_tensor_frame = categorical_transform(val_tensor_frame)
    test_tensor_frame = categorical_transform(test_tensor_frame)
    col_stats = categorical_transform.transformed_stats

    # 10. 基於互信息的特徵排序
    mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
    mutual_info_sort.fit(train_tensor_frame, col_stats)
    train_tensor_frame = mutual_info_sort(train_tensor_frame)
    val_tensor_frame = mutual_info_sort(val_tensor_frame)
    test_tensor_frame = mutual_info_sort(test_tensor_frame)

    # 11. 創建數據加載器
    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    return (train_loader, val_loader, test_loader,
            col_stats, mutual_info_sort,
            dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs)

# 取得所有 row 的 embedding
def get_all_embeddings_and_targets(loader, encoder, convs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_embeds, all_targets = [], []
    for tf in loader:
        tf = tf.to(device)
        x, _ = encoder(tf)
        for conv in convs:
            x = conv(x)
        x_pooled = x.mean(dim=2)  # (batch_size, num_cols)
        all_embeds.append(x_pooled.detach().cpu())
        all_targets.append(tf.y.detach().cpu())
    return torch.cat(all_embeds, dim=0), torch.cat(all_targets, dim=0)
def gnn_decoding_eval(train_loader, val_loader, test_loader, config, task_type, metric_computer,encoder, convs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_emb, train_y = get_all_embeddings_and_targets(train_loader, encoder, convs)
    val_emb, val_y = get_all_embeddings_and_targets(val_loader, encoder, convs)
    test_emb, test_y = get_all_embeddings_and_targets(test_loader, encoder, convs)
    # 合併
    all_emb = torch.cat([train_emb, val_emb, test_emb], dim=0)  # (total_rows, num_cols)
    all_y = torch.cat([train_y, val_y, test_y], dim=0)
    # print(f"all_emb shape: {all_emb.shape}, all_y shape: {all_y.shape}")
    # 合併成 DataFrame
    all_df = pd.DataFrame(all_emb.numpy())
    all_df['target'] = all_y.numpy()
    # print(f"all_df shape: {all_df.shape}")
    # print(f"all_df head:\n{all_df.head()}")
    # print(f"all_df columns: {all_df.columns}")
    feature_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
    y = all_df['target'].values
    k=5
            # 自動計算 num_classes
    if task_type in ['binclass', 'multiclass']:
        num_classes = len(pd.unique(y))
        print(f"Detected num_classes: {num_classes}")
    else:
        num_classes = 1
    # label 處理
    if task_type == 'binclass':
        y = torch.tensor(y, dtype=torch.float32, device=device)
    elif task_type == 'multiclass':
        y = torch.tensor(y, dtype=torch.long, device=device)
    else:  # regression
        y = torch.tensor(y, dtype=torch.float32, device=device)
    # 建圖
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    edge_index = knn_graph(x, k).to(device)
    # mask
    n_train = len(train_emb)
    n_val = len(val_emb)
    n_test = len(test_emb)
    print(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")
    N = n_train + n_val + n_test
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    hidden_dim = config.get('hidden_dim', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    in_dim = x.shape[1]
    out_dim = 1 if (task_type == 'regression' or task_type=="binclass") else num_classes
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    gnn.train()
    best_val_metric = None
    best_test_metric = None
    best_val = -float('inf') if (task_type=="binclass" or task_type=="multiclass") else float('inf')
    is_binary_class = task_type == 'binclass'
    is_classification = task_type in ['binclass', 'multiclass']
    metric_computer = metric_computer.to(device)
    best_epoch = 0
    early_stop_counter = 0
    patience = config.get('gnn_patience', 10)
    early_stop_epoch = 0
    best_val_metric = -float('inf') if is_classification else float('inf')
    for epoch in tqdm(range(gnn_epochs),desc="GNN Training"):
        optimizer.zero_grad()
        out = gnn(x, edge_index)
        if task_type == 'binclass':
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out[train_mask][:, 0], y[train_mask])
        elif task_type == 'multiclass':
            loss = torch.nn.functional.cross_entropy(
                out[train_mask], y[train_mask])
        else:
            loss = torch.nn.functional.mse_loss(
                out[train_mask][:, 0], y[train_mask])
        loss.backward()
        optimizer.step()
        gnn.eval()
        with torch.no_grad():
            out_val = gnn(x, edge_index)
            val_idx = torch.arange(n_train, n_train+n_val)
            if is_binary_class:
                val_metric = metric_computer(out_val[val_idx, 0], y[val_idx])
            elif is_classification:
                pred_class = out_val[val_idx].argmax(dim=-1)
                val_metric = metric_computer(pred_class, y[val_idx])
            else:
                val_metric = metric_computer(out_val[val_idx].view(-1), y[val_idx].view(-1))
            val_metric = val_metric.item() if hasattr(val_metric, 'item') else float(val_metric)
            improved = (val_metric > best_val_metric) if is_classification else (val_metric < best_val_metric)
            print(f"epoch {epoch+1}/{gnn_epochs}, val_metric: {val_metric:.4f}, best_val_metric: {best_val_metric:.4f}, improved: {improved}, task_type: {task_type}")
            if improved:
                best_val_metric = val_metric
                best_epoch = epoch + 1
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            # test_metric 仍用 test set
            test_idx = torch.arange(n_train+n_val, N)
            if is_binary_class:
                test_metric = metric_computer(out_val[test_idx, 0], y[test_idx])
            elif is_classification:
                pred_class = out_val[test_idx].argmax(dim=-1)
                test_metric = metric_computer(pred_class, y[test_idx])
            else:
                test_metric = metric_computer(out_val[test_idx].view(-1), y[test_idx].view(-1))
            test_metric = test_metric.item() if hasattr(test_metric, 'item') else float(test_metric)
        if early_stop_counter >= patience:
            early_stop_epoch = epoch + 1
            print(f"GNN Early stopping at epoch {early_stop_epoch}")
            break
    return best_val_metric, test_metric, early_stop_epoch
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    # 新增：引入必要的GNN模組（以PyG為例）
    from torch_geometric.nn import GCNConv
    import torch.nn as nn
    class SimpleGCN_INTERNAL(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, out_channels)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            return x
    gnn = SimpleGCN_INTERNAL(channels, channels).to(device)
    
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
    
    # 創建解碼器ExcelFormerDecoder
    decoder = ExcelFormerDecoder(
        channels, 
        out_channels, 
        train_tensor_frame.num_cols
    ).to(device)
    
    # 模擬完整的ExcelFormer前向傳播
    def model_forward(tf, mixup_encoded=False):
        # 編碼階段
        x, _ = encoder(tf)  # x: [batch, num_cols, channels]
        batch_size, num_cols, channels_ = x.shape
        # 判斷是否在encoding後插入GNN
        if gnn_stage == 'encoding':
            x_reshape = x.view(-1, channels_)
            row = torch.arange(num_cols).repeat(num_cols, 1).view(-1)
            col = torch.arange(num_cols).repeat(1, num_cols).view(-1)
            edge_index_single = torch.stack([row, col], dim=0)
            edge_index = []
            for i in range(batch_size):
                offset = i * num_cols
                edge_index.append(edge_index_single + offset)
            edge_index = torch.cat(edge_index, dim=1).to(x.device)
            x = gnn(x_reshape, edge_index).view(batch_size, num_cols, channels_)
        # 如需mixup，則在GNN後應用
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
        # 列間交互階段
        for conv in convs:
            x = conv(x)
        # 判斷是否在conv與decoder之間插入GNN
        if gnn_stage == 'columnwise':
            x_reshape = x.view(-1, channels_)
            row = torch.arange(num_cols).repeat(num_cols, 1).view(-1)
            col = torch.arange(num_cols).repeat(1, num_cols).view(-1)
            edge_index_single = torch.stack([row, col], dim=0)
            edge_index = []
            for i in range(batch_size):
                offset = i * num_cols
                edge_index.append(edge_index_single + offset)
            edge_index = torch.cat(edge_index, dim=1).to(x.device)
            x = gnn(x_reshape, edge_index).view(batch_size, num_cols, channels_)
        # 解碼階段
        out = decoder(x)
        return out, y_mixedup  # 始終返回兩個值
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.001)
    gamma = config.get('gamma', 0.95)
    
    # 收集所有參數
    all_params = list(encoder.parameters()) + list(gnn.parameters()) + [p for conv in convs for p in conv.parameters()] + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        encoder.train()
        for conv in convs:
            conv.train()
        decoder.train()
        
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
        decoder.eval()
        
        metric_computer.reset()
        
        for tf in loader:
            tf = tf.to(device)
            pred, _ = model_forward(tf)
            
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
    best_epoch = 0
    early_stop_counter = 0
    train_losses = []
    train_metrics = []
    val_metrics = []
    # test_metrics = []  # 不再每個epoch計算test

    epochs = config.get('epochs', 200)
    early_stop_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        train_metric = test(train_loader)
        val_metric = test(val_loader)
        # test_metric = test(test_loader)  # 不再每個epoch計算test

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

    # 決定最終 metric 輸出
    final_metric = None
    test_metric = None
    if gnn_stage == 'decoding':
        # 確保gnn在gnn_decoding_eval作用域可見
        best_val_metric, test_metric, gnn_early_stop_epochs = gnn_decoding_eval(train_loader, val_loader, test_loader, config, task_type, metric_computer, encoder, convs)
        final_metric = best_val_metric
    else:
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
        'encoder': encoder,
        'convs': convs,
        'decoder': decoder,
        'model_forward': model_forward,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'metric_computer': metric_computer,
        'metric': metric,
        'early_stop_epochs': early_stop_epochs,
        'gnn_early_stop_epochs': 0 if gnn_stage != 'decoding' else gnn_early_stop_epochs,
    }





def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    print("ExcelFormer - 五階段執行")
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    # df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    # class_count_dict = df['target'].value_counts().to_dict()
    # # 按 key 排序
    # class_count_dict_sorted = dict(sorted(class_count_dict.items()))
    # print("各類別數量統計（dict，已排序）：", class_count_dict_sorted)

    # print("總共有幾個類別：", df['target'].nunique())
    # 獲取配置參數
    try:
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        
        if gnn_stage=='start':
            # 在 start_fn 和 materialize_fn 之間插入 GNN
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
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