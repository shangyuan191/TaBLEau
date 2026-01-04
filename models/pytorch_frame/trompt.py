from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleList, Parameter, ReLU, Sequential

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TromptConv
from torch_frame.nn.decoder import TromptDecoder
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame.typing import NAStrategy



"""Reported (reproduced) results of of Trompt model based on Tables 9--20 of
the original paper: https://arxiv.org/abs/2305.18446.

electricity (A4): 84.50 (84.17)
eye_movements (A5): 64.25 (63.02)
MagicTelescope (B2): 86.30 (86.93)
bank-marketing (B4): 79.36 (80.59)
california (B5): 89.09 (89.17)
credit (B7): 75.84 (76.01)
pol (B14): 98.49 (98.82)
jannis (mathcal B4): 79.54 (80.29)

Reported results of Trompt model on Yandex dataset
helena : 37.90
jannis : 72.98
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import TabularBenchmark
from torch_frame.nn import Trompt

# GNN 相關導入
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import math
import sys

# 引入 DGM 模組
sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
from DGMlib.layers import DGM_d

# 統一的裝置選擇函式
def resolve_device(config: dict) -> torch.device:
    """從 config 選擇 GPU，否則回退到可用裝置"""
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
    """根據數據集規模自動計算最合適的 dgm_k（DGM 候選池大小）"""
    base_k = int(np.sqrt(num_samples))
    feature_factor = 1.0 + np.log1p(num_features) / 10
    adjusted_k = int(base_k * feature_factor)
    density_factor = 1.0
    if num_samples < 500:
        density_factor = 1.3
    elif num_samples > 5000:
        density_factor = 0.9
    adaptive_k = int(adjusted_k * density_factor)
    upper_limit = min(30, max(15, int(4 * np.sqrt(num_samples))))
    if num_samples < 1000:
        upper_limit = min(20, int(3 * np.sqrt(num_samples)))
    adaptive_k = max(5, min(adaptive_k, upper_limit))
    print(f"[DGM-K] Adaptive Calculation:")
    print(f"  - Dataset: {dataset_name if dataset_name else 'unknown'} | N={num_samples}, D={num_features}")
    print(f"  - Base k (√N): {base_k}")
    print(f"  - Adaptive k: {adaptive_k} (valid range: [5, {upper_limit}])")
    return adaptive_k

def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    """沿指定維度做 z-score 標準化"""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std

def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """對 edge_index 做對稱化並加入自迴路，移除重複邊"""
    device = edge_index.device
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    self_loops = torch.arange(num_nodes, device=device)
    self_edge = torch.stack([self_loops, self_loops], dim=0)
    ei = torch.cat([edge_index, rev, self_edge], dim=1)
    edge_ids = ei[0] * num_nodes + ei[1]
    unique_ids = torch.unique(edge_ids, sorted=False)
    ei0 = unique_ids // num_nodes
    ei1 = unique_ids % num_nodes
    ei_unique = torch.stack([ei0, ei1], dim=0)
    return ei_unique

# GNN 相關類和函數
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
    """構建 k-NN 圖"""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_list = []
    N = x_np.shape[0]
    for i in range(N):
        for j in indices[i][1:]:
            edge_list.append([i, j])
            if not directed:
                edge_list.append([j, i])
    if len(edge_list) == 0:
        edge_list = [[i, i] for i in range(N)]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


# class Trompt(Module):
#     r"""The Trompt model introduced in the
#     `"Trompt: Towards a Better Deep Neural Network for Tabular Data"
#     <https://arxiv.org/abs/2305.18446>`_ paper.

#     .. note::

#         For an example of using Trompt, see `examples/trompt.py
#         <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
#         trompt.py>`_.

#     Args:
#         channels (int): Hidden channel dimensionality
#         out_channels (int): Output channels dimensionality
#         num_prompts (int): Number of prompt columns.
#         num_layers (int, optional): Number of :class:`TromptConv` layers.
#             (default: :obj:`6`)
#         col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
#              A dictionary that maps column name into stats.
#              Available as :obj:`dataset.col_stats`.
#         col_names_dict (Dict[:obj:`torch_frame.stype`, List[str]]): A
#             dictionary that maps stype to a list of column names. The column
#             names are sorted based on the ordering that appear in
#             :obj:`tensor_frame.feat_dict`. Available as
#             :obj:`tensor_frame.col_names_dict`.
#         stype_encoder_dicts
#             (list[dict[:class:`torch_frame.stype`,
#             :class:`torch_frame.nn.encoder.StypeEncoder`]], optional):
#             A list of :obj:`num_layers` dictionaries that each dictionary maps
#             stypes into their stype encoders.
#             (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
#             for categorical feature and :obj:`LinearEncoder()` for
#             numerical feature)
#     """
#     def __init__(
#         self,
#         channels: int,
#         out_channels: int,
#         num_prompts: int,
#         num_layers: int,
#         # kwargs for encoder
#         col_stats: dict[str, dict[StatType, Any]],
#         col_names_dict: dict[torch_frame.stype, list[str]],
#         stype_encoder_dicts: list[dict[torch_frame.stype, StypeEncoder]]
#         | None = None,
#     ) -> None:
#         super().__init__()
#         if num_layers <= 0:
#             raise ValueError(
#                 f"num_layers must be a positive integer (got {num_layers})")

#         self.channels = channels
#         self.out_channels = out_channels
#         self.num_layers = num_layers
#         num_cols = sum(
#             [len(col_names) for col_names in col_names_dict.values()])

#         self.x_prompt = Parameter(torch.empty(num_prompts, channels))
#         self.encoders = ModuleList()
#         self.trompt_convs = ModuleList()
#         for i in range(num_layers):
#             if stype_encoder_dicts is None:
#                 stype_encoder_dict_layer = {
#                     stype.categorical:
#                     EmbeddingEncoder(
#                         post_module=LayerNorm(channels),
#                         na_strategy=NAStrategy.MOST_FREQUENT,
#                     ),
#                     stype.numerical:
#                     LinearEncoder(
#                         post_module=Sequential(
#                             ReLU(),
#                             LayerNorm(channels),
#                         ),
#                         na_strategy=NAStrategy.MEAN,
#                     ),
#                 }
#             else:
#                 stype_encoder_dict_layer = stype_encoder_dicts[i]

#             self.encoders.append(
#                 StypeWiseFeatureEncoder(
#                     out_channels=channels,
#                     col_stats=col_stats,
#                     col_names_dict=col_names_dict,
#                     stype_encoder_dict=stype_encoder_dict_layer,
#                 ))
#             self.trompt_convs.append(
#                 TromptConv(channels, num_cols, num_prompts))
#         # Decoder is shared across layers.
#         self.trompt_decoder = TromptDecoder(channels, out_channels,
#                                             num_prompts)
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         torch.nn.init.normal_(self.x_prompt, std=0.01)
#         for encoder in self.encoders:
#             encoder.reset_parameters()
#         for trompt_conv in self.trompt_convs:
#             trompt_conv.reset_parameters()
#         self.trompt_decoder.reset_parameters()

#     def forward_stacked(self, tf: TensorFrame) -> Tensor:
#         r"""Transforming :class:`TensorFrame` object into a series of output
#         predictions at each layer. Used during training to compute layer-wise
#         loss.

#         Args:
#             tf (:class:`torch_frame.TensorFrame`):
#                 Input :class:`TensorFrame` object.

#         Returns:
#             torch.Tensor: Output predictions stacked across layers. The
#                 shape is :obj:`[batch_size, num_layers, out_channels]`.
#         """
#         batch_size = len(tf)
#         outs = []
#         # [batch_size, num_prompts, channels]
#         x_prompt = self.x_prompt.repeat(batch_size, 1, 1)
#         for i in range(self.num_layers):
#             # [batch_size, num_cols, channels]
#             x, _ = self.encoders[i](tf)
#             # [batch_size, num_prompts, channels]
#             x_prompt = self.trompt_convs[i](x, x_prompt)
#             # [batch_size, out_channels]
#             out = self.trompt_decoder(x_prompt)
#             # [batch_size, 1, out_channels]
#             out = out.view(batch_size, 1, self.out_channels)
#             outs.append(out)
#         # [batch_size, num_layers, out_channels]
#         stacked_out = torch.cat(outs, dim=1)
#         return stacked_out

#     def forward(self, tf: TensorFrame) -> Tensor:
#         return self.forward_stacked(tf).mean(dim=1)
import torch
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import pandas as pd


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


    # 數據集包裝（直接合併三份 DataFrame，標記 split_col）
    from torch_frame.datasets import Yandex
    dataset = Yandex(train_df_gnn, val_df_gnn, test_df_gnn, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    # 根據 split_col 取得三份 tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 256)
    from torch_frame.data import DataLoader
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    
    return (train_loader, val_loader, test_loader,
            dataset.col_stats,
            dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 數據集包裝（直接合併三份 DataFrame，標記 split_col）
    from torch_frame.datasets import Yandex
    dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    # 根據 split_col 取得三份 tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 256)
    from torch_frame.data import DataLoader
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

def trompt_core_fn(material_outputs, config, task_type, gnn_stage='start'):
    print(f"[TromPT Core] Starting with gnn_stage: {gnn_stage}")
    train_tensor_frame = material_outputs['train_tensor_frame']
    val_tensor_frame = material_outputs['val_tensor_frame']
    test_tensor_frame = material_outputs['test_tensor_frame']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    device = material_outputs['device']
    out_channels = material_outputs['out_channels']
    device = material_outputs['device']
    is_classification = material_outputs['is_classification']
    is_binary_class = material_outputs['is_binary_class']
    metric_computer = material_outputs['metric_computer']
    metric = material_outputs['metric']
    col_stats = material_outputs['col_stats']
    col_names_dict = train_tensor_frame.col_names_dict
    
    # 獲取Trompt模型參數
    channels = config.get('channels', 128)
    num_prompts = config.get('num_prompts', 128)
    num_layers = config.get('num_layers', 6)
    print(f"[TromPT Core] Encoding with channels: {channels}, num_prompts: {num_prompts}, num_layers: {num_layers}")
    
    # 計算總列數
    num_cols = sum([len(col_names) for col_names in col_names_dict.values()])
    print(f"[TromPT Core] Total number of columns: {num_cols}")
    
    # 創建提示向量參數
    x_prompt = Parameter(torch.empty(num_prompts, channels, device=device).normal_(std=0.01))
    torch.nn.init.normal_(x_prompt, std=0.01)
    
    # GNN 組件初始化（對齊 ExcelFormer）
    gnn = None
    dgm_module = None
    self_attn = None
    self_attn_out = None
    attn_norm = None
    attn_out_norm = None
    column_embed = None
    pool_query = None
    gcn_to_attn = None
    ffn_pre = None
    ffn_post = None
    fusion_alpha_param = None
    gnn_hidden = config.get('gnn_hidden', 64)
    
    if gnn_stage in ['encoding', 'columnwise', 'decoding']:
        # 1. Multi-Head Self-Attention (列間交互)
        gnn_num_heads = config.get('gnn_num_heads', 4)
        self_attn = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=gnn_num_heads, batch_first=True).to(device)
        attn_norm = torch.nn.LayerNorm(channels).to(device)
        
        # 2. Column embedding (可學習的列位置編碼)
        column_embed = torch.nn.Parameter(torch.randn(num_cols, channels, device=device))
        
        # 3. Attention pooling query (用於將 self-attention 後的結果聚合為 row-level)
        pool_query = torch.nn.Parameter(torch.randn(channels, device=device))
        
        # 4. DGM_d 動態圖模組
        dgm_k = config.get('dgm_k', config.get('gnn_knn', 5))
        dgm_distance = config.get('dgm_distance', 'euclidean')
        
        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x
        
        dgm_embed_f = DGMEmbedWrapper()
        dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
        
        # 5. Batch GCN
        if gnn_stage == 'decoding':
            gnn = SimpleGCN(channels, gnn_hidden, out_channels, num_layers=2).to(device)
        else:
            gnn = SimpleGCN(channels, gnn_hidden, channels, num_layers=2).to(device)
        
        # 6. Self-Attention 解碼層（encoding/columnwise 需要）
        if gnn_stage in ['encoding', 'columnwise']:
            self_attn_out = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=gnn_num_heads, batch_first=True).to(device)
            gcn_to_attn = torch.nn.Linear(channels, channels).to(device)
            attn_out_norm = torch.nn.LayerNorm(channels).to(device)
            gnn_dropout = config.get('gnn_dropout', 0.1)
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
            # 7. 可學習的融合權重
            fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))
        
        print(f"[TromPT Core] Created complete GNN pipeline for {gnn_stage} stage:")
        print(f"  - Multi-Head Self-Attention (heads={gnn_num_heads})")
        print(f"  - DGM_d (k={dgm_k}, distance={dgm_distance})")
        print(f"  - Batch GCN (input={channels}, hidden={gnn_hidden}, output={out_channels if gnn_stage=='decoding' else channels})")
    
    # 創建多層編碼器
    encoders = ModuleList()
    for i in range(num_layers):
        # 為每一層創建編碼器字典
        stype_encoder_dict_layer = {
            stype.categorical: EmbeddingEncoder(
                post_module=LayerNorm(channels),
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: LinearEncoder(
                post_module=Sequential(
                    ReLU(),
                    LayerNorm(channels),
                ),
                na_strategy=NAStrategy.MEAN,
            ),
        }
        
        # 創建並添加該層的StypeWiseFeatureEncoder
        encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict_layer,
        ).to(device)
        encoders.append(encoder)
    
    # 對批次數據進行編碼的函數 - 添加 GNN 支持
    def encode_batch(tf, debug=False):
        batch_size = len(tf)
        layer_outputs = []
        
        if debug:
            print(f"[TromPT] Input batch_size: {batch_size}")
        
        # 為每一層執行編碼
        for i, encoder in enumerate(encoders):
            # 編碼特徵 [batch_size, num_cols, channels]
            x, _ = encoder(tf)
            
            if debug:
                print(f"[TromPT] Layer {i} encoder output: {x.shape}")
            
            # Stage: encoding GNN 處理 - 完整的 Self-Attention + DGM 管線
            if gnn_stage == 'encoding' and dgm_module is not None and i == 0:  # 只在第一層應用 encoding GNN
                if x.dim() == 3 and batch_size > 1:
                    # Step 1: Self-Attention 列間交互
                    tokens = x + column_embed.unsqueeze(0)  # [batch, num_cols, channels]
                    tokens_norm = attn_norm(tokens)
                    attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
                    tokens_attn = tokens + attn_out1
                    # FFN + 殘差
                    ffn_out1 = ffn_pre(attn_norm(tokens_attn))
                    tokens_attn = tokens_attn + ffn_out1
                    
                    # Step 2: Attention Pooling [batch, num_cols, channels] → [batch, channels]
                    pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels)
                    pool_weights = torch.softmax(pool_logits, dim=1)
                    x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)
                    
                    # Step 3: Mini-batch DGM 動態建圖
                    x_pooled_std = _standardize(x_pooled, dim=0)
                    x_pooled_batched = x_pooled_std.unsqueeze(0)
                    if hasattr(dgm_module, 'k'):
                        Ns_enc = x_pooled_batched.shape[1]
                        dgm_module.k = int(min(int(dgm_module.k), max(1, Ns_enc - 1)))
                    x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
                    x_dgm = x_dgm.squeeze(0)
                    edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])
                    
                    # Step 4: Batch GCN 處理
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
                    
                    if debug:
                        print(f"[TromPT] After encoding GNN (layer {i}): {x.shape}")
            
            layer_outputs.append(x)
        
        return layer_outputs, batch_size
    
    # 創建TromptConv層列表
    trompt_convs = ModuleList([
        TromptConv(channels, num_cols, num_prompts).to(device)
        for _ in range(num_layers)
    ])
    
    # 所有TromptConv層重置參數
    for trompt_conv in trompt_convs:
        trompt_conv.reset_parameters()
    
    # 定義列間交互處理函數 - 添加 GNN 支持
    def process_batch_interaction(encoded_features, batch_size, debug=False):
        """
        處理編碼後的特徵通過TromptConv進行列間交互
        """
        # 拓展提示向量以匹配批次大小 [batch_size, num_prompts, channels]
        x_prompt_batch = x_prompt.repeat(batch_size, 1, 1)
        
        # 每層的提示向量輸出
        prompts_outputs = []
        
        # 通過每一層的TromptConv處理
        for i in range(num_layers):
            # 獲取當前層的編碼特徵
            x = encoded_features[i]
            
            # 如果是第一層，使用初始提示向量
            if i == 0:
                prompt = x_prompt_batch
            else:
                # 否則使用上一層的提示向量輸出
                prompt = prompts_outputs[-1]
            
            # 應用TromptConv - 輸入是編碼特徵和提示向量，輸出是更新的提示向量
            updated_prompt = trompt_convs[i](x, prompt)
            
            if debug:
                print(f"[TromPT] Layer {i} TromptConv output: {updated_prompt.shape}")
            
            # Stage: columnwise GNN 處理 - 完整的 Self-Attention + DGM 管線
            if gnn_stage == 'columnwise' and dgm_module is not None:
                # 對提示向量應用完整的 GNN pipeline
                if updated_prompt.dim() == 3 and batch_size > 1:
                    # 獲取 updated_prompt 的維度信息
                    _, num_prompts_inner, channels_inner = updated_prompt.shape
                    
                    # Step 1: Self-Attention 列間交互（對 prompts）
                    # 這裡 prompts 相當於 columns
                    tokens = updated_prompt + column_embed[:num_prompts_inner].unsqueeze(0) if num_prompts_inner <= num_cols else updated_prompt
                    tokens_norm = attn_norm(tokens)
                    attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
                    tokens_attn = tokens + attn_out1
                    # FFN + 殘差
                    ffn_out1 = ffn_pre(attn_norm(tokens_attn))
                    tokens_attn = tokens_attn + ffn_out1
                    
                    # Step 2: Attention Pooling
                    pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels_inner)
                    pool_weights = torch.softmax(pool_logits, dim=1)
                    prompt_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)
                    
                    # Step 3: Mini-batch DGM 動態建圖
                    prompt_pooled_std = _standardize(prompt_pooled, dim=0)
                    prompt_pooled_batched = prompt_pooled_std.unsqueeze(0)
                    if hasattr(dgm_module, 'k'):
                        Ns_col = prompt_pooled_batched.shape[1]
                        dgm_module.k = int(min(int(dgm_module.k), max(1, Ns_col - 1)))
                    prompt_dgm, edge_index_dgm, logprobs_dgm = dgm_module(prompt_pooled_batched, A=None)
                    prompt_dgm = prompt_dgm.squeeze(0)
                    edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, prompt_dgm.shape[0])
                    
                    # Step 4: Batch GCN 處理
                    prompt_gnn_out = gnn(prompt_dgm, edge_index_dgm)
                    
                    # Step 5: Self-Attention 解碼
                    gcn_ctx = gcn_to_attn(prompt_gnn_out).unsqueeze(1)
                    tokens_with_ctx = tokens_attn + gcn_ctx
                    tokens_ctx_norm = attn_out_norm(tokens_with_ctx)
                    attn_out2, _ = self_attn_out(tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
                    tokens_mid = tokens_with_ctx + attn_out2
                    ffn_out2 = ffn_post(attn_out_norm(tokens_mid))
                    tokens_out = tokens_mid + ffn_out2
                    
                    # Step 6: 殘差融合
                    fusion_alpha = torch.sigmoid(fusion_alpha_param)
                    updated_prompt = updated_prompt + fusion_alpha * tokens_out
                    
                    if debug:
                        print(f"[TromPT] After columnwise GNN (layer {i}): {updated_prompt.shape}")
            
            prompts_outputs.append(updated_prompt)
        
        return prompts_outputs
    
    # 創建TromptDecoder
    trompt_decoder = TromptDecoder(
        channels, 
        out_channels, 
        num_prompts
    ).to(device)
    trompt_decoder.reset_parameters()
    
    # 定義前向傳播函數 - 分層版本（添加 debug 支持）
    def forward_stacked(tf, debug=False):
        """
        前向傳播 - 返回每一層的預測結果
        """
        # 編碼特徵
        encoded_features, batch_size = encode_batch(tf, debug=debug)
        
        # 通過TromptConv進行列間交互
        prompts_outputs = process_batch_interaction(encoded_features, batch_size, debug=debug)
        
        # 對每層的提示向量應用解碼器生成預測
        outs = []
        for i in range(num_layers):
            # 獲取當前層的提示向量
            prompt = prompts_outputs[i]
            
            # 應用解碼器
            out = trompt_decoder(prompt)
            
            if debug:
                print(f"[TromPT] Layer {i} decoder output: {out.shape}")
            
            # 調整形狀為[batch_size, 1, out_channels]
            out = out.view(batch_size, 1, out_channels)
            outs.append(out)
        
        # 堆疊所有層的輸出
        stacked_out = torch.cat(outs, dim=1)
        
        if debug:
            print(f"[TromPT] Final stacked output: {stacked_out.shape}")
        
        return stacked_out
    
    # 定義前向傳播函數 - 平均版本
    def forward(tf, debug=False):
        """
        前向傳播 - 返回所有層預測結果的平均（或 decoding 階段的 GCN 輸出）
        """
        # Decoding 階段：使用 GCN 作為 decoder
        if gnn_stage == 'decoding' and dgm_module is not None:
            # 編碼特徵
            encoded_features, batch_size = encode_batch(tf, debug=debug)
            
            # 通過TromptConv進行列間交互
            prompts_outputs = process_batch_interaction(encoded_features, batch_size, debug=debug)
            
            # 使用最後一層的 prompts
            final_prompts = prompts_outputs[-1]  # [batch, num_prompts, channels]
            
            # Step 1: Self-Attention 列間交互
            tokens = final_prompts + column_embed[:num_prompts].unsqueeze(0) if num_prompts <= num_cols else final_prompts
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            
            # Step 2: Attention Pooling
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)
            
            # Step 3: Mini-batch DGM 動態建圖
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)
            if hasattr(dgm_module, 'k'):
                Ns_dec = x_pooled_batched.shape[1]
                dgm_module.k = int(min(int(dgm_module.k), max(1, Ns_dec - 1)))
            x_dgm, edge_index_dgm, logprobs_dgm = dgm_module(x_pooled_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, x_dgm.shape[0])
            
            # Step 4: Batch GCN 作為 Decoder 直接輸出預測
            out = gnn(x_dgm, edge_index_dgm)  # [batch, out_channels]
            
            if debug:
                print(f"[TromPT] Decoding GCN output: {out.shape}")
            
            return out
        else:
            # 其他階段：使用標準 TromPT decoder
            stacked_out = forward_stacked(tf, debug=debug)
            return stacked_out.mean(dim=1)
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.001)
    
    # 收集所有參數 - 包括 GNN 參數（對齊 ExcelFormer）
    all_params = [x_prompt] + \
                 [p for encoder in encoders for p in encoder.parameters()] + \
                 [p for conv in trompt_convs for p in conv.parameters()] + \
                 list(trompt_decoder.parameters())
    
    # 添加 GNN 組件參數（根據階段）
    if gnn is not None:
        all_params.extend(list(gnn.parameters()))
    if dgm_module is not None:
        all_params.extend(list(dgm_module.parameters()))
    
    # encoding 階段的額外組件
    if gnn_stage == 'encoding':
        if self_attn is not None:
            all_params.extend(list(self_attn.parameters()))
        if attn_norm is not None:
            all_params.extend(list(attn_norm.parameters()))
        if self_attn_out is not None:
            all_params.extend(list(self_attn_out.parameters()))
        if attn_out_norm is not None:
            all_params.extend(list(attn_out_norm.parameters()))
        if column_embed is not None:
            all_params.append(column_embed)
        if gcn_to_attn is not None:
            all_params.extend(list(gcn_to_attn.parameters()))
        if ffn_pre is not None:
            all_params.extend(list(ffn_pre.parameters()))
        if ffn_post is not None:
            all_params.extend(list(ffn_post.parameters()))
        if pool_query is not None:
            all_params.append(pool_query)
        if fusion_alpha_param is not None:
            all_params.append(fusion_alpha_param)
    
    # decoding 階段的額外組件
    elif gnn_stage == 'decoding':
        if self_attn is not None:
            all_params.extend(list(self_attn.parameters()))
        if attn_norm is not None:
            all_params.extend(list(attn_norm.parameters()))
        if column_embed is not None:
            all_params.append(column_embed)
        if pool_query is not None:
            all_params.append(pool_query)
    
    # columnwise 階段的額外組件
    elif gnn_stage == 'columnwise':
        if self_attn is not None:
            all_params.extend(list(self_attn.parameters()))
        if attn_norm is not None:
            all_params.extend(list(attn_norm.parameters()))
        if self_attn_out is not None:
            all_params.extend(list(self_attn_out.parameters()))
        if attn_out_norm is not None:
            all_params.extend(list(attn_out_norm.parameters()))
        if column_embed is not None:
            all_params.append(column_embed)
        if gcn_to_attn is not None:
            all_params.extend(list(gcn_to_attn.parameters()))
        if ffn_pre is not None:
            all_params.extend(list(ffn_pre.parameters()))
        if ffn_post is not None:
            all_params.extend(list(ffn_post.parameters()))
        if pool_query is not None:
            all_params.append(pool_query)
        if fusion_alpha_param is not None:
            all_params.append(fusion_alpha_param)
    
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        for encoder in encoders:
            encoder.train()
        for conv in trompt_convs:
            conv.train()
        trompt_decoder.train()
        
        # 設置 GNN 組件為訓練模式（對齊 ExcelFormer）
        if gnn is not None:
            gnn.train()
        if dgm_module is not None:
            dgm_module.train()
        
        # encoding 階段的額外組件
        if gnn_stage == 'encoding':
            if self_attn is not None:
                self_attn.train()
            if attn_norm is not None:
                attn_norm.train()
            if self_attn_out is not None:
                self_attn_out.train()
            if attn_out_norm is not None:
                attn_out_norm.train()
            if gcn_to_attn is not None:
                gcn_to_attn.train()
            if ffn_pre is not None:
                ffn_pre.train()
            if ffn_post is not None:
                ffn_post.train()
        
        # decoding 階段的額外組件
        elif gnn_stage == 'decoding':
            if self_attn is not None:
                self_attn.train()
            if attn_norm is not None:
                attn_norm.train()
        
        # columnwise 階段的額外組件
        elif gnn_stage == 'columnwise':
            if self_attn is not None:
                self_attn.train()
            if attn_norm is not None:
                attn_norm.train()
            if self_attn_out is not None:
                self_attn_out.train()
            if attn_out_norm is not None:
                attn_out_norm.train()
            if gcn_to_attn is not None:
                gcn_to_attn.train()
            if ffn_pre is not None:
                ffn_pre.train()
            if ffn_post is not None:
                ffn_post.train()
        
        loss_accum = total_count = 0
        first_batch = True
        
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            
            # 只在第一個epoch的第一個batch啟用調試
            debug = (epoch == 1 and first_batch and gnn_stage in ['encoding', 'columnwise'])
            first_batch = False
            
            # 所有階段都使用 forward()（平均輸出），保證訓練和評估一致性
            pred = forward(tf, debug=debug)
            y = tf.y
            batch_size = len(tf)
            
            # 計算損失（使用平均層輸出，與評估邏輯一致）
            if is_classification:
                loss = F.cross_entropy(pred, y)
            else:
                loss = F.mse_loss(pred.view(-1), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * batch_size
            total_count += batch_size
            optimizer.step()
        
        return loss_accum / total_count
    
    # 定義測試函數
    @torch.no_grad()
    def test(loader):
        # 設置為評估模式
        for encoder in encoders:
            encoder.eval()
        for conv in trompt_convs:
            conv.eval()
        trompt_decoder.eval()
        
        # 設置 GNN 組件為評估模式（對齊 ExcelFormer）
        if gnn is not None:
            gnn.eval()
        if dgm_module is not None:
            dgm_module.eval()
        
        # encoding 階段的額外組件
        if gnn_stage == 'encoding':
            if self_attn is not None:
                self_attn.eval()
            if attn_norm is not None:
                attn_norm.eval()
            if self_attn_out is not None:
                self_attn_out.eval()
            if attn_out_norm is not None:
                attn_out_norm.eval()
            if gcn_to_attn is not None:
                gcn_to_attn.eval()
            if ffn_pre is not None:
                ffn_pre.eval()
            if ffn_post is not None:
                ffn_post.eval()
        
        # decoding 階段的額外組件
        elif gnn_stage == 'decoding':
            if self_attn is not None:
                self_attn.eval()
            if attn_norm is not None:
                attn_norm.eval()
        
        # columnwise 階段的額外組件
        elif gnn_stage == 'columnwise':
            if self_attn is not None:
                self_attn.eval()
            if attn_norm is not None:
                attn_norm.eval()
            if self_attn_out is not None:
                self_attn_out.eval()
            if attn_out_norm is not None:
                attn_out_norm.eval()
            if gcn_to_attn is not None:
                gcn_to_attn.eval()
            if ffn_pre is not None:
                ffn_pre.eval()
            if ffn_post is not None:
                ffn_post.eval()
        
        metric_computer.reset()
        
        for tf in loader:
            tf = tf.to(device)
            # 使用平均前向傳播
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
    
    # 標準訓練流程（用於所有 GNN 階段，包括 decoding）
    # decoding 階段已在 forward() 函數中實現（使用 GNN as Decoder）
    # 初始化最佳指標
    if is_classification:
        best_val_metric = -float('inf')  # Classification metrics are maximized
        best_test_metric = -float('inf')
    else:
        best_val_metric = float('inf')  # Regression metrics (RMSE) are minimized
        best_test_metric = float('inf')
    
    # 記錄訓練過程
    train_losses = []
    train_metrics = []
    val_metrics = []
    test_metrics = []
    
    # 訓練循環
    epochs = config.get('epochs', 300)
    patience = config.get('patience', 10)
    early_stop_counter = 0
    best_epoch = 0
    stopped_epoch = epochs  # 預設停止在最後一個 epoch
    
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        train_metric = test(train_loader)
        val_metric = test(val_loader)
        test_metric = test(test_loader)
        
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        test_metrics.append(test_metric)
        
        improved = (val_metric > best_val_metric) if is_classification else (val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            best_test_metric = test_metric
            best_epoch = epoch
            early_stop_counter = 0
            improvement_marker = "✓"
        else:
            early_stop_counter += 1
            improvement_marker = "✗"
        
        print(f'[TromPT Core] Epoch {epoch:3d} {improvement_marker} | Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
              f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f} | Early Stop: {early_stop_counter}/{patience}')
        
        # 學習率調整
        lr_scheduler.step()
        
        if early_stop_counter >= patience:
            stopped_epoch = epoch  # 記錄停止的 epoch 編號
            print(f"[TromPT Core] Early stopping at epoch {epoch}")
            break
        
        print(f'[TromPT Core] Best Val {metric}: {best_val_metric:.4f}, '
              f'Best Test {metric}: {best_test_metric:.4f}')
    
    # 訓練完成，返回結果
    return {
        'train_losses': train_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_val_metric': best_val_metric,
        'final_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'metric_computer': metric_computer,
        'metric': metric,
        'stopped_epoch': stopped_epoch,  # 訓練停止的 epoch 編號
        'early_stop_counter': early_stop_counter,  # 沒有改進的連續 epoch 數（距離最佳的輪數）
        'best_epoch': best_epoch,  # 最佳驗證指標出現的 epoch
        'early_stop_epochs': stopped_epoch,  # 早停輪數（用於結果報告）
        'gnn_early_stop_epochs': material_outputs.get('gnn_early_stop_epochs', 0),
        'model_type': f'trompt_{gnn_stage}_gnn' if gnn_stage != 'none' else 'standard_trompt'
    }
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.001)
    
    # 收集所有參數 - 包括 GNN 參數
    all_params = [x_prompt] + \
                 [p for encoder in encoders for p in encoder.parameters()] + \
                 [p for conv in trompt_convs for p in conv.parameters()] + \
                 list(trompt_decoder.parameters())
    
    # 只在需要時添加 GNN 參數
    if gnn is not None:
        all_params.extend(list(gnn.parameters()))
    
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    # 訓練和測試函數將在上面的 if-else 邏輯中定義和使用


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    print("Trompt - 五階段執行")
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    try:
        # 階段0: 開始
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        gnn_early_stop_epochs=0
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

        results=trompt_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)
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



#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models trompt --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models trompt --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models trompt --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models trompt --gnn_stages all --epochs 2
#  python main.py --dataset helena --models trompt --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models trompt --gnn_stages all --epochs 2