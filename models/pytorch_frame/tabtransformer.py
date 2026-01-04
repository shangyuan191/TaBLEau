from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    SELU,
    BatchNorm1d,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TabTransformerConv
from torch_frame.nn.encoder.stype_encoder import EmbeddingEncoder, StackEncoder
from torch_frame.typing import NAStrategy


"""Reported (reproduced, xgboost) results of of TabTransformer model based on
Table 1 of original paper https://arxiv.org/abs/2012.06678.

adult: 73.8 (88.86)
bank-marketing: 93.4 (90.83, 81.00)
dota2: 63.3 (52.44, 53.75)
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import AdultCensusIncome, BankMarketing, Dota2
from torch_frame.nn import TabTransformer
from torch_frame.datasets.yandex import Yandex

# 新增 imports（對齊 ExcelFormer）
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import sys

# 引入 DGM 模組
sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
from DGMlib.layers import DGM_d

# 統一的裝置選擇函式：從 config 選擇 GPU，否則回退到可用裝置
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
    print(f"  - Feature Factor: {feature_factor:.2f}x (features={num_features})")
    print(f"  - Density Factor: {density_factor:.2f}x (samples={num_samples})")
    print(f"  - Adaptive k: {adaptive_k} (valid range: [5, {upper_limit}])")
    return adaptive_k

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


    r"""The Tab-Transformer model introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    The model pads a column positional embedding in categorical feature
    embeddings and executes multi-layer column-interaction modeling exclusively
    on the categorical features. For numerical features, the model simply
    applies layer normalization on input features. The model utilizes an
    MLP(Multilayer Perceptron) for decoding.

    .. note::

        For an example of using TabTransformer, see `examples/tabtransformer.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        tabtransformer.py>`_.

    Args:
        channels (int): Input channel dimensionality.
        out_channels (int): Output channels dimensionality.
        num_layers (int): Number of convolution layers.
        num_heads (int): Number of heads in the self-attention layer.
        encoder_pad_size (int): Size of positional encoding padding to the
            categorical embeddings.
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        encoder_pad_size: int,
        attn_dropout: float,
        ffn_dropout: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        self.col_names_dict = col_names_dict
        categorical_col_len = 0
        numerical_col_len = 0
        if stype.categorical in self.col_names_dict:
            categorical_stats_list = [
                col_stats[col_name]
                for col_name in self.col_names_dict[stype.categorical]
            ]
            categorical_col_len = len(self.col_names_dict[stype.categorical])
            self.cat_encoder = EmbeddingEncoder(
                out_channels=channels - encoder_pad_size,
                stats_list=categorical_stats_list,
                stype=stype.categorical,
                na_strategy=NAStrategy.MOST_FREQUENT,
            )
            # Use the categorical embedding with EmbeddingEncoder and
            # added contextual padding to the end of each feature.
            self.pad_embedding = Embedding(categorical_col_len,
                                           encoder_pad_size)
            # Apply transformer convolution only over categorical columns
            self.tab_transformer_convs = ModuleList([
                TabTransformerConv(channels=channels, num_heads=num_heads,
                                   attn_dropout=attn_dropout,
                                   ffn_dropout=ffn_dropout)
                for _ in range(num_layers)
            ])
        if stype.numerical in self.col_names_dict:
            numerical_stats_list = [
                col_stats[col_name]
                for col_name in self.col_names_dict[stype.numerical]
            ]
            numerical_col_len = len(self.col_names_dict[stype.numerical])
            # Use stack encoder to normalize the numerical columns.
            self.num_encoder = StackEncoder(
                out_channels=1,
                stats_list=numerical_stats_list,
                stype=stype.numerical,
            )
            self.num_norm = LayerNorm(numerical_col_len)
        mlp_input_len = categorical_col_len * channels + numerical_col_len
        mlp_first_hidden_layer_size = 2 * mlp_input_len
        mlp_second_hidden_layer_size = 4 * mlp_input_len
        self.decoder = Sequential(
            Linear(mlp_input_len, mlp_first_hidden_layer_size),
            BatchNorm1d(mlp_first_hidden_layer_size), SELU(),
            Linear(2 * mlp_input_len, mlp_second_hidden_layer_size),
            BatchNorm1d(mlp_second_hidden_layer_size), SELU(),
            Linear(mlp_second_hidden_layer_size, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if stype.categorical in self.col_names_dict:
            self.cat_encoder.reset_parameters()
            torch.nn.init.normal_(self.pad_embedding.weight, std=0.01)
            for tab_transformer_conv in self.tab_transformer_convs:
                tab_transformer_conv.reset_parameters()
        if stype.numerical in self.col_names_dict:
            self.num_encoder.reset_parameters()
            self.num_norm.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, SELU):
                m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        xs = []
        batch_size = len(tf)
        if stype.categorical in self.col_names_dict:
            x_cat = self.cat_encoder(tf.feat_dict[stype.categorical])
            # A positional embedding [batch_size, num_cols, encoder_pad_size]
            # is padded to the categorical embedding
            # [batch_size, num_cols, channels].
            pos_enc_pad = self.pad_embedding.weight.unsqueeze(0).repeat(
                batch_size, 1, 1)
            # The final categorical embedding is of size [B, num_cols,
            # channels + encoder_pad_size]
            x_cat = torch.cat((x_cat, pos_enc_pad), dim=-1)
            for tab_transformer_conv in self.tab_transformer_convs:
                x_cat = tab_transformer_conv(x_cat)
            x_cat = x_cat.reshape(batch_size, math.prod(x_cat.shape[1:]))
            xs.append(x_cat)
        if stype.numerical in self.col_names_dict:
            x_num = self.num_encoder(tf.feat_dict[stype.numerical])
            x_num = x_num.view(batch_size, math.prod(x_num.shape[1:]))
            x_num = self.num_norm(x_num)
            xs.append(x_num)
        x = torch.cat(xs, dim=1)
        out = self.decoder(x)
        return out

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

    # 7. Yandex 數據集包裝
    dataset = Yandex(train_df_gnn, val_df_gnn, test_df_gnn, name=dataset_name, task_type=task_type)
    dataset.materialize()

    # 8. split tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 4096)  # TabNet通常使用較大的批次
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    return train_loader, val_loader, test_loader,dataset.col_stats, dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs


def gnn_decoding_eval(train_loader, val_loader, test_loader, config, task_type, metric_computer, 
                      cat_encoder, num_encoder, num_norm, pad_embedding, tab_transformer_convs, device):
    """
    TabTransformer 的 GNN decoding 評估函數 - 修復 embedding_dim 錯誤
    """
    print("[TabTransformer GNN Decoder] Starting evaluation...")
    is_classification = task_type in ['binclass', 'multiclass']
    is_binary_class = task_type == 'binclass'
    
    # 修復1: 建立全局映射
    def build_global_target_mapping():
        all_targets = []
        for loader in [train_loader, val_loader, test_loader]:
            for tf in loader:
                all_targets.append(tf.y)
                if len(all_targets) > 100:
                    break
            if len(all_targets) > 100:
                break
        
        if all_targets:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            unique_targets = torch.unique(all_targets_tensor).sort()[0]
            
            print(f"[TabTransformer GNN Decoder] Global target analysis:")
            print(f"  Unique targets: {unique_targets.tolist()}")
            print(f"  Target count: {len(unique_targets)}")
            
            global_mapping = {val.item(): i for i, val in enumerate(unique_targets)}
            return global_mapping, len(unique_targets)
        else:
            return {}, 1
    
    # 建立全局標籤映射
    if is_classification and not is_binary_class:
        global_target_mapping, out_channels = build_global_target_mapping()
        print(f"[TabTransformer GNN Decoder] Global mapping: {global_target_mapping}")
    else:
        global_target_mapping = {}
        out_channels = 1
    
    print(f"[TabTransformer GNN Decoder] Output channels: {out_channels}")
    
    # 修復2: embedding 提取函數
    def get_embeddings_batch_wise(loader, max_samples=3000):
        all_embeds, all_targets = [], []
        total_samples = 0
        
        with torch.no_grad():
            for tf in loader:
                if total_samples >= max_samples:
                    break
                    
                tf = tf.to(device)
                batch_size = len(tf)
                xs = []
                
                # 類別特徵處理
                if torch_frame.stype.categorical in tf.col_names_dict and cat_encoder is not None:
                    try:
                        x_cat = cat_encoder(tf.feat_dict[torch_frame.stype.categorical])
                        if pad_embedding is not None:
                            pos_enc_pad = pad_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                            x_cat_with_pos = torch.cat((x_cat, pos_enc_pad), dim=-1)
                        else:
                            x_cat_with_pos = x_cat
                        
                        if tab_transformer_convs is not None:
                            for conv in tab_transformer_convs:
                                x_cat_with_pos = conv(x_cat_with_pos)
                        
                        x_cat_flat = x_cat_with_pos.reshape(batch_size, -1)
                        xs.append(x_cat_flat)
                    except Exception as e:
                        print(f"[Debug] Failed to process categorical features: {e}")
                
                # 數值特徵處理
                if torch_frame.stype.numerical in tf.col_names_dict and num_encoder is not None:
                    try:
                        x_num = num_encoder(tf.feat_dict[torch_frame.stype.numerical])
                        x_num = x_num.view(batch_size, -1)
                        if num_norm is not None:
                            x_num = num_norm(x_num)
                        xs.append(x_num)
                    except Exception as e:
                        print(f"[Debug] Failed to process numerical features: {e}")
                
                if xs:
                    embeddings = torch.cat(xs, dim=1)
                    all_embeds.append(embeddings.cpu())
                    
                    # 使用全局映射處理目標標籤
                    targets = tf.y.cpu()
                    if is_classification and not is_binary_class and global_target_mapping:
                        mapped_targets = torch.tensor([
                            global_target_mapping.get(t.item(), 0) for t in targets
                        ])
                        all_targets.append(mapped_targets)
                    else:
                        all_targets.append(targets)
                    total_samples += batch_size
        
        if all_embeds:
            return torch.cat(all_embeds, dim=0), torch.cat(all_targets, dim=0)
        else:
            return torch.empty(0, 1), torch.empty(0)
    
    # 提取 embeddings
    max_samples = config.get('gnn_max_samples', 3000)
    print(f"[TabTransformer GNN Decoder] Computing embeddings...")
    
    try:
        train_embeddings, train_targets = get_embeddings_batch_wise(train_loader, max_samples)
        val_embeddings, val_targets = get_embeddings_batch_wise(val_loader, max_samples)
        test_embeddings, test_targets = get_embeddings_batch_wise(test_loader, max_samples)
        
        # 檢查是否成功提取到 embeddings
        if (train_embeddings.numel() == 0 or val_embeddings.numel() == 0 or 
            test_embeddings.numel() == 0):
            print("[TabTransformer GNN Decoder] Failed to extract embeddings")
            return 0.0, 0.0, 0
        
        # 關鍵修復：在這裡定義 embedding_dim
        embedding_dim = train_embeddings.shape[1]
        print(f"[TabTransformer GNN Decoder] Embedding dimension: {embedding_dim}")
        print(f"[TabTransformer GNN Decoder] Train embeddings: {train_embeddings.shape}")
        print(f"[TabTransformer GNN Decoder] Val embeddings: {val_embeddings.shape}")
        print(f"[TabTransformer GNN Decoder] Test embeddings: {test_embeddings.shape}")
        
    except Exception as e:
        print(f"[TabTransformer GNN Decoder] Error during embedding extraction: {e}")
        return 0.0, 0.0, 0
    
    # 驗證標籤範圍
    print(f"[TabTransformer GNN Decoder] Target verification:")
    print(f"  Train targets range: {train_targets.min().item()} - {train_targets.max().item()}")
    print(f"  Val targets range: {val_targets.min().item()} - {val_targets.max().item()}")
    print(f"  Test targets range: {test_targets.min().item()} - {test_targets.max().item()}")
    print(f"  Expected range: 0 - {out_channels - 1}")
    
    # 確保所有標籤都在有效範圍內
    if is_classification and not is_binary_class:
        train_targets = torch.clamp(train_targets, 0, out_channels - 1)
        val_targets = torch.clamp(val_targets, 0, out_channels - 1)
        test_targets = torch.clamp(test_targets, 0, out_channels - 1)
    
    # 合併 embeddings
    all_embeddings = torch.cat([train_embeddings, val_embeddings, test_embeddings], dim=0).to(device)
    all_targets = torch.cat([train_targets, val_targets, test_targets], dim=0).to(device)
    
    # 再次確認目標標籤範圍
    if is_classification and not is_binary_class:
        max_target_val = all_targets.max().item()
        if max_target_val >= out_channels:
            print(f"[TabTransformer GNN Decoder] ERROR: Max target {max_target_val} >= out_channels {out_channels}")
            out_channels = max_target_val + 1
            print(f"[TabTransformer GNN Decoder] Adjusted output channels to: {out_channels}")
    
    # 建立 KNN 圖
    def safe_knn_graph(x, k):
        try:
            n_samples = x.shape[0]
            k = min(k, n_samples - 1)
            
            if k <= 0:
                edge_index = torch.arange(n_samples).repeat(2, 1)
                return edge_index
            
            x_np = x.cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='euclidean')
            nbrs.fit(x_np)
            _, indices = nbrs.kneighbors(x_np)
            
            edge_index = []
            for i in range(n_samples):
                for j in indices[i][1:]:
                    edge_index.append([i, j])
            
            if not edge_index:
                edge_index = [[i, i] for i in range(n_samples)]
            
            return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        except Exception as e:
            print(f"[TabTransformer GNN Decoder] KNN graph construction failed: {e}")
            n_samples = x.shape[0]
            edge_index = torch.arange(n_samples).repeat(2, 1)
            return edge_index
    
    k = 5
    print(f"[TabTransformer GNN Decoder] Building KNN graph with k={k}")
    edge_index = safe_knn_graph(all_embeddings, k).to(device)
    
    # 創建 GNN 架構（現在 embedding_dim 已經定義了）
    gnn_hidden = config.get('gnn_hidden', 64)
    
    class StableGNNDecoder(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
            self.dropout = torch.nn.Dropout(0.2)
            self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return x
    
    try:
        gnn_decoder = StableGNNDecoder(embedding_dim, gnn_hidden, out_channels).to(device)
        print(f"[TabTransformer GNN Decoder] Created GNN decoder: {embedding_dim} -> {gnn_hidden} -> {out_channels}")
    except Exception as e:
        print(f"[TabTransformer GNN Decoder] Failed to create GNN decoder: {e}")
        return 0.0, 0.0, 0
    # 優化器設置
    gnn_lr = config.get('gnn_lr', 0.0001)
    gnn_optimizer = torch.optim.Adam(gnn_decoder.parameters(), lr=gnn_lr, weight_decay=1e-5)
    
    # 數據分割
    n_train = len(train_embeddings)
    n_val = len(val_embeddings)
    n_test = len(test_embeddings)
    N = len(all_embeddings)
    
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    
    # 訓練設置
    gnn_epochs = 200
    gnn_patience = config.get('gnn_patience', 10)
    
    print(f"[TabTransformer GNN Decoder] Training GNN decoder for {gnn_epochs} epochs...")
    
    if is_classification:
        best_val_metric = 0
    else:
        best_val_metric = float('inf')
    
    best_test_metric = best_val_metric
    early_stop_counter = 0
    early_stop_epochs = 0
    
    gnn_decoder.train()
    
    for epoch in range(1, gnn_epochs + 1):
        try:
            gnn_optimizer.zero_grad()
            
            predictions = gnn_decoder(all_embeddings, edge_index)
            
            # 檢查預測值的有效性
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print(f"[TabTransformer GNN Decoder] Invalid predictions at epoch {epoch}, stopping")
                break
            
            # 計算損失
            if is_classification:
                if is_binary_class:
                    train_loss = F.binary_cross_entropy_with_logits(
                        predictions[train_mask].squeeze(), all_targets[train_mask].float()
                    )
                else:
                    # 確保目標標籤在有效範圍內
                    train_targets_clamped = torch.clamp(all_targets[train_mask], 0, out_channels-1)
                    train_loss = F.cross_entropy(predictions[train_mask], train_targets_clamped)
            else:
                train_loss = F.mse_loss(predictions[train_mask].squeeze(), all_targets[train_mask])
            
            # 檢查損失值
            if torch.isnan(train_loss) or torch.isinf(train_loss):
                print(f"[TabTransformer GNN Decoder] Invalid loss at epoch {epoch}, stopping")
                break
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_decoder.parameters(), max_norm=1.0)
            gnn_optimizer.step()
            
            # 評估
            if epoch % 10 == 0 or epoch == gnn_epochs:
                gnn_decoder.eval()
                
                with torch.no_grad():
                    eval_predictions = gnn_decoder(all_embeddings, edge_index)
                    
                    if torch.isnan(eval_predictions).any() or torch.isinf(eval_predictions).any():
                        print(f"[TabTransformer GNN Decoder] Invalid eval predictions at epoch {epoch}")
                        gnn_decoder.train()
                        continue
                    
                    # 計算指標
                    if is_classification:
                        if is_binary_class:
                            val_probs = torch.sigmoid(eval_predictions[val_mask].squeeze())
                            val_metric = roc_auc_score(all_targets[val_mask].cpu(), val_probs.cpu())
                            
                            test_probs = torch.sigmoid(eval_predictions[test_mask].squeeze())
                            test_metric = roc_auc_score(all_targets[test_mask].cpu(), test_probs.cpu())
                        else:
                            val_preds = eval_predictions[val_mask].argmax(dim=1)
                            val_targets_clamped = torch.clamp(all_targets[val_mask], 0, out_channels-1)
                            val_metric = (val_preds == val_targets_clamped).float().mean().item()
                            
                            test_preds = eval_predictions[test_mask].argmax(dim=1)
                            test_targets_clamped = torch.clamp(all_targets[test_mask], 0, out_channels-1)
                            test_metric = (test_preds == test_targets_clamped).float().mean().item()
                    else:
                        val_metric = F.mse_loss(eval_predictions[val_mask].squeeze(), all_targets[val_mask]).sqrt().item()
                        test_metric = F.mse_loss(eval_predictions[test_mask].squeeze(), all_targets[test_mask]).sqrt().item()
                    
                    # 檢查指標有效性
                    if math.isnan(val_metric) or math.isinf(val_metric):
                        print(f"[TabTransformer GNN Decoder] Invalid metric at epoch {epoch}")
                        gnn_decoder.train()
                        continue
                    
                    # 更新最佳指標
                    improved = (val_metric > best_val_metric) if is_classification else (val_metric < best_val_metric)
                    if improved:
                        best_val_metric = val_metric
                        best_test_metric = test_metric
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    
                    print(f'[TabTransformer GNN Decoder] Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                          f'Val Metric: {val_metric:.4f}, Test Metric: {test_metric:.4f}')
                    
                    if early_stop_counter >= gnn_patience:
                        print(f"[TabTransformer GNN Decoder] Early stopping at epoch {epoch}")
                        early_stop_epochs = epoch
                        break
                
                gnn_decoder.train()
                
        except Exception as e:
            print(f"[TabTransformer GNN Decoder] Error at epoch {epoch}: {e}")
            break
    
    # 確保返回有效值
    if math.isnan(best_val_metric) or math.isinf(best_val_metric):
        best_val_metric = 0.0 if is_classification else float('inf')
    if math.isnan(best_test_metric) or math.isinf(best_test_metric):
        best_test_metric = 0.0 if is_classification else float('inf')
    
    print(f"[TabTransformer GNN Decoder] Best Val Metric: {best_val_metric:.4f}")
    print(f"[TabTransformer GNN Decoder] Best Test Metric: {best_test_metric:.4f}")
    
    return best_val_metric, best_test_metric, early_stop_epochs
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
    dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    # 根據 split_col 取得三份 tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 128)
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
        'device': device,
        'gnn_early_stop_epochs': 0  # 初始值，只在 start/materialize 阶段会被更新
    }


def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    """
    在 start_fn 和 materialize_fn 之間插入「自注意力 → DGM動態圖 → GCN → 自注意力」的管線。
    
    與 tabnet.py 中的 gnn_after_start_fn 相同實現。
    """
    device = resolve_device(config)
    print(f"[START-GNN-DGM] Using device: {device}")
    
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

    params = list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters()) \
        + list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters()) \
        + list(out_proj.parameters()) + list(dgm_module.parameters()) + [column_embed, pool_query]

    optimizer = torch.optim.Adam(params, lr=lr)

    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_states = None

    def forward_pass(x_tensor, dgm_module_inst):
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
        print(f"[START-GNN-DGM] split Ns={Ns}, edges={E}, avg_deg={avg_deg:.2f}")
        
        return logits, recon, logprobs_dgm

    for epoch in range(gnn_epochs):
        dgm_module.train()
        gnn.train()
        attn_in.train(); attn_out.train(); input_proj.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()

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

        dgm_module.eval()
        gnn.eval()
        attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
        
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

    dgm_module.eval()
    gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
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


    """
    階段1: Materialization - 將已切分的 train/val/test DataFrame 合併並轉換為張量格式
    """
    print("Executing materialize_fn")
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    dataset_name = dataset_results['dataset']
    task_type = dataset_results['info']['task_type']

    # 使用統一的設備管理
    device = resolve_device(config)
    print(f"[MATERIALIZE] Final device: {device}")

    # 數據集包裝（直接合併三份 DataFrame，標記 split_col）
    dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    # 根據 split_col 取得三份 tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 128)
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


def gnn_after_materialize_fn(train_tensor_frame, val_tensor_frame, test_tensor_frame, config, dataset_name, task_type):
    """
    在 materialize_fn 之後插入「自注意力 → DGM動態圖 → GCN → 自注意力」的管線。
    
    與 tabnet.py 中的 gnn_after_materialize_fn 相同實現。
    """
    device = resolve_device(config)
    print(f"[MATERIALIZE-GNN-DGM] Using device: {device}")
    
    # 從 TensorFrame 提取 DataFrame
    def tensor_frame_to_df(tensor_frame):
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
        dgm_k = config['dgm_k']
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
        print(f"[MATERIALIZE-GNN-DGM] Detected num_classes: {num_classes}")
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
    
    params = list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters()) \
        + list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters()) \
        + list(out_proj.parameters()) + list(dgm_module.parameters()) + [column_embed, pool_query]
    
    optimizer = torch.optim.Adam(params, lr=lr)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_states = None
    
    def forward_pass(x_tensor, dgm_module_inst):
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
        dgm_module.train()
        gnn.train()
        attn_in.train(); attn_out.train(); input_proj.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()
        
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
        
        dgm_module.eval()
        gnn.eval()
        attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
        
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
    
    dgm_module.eval()
    gnn.eval(); attn_in.eval(); attn_out.eval(); input_proj.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
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
    
    batch_size = config.get('batch_size', 128)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)
    
    return (train_loader, val_loader, test_loader,
            dataset.col_stats, dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs)


def tabtransformer_core_fn(material_outputs, config, task_type, gnn_stage=None):
    train_tensor_frame = material_outputs['train_tensor_frame']
    val_tensor_frame = material_outputs['val_tensor_frame']
    test_tensor_frame = material_outputs['test_tensor_frame']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    col_stats = material_outputs['col_stats']
    col_names_dict = train_tensor_frame.col_names_dict
    device = material_outputs['device']
    is_classification = material_outputs['is_classification']
    is_binary_class = material_outputs['is_binary_class']
    metric_computer = material_outputs['metric_computer']
    metric = material_outputs['metric']
    # 獲取模型參數
    channels = config.get('channels', 32)
    out_channels = material_outputs['out_channels']
    encoder_pad_size = config.get('encoder_pad_size', 2)
    print(f"Encoding with channels: {channels}, encoder_pad_size: {encoder_pad_size}")
    # 初始化編碼器變量
    cat_encoder = None
    pad_embedding = None
    num_encoder = None
    num_norm = None
    
    # 計算特徵維度
    categorical_col_len = 0
    numerical_col_len = 0
    patience = config.get('patience', 10)
    
    # GNN 組件初始化（對齐 ExcelFormer 的完整實現）
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
    
    if gnn_stage in ['encoding', 'columnwise', 'decoding']:
        # 1. Multi-Head Self-Attention (列間交互)
        gnn_num_heads = config.get('gnn_num_heads', 4)
        self_attn = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=gnn_num_heads, batch_first=True).to(device)
        attn_norm = torch.nn.LayerNorm(channels).to(device)
        
        # 2. Column embedding (可學習的列位置編碼)
        num_cols = train_tensor_frame.num_cols
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
        gnn_hidden = config.get('gnn_hidden', 64)
        if gnn_stage == 'decoding':
            gnn = SimpleGCN(channels, gnn_hidden, out_channels, num_layers=2).to(device)
        else:
            gnn = SimpleGCN(channels, gnn_hidden, channels, num_layers=2).to(device)
        
        # 6. Self-Attention 解碼層 (encoding/columnwise 需要)
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
    # 創建類別特徵編碼器
    if stype.categorical in col_names_dict:
        categorical_stats_list = [
            col_stats[col_name]
            for col_name in col_names_dict[stype.categorical]
        ]
        categorical_col_len = len(col_names_dict[stype.categorical])
        
        # 使用EmbeddingEncoder編碼類別特徵
        cat_encoder = EmbeddingEncoder(
            out_channels=channels - encoder_pad_size,
            stats_list=categorical_stats_list,
            stype=stype.categorical,
            na_strategy=NAStrategy.MOST_FREQUENT,
        ).to(device)
        
        # 用於列位置嵌入的padding
        pad_embedding = Embedding(categorical_col_len, encoder_pad_size).to(device)
        torch.nn.init.normal_(pad_embedding.weight, std=0.01)
    
    # 創建數值特徵編碼器
    if stype.numerical in col_names_dict:
        numerical_stats_list = [
            col_stats[col_name]
            for col_name in col_names_dict[stype.numerical]
        ]
        numerical_col_len = len(col_names_dict[stype.numerical])
        
        # 使用StackEncoder對數值列進行標準化
        num_encoder = StackEncoder(
            out_channels=1,
            stats_list=numerical_stats_list,
            stype=stype.numerical,
        ).to(device)
        
        num_norm = LayerNorm(numerical_col_len).to(device)
    
    # 對批次數據進行編碼的函數
    def encode_batch(tf):
        xs = []
        batch_size = len(tf)
        
        # 編碼類別特徵
        if stype.categorical in col_names_dict and cat_encoder is not None:
            x_cat = cat_encoder(tf.feat_dict[stype.categorical])
            # 添加位置嵌入
            pos_enc_pad = pad_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x_cat_with_pos = torch.cat((x_cat, pos_enc_pad), dim=-1)
            xs.append((x_cat_with_pos, "categorical"))
        
        # 編碼數值特徵
        if stype.numerical in col_names_dict and num_encoder is not None:
            x_num = num_encoder(tf.feat_dict[stype.numerical])
            x_num = x_num.view(batch_size, -1)
            x_num_norm = num_norm(x_num)
            xs.append((x_num_norm, "numerical"))
        
        return xs
    
    # 獲取Transformer的參數
    num_layers = config.get('num_layers', 6)
    num_heads = config.get('num_heads', 8)
    attn_dropout = config.get('attention_dropout', 0.3)
    ffn_dropout = config.get('ffn_dropout', 0.3)
    
    print(f"Building TabTransformer with {num_layers} layers, {num_heads} heads")
    # 創建TabTransformer的列間交互層 - 僅用於類別特徵
    tab_transformer_convs = None
    if stype.categorical in col_names_dict:
        tab_transformer_convs = ModuleList([
            TabTransformerConv(
                channels=channels,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout
            ).to(device)
            for _ in range(num_layers)
        ])
    
    # 定義列間交互處理函數
    def process_batch_interaction(encoded_features):
        processed_features = []
        
        for feature, feature_type in encoded_features:
            # 僅對類別特徵應用 TabTransformerConv
            if feature_type == "categorical" and tab_transformer_convs is not None:
                x = feature
                for conv in tab_transformer_convs:
                    x = conv(x)
                processed_features.append((x, feature_type))
            else:
                # 數值特徵保持不變
                processed_features.append((feature, feature_type))
        
        return processed_features

    # 計算MLP輸入維度
    # categorical 特徵展平：num_cat_cols * channels
    # numerical 特徵保持：num_numerical_features
    mlp_input_len = categorical_col_len * channels + numerical_col_len
    
    # MLP層的大小
    mlp_first_hidden_layer_size = 2 * mlp_input_len
    mlp_second_hidden_layer_size = 4 * mlp_input_len
    
    # 創建TabTransformer的解碼器 - MLP
    decoder = Sequential(
        Linear(mlp_input_len, mlp_first_hidden_layer_size),
        BatchNorm1d(mlp_first_hidden_layer_size),
        SELU(),
        Linear(mlp_first_hidden_layer_size, mlp_second_hidden_layer_size),
        BatchNorm1d(mlp_second_hidden_layer_size),
        SELU(),
        Linear(mlp_second_hidden_layer_size, out_channels)
    ).to(device)
    
    # 初始化解碼器參數
    for m in decoder:
        if not isinstance(m, SELU):
            m.reset_parameters()
    
    # 定義完整的前向傳播函數 (對齐 ExcelFormer，支持完整的 GNN 處理)
    def forward(tf, debug=False):
        batch_size = len(tf)
        if debug:
            print(f"[TabTransformer] Input batch_size: {batch_size}")
        
        # Stage 1: 編碼階段
        encoded_features = encode_batch(tf)
        if debug:
            print(f"[TabTransformer] After encoding: {len(encoded_features)} feature types")
            for i, (feature, feature_type) in enumerate(encoded_features):
                print(f"[TabTransformer]   {feature_type} features: {feature.shape}")
        
        # 對於 TabTransformer，我們主要在類別特徵上應用 Transformer
        # 將編碼的特徵轉換為統一的列級表示 [batch_size, num_cols, channels]
        # 為了支持 GNN，我們需要將數值特徵也轉換為同樣的形狀
        
        # 合併類別和數值特徵為統一的列級表示
        combined_features = []
        col_idx = 0
        for feature, feature_type in encoded_features:
            if feature_type == "categorical":
                # [batch_size, num_cat_cols, channels]
                combined_features.append(feature)
                col_idx += feature.shape[1]
            else:
                # 數值特徵 [batch_size, num_num_features]
                # 需要轉換為 [batch_size, num_num_features, channels]
                if feature.shape[1] > 0:
                    # 添加列維度並擴展: [batch_size, num_features, 1] -> [batch_size, num_features, channels]
                    feature_expanded = feature.unsqueeze(-1).expand(-1, -1, channels)
                    combined_features.append(feature_expanded)
                    col_idx += feature.shape[1]
        
        if combined_features:
            x = torch.cat(combined_features, dim=1)  # [batch_size, num_cols, channels]
        else:
            x = torch.zeros(batch_size, 1, channels, device=device)
        
        if debug:
            print(f"[TabTransformer] Combined features shape: {x.shape}")
        
        # ======================== 編碼後的 GNN 階段 ========================
        if gnn_stage == 'encoding' and dgm_module is not None:
            # 完整的 ExcelFormer Encoding 段實現
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
        
        # Stage 3: 列間交互階段 (Transformer)
        # 對於 encoding GNN，特徵已經增強，需要重新走 TabTransformerConv
        if gnn_stage == 'encoding':
            # 將增強後的特徵分解回 categorical 和 numerical 部分
            if categorical_col_len > 0 and numerical_col_len > 0:
                x_cat = x[:, :categorical_col_len, :]
                x_num_expanded = x[:, categorical_col_len:, :]
                x_num = x_num_expanded.mean(dim=-1)  # [batch, num_cols] -> [batch, num_num_features]
                processed_features = process_batch_interaction([(x_cat, "categorical"), (x_num, "numerical")])
            elif categorical_col_len > 0:
                processed_features = process_batch_interaction([(x, "categorical")])
            else:
                processed_features = [(x.mean(dim=-1), "numerical")]
        else:
            # none, start, materialize 階段：使用原始編碼特徵
            processed_features = process_batch_interaction(encoded_features)
        
        # ======================== 列間交互後的 GNN 階段 ========================
        if gnn_stage == 'columnwise' and dgm_module is not None:
            # 完整的 ExcelFormer Columnwise 段實現
            # Step 1: Self-Attention 列間交互
            tokens = x + column_embed.unsqueeze(0)
            tokens_norm = attn_norm(tokens)
            attn_out1, _ = self_attn(tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + attn_out1
            ffn_out1 = ffn_pre(attn_norm(tokens_attn))
            tokens_attn = tokens_attn + ffn_out1
            
            # Step 2: Attention Pooling
            pool_logits = (tokens_attn * pool_query).sum(dim=-1) / math.sqrt(channels)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)
            
            # Step 3: Mini-batch DGM 動態建圖
            x_pooled_std = _standardize(x_pooled, dim=0)
            x_pooled_batched = x_pooled_std.unsqueeze(0)
            if hasattr(dgm_module, 'k'):
                Ns_col = x_pooled_batched.shape[1]
                dgm_module.k = int(min(int(dgm_module.k), max(1, Ns_col - 1)))
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
            
            # columnwise GNN 後，將增強特徵分解回 categorical 和 numerical
            if categorical_col_len > 0 and numerical_col_len > 0:
                x_cat = x[:, :categorical_col_len, :]
                x_num_expanded = x[:, categorical_col_len:, :]
                x_num = x_num_expanded.mean(dim=-1)
                processed_features = [(x_cat, "categorical"), (x_num, "numerical")]
            elif categorical_col_len > 0:
                processed_features = [(x, "categorical")]
            else:
                processed_features = [(x.mean(dim=-1), "numerical")]
        
        # ======================== 特徵整合 ========================
        # 將處理後的特徵展平用於 decoder
        flat_features = []
        for feature, feature_type in processed_features:
            if feature_type == "categorical":
                # [batch, num_cat_cols, channels] -> [batch, num_cat_cols * channels]
                flat_features.append(feature.reshape(batch_size, -1))
            else:  # numerical
                # [batch, num_num_features]
                flat_features.append(feature)
        
        if flat_features:
            x_flat = torch.cat(flat_features, dim=1)
        else:
            x_flat = torch.zeros(batch_size, mlp_input_len, device=device)
        
        # ======================== 解碼階段 ========================
        if gnn_stage == 'decoding' and dgm_module is not None:
            # 對於 decoding，使用合併的特徵進行 GNN 處理
            # 先將特徵重新組合為列級表示
            if categorical_col_len > 0 and numerical_col_len > 0:
                x_cat = processed_features[0][0]  # [batch, cat_cols, channels]
                x_num = processed_features[1][0].unsqueeze(-1).expand(-1, -1, channels)  # [batch, num_cols, channels]
                x = torch.cat([x_cat, x_num], dim=1)
            elif categorical_col_len > 0:
                x = processed_features[0][0]
            else:
                x = processed_features[0][0].unsqueeze(-1).expand(-1, -1, channels)
            
            # Step 1: Self-Attention 列間交互
            tokens = x + column_embed.unsqueeze(0)
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
            return out
        else:
            # 使用原有 decoder - MLP
            out = decoder(x_flat)
            return out
        
    # 設置優化器和學習率調度器 (參考 ExcelFormer 風格)
    lr = config.get('lr', 0.0001)
    gamma = config.get('gamma', 0.95)
    
    # 修復：只收集實際使用的參數（對齐 ExcelFormer）
    all_params = []
    if cat_encoder is not None:
        all_params.extend(list(cat_encoder.parameters()))
    if pad_embedding is not None:
        all_params.extend(list(pad_embedding.parameters()))
    if num_encoder is not None:
        all_params.extend(list(num_encoder.parameters()))
    if num_norm is not None:
        all_params.extend(list(num_norm.parameters()))
    if tab_transformer_convs is not None:
        all_params.extend([p for conv in tab_transformer_convs for p in conv.parameters()])
    all_params.extend(list(decoder.parameters()))
    
    # 添加 GNN 相關參數（對齐 ExcelFormer）
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
    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    # 定義訓練函數 (對齐 ExcelFormer 風格)
    def train(epoch):
        # 設置所有模組為訓練模式
        if cat_encoder is not None:
            cat_encoder.train()
        if num_encoder is not None:
            num_encoder.train()
        if num_norm is not None:
            num_norm.train()
        if tab_transformer_convs is not None:
            for conv in tab_transformer_convs:
                conv.train()
        decoder.train()
        
        # 設置 GNN 相關模組為訓練模式（對齐 ExcelFormer）
        if gnn is not None:
            gnn.train()
        if dgm_module is not None:
            dgm_module.train()
        
        # encoding 階段組件
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
        
        # decoding 階段組件
        elif gnn_stage == 'decoding':
            if self_attn is not None:
                self_attn.train()
            if attn_norm is not None:
                attn_norm.train()
        
        # columnwise 階段組件
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
            debug = (epoch == 1 and first_batch)
            pred = forward(tf, debug=debug)
            first_batch = False
            
            if is_classification:
                loss = F.cross_entropy(pred, tf.y)
            else:
                loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
        
        return loss_accum / total_count
    
    # 定義測試函數 (對齐 ExcelFormer 風格)
    @torch.no_grad()
    def test(loader):
        # 設置所有模組為評估模式
        if cat_encoder is not None:
            cat_encoder.eval()
        if num_encoder is not None:
            num_encoder.eval()
        if num_norm is not None:
            num_norm.eval()
        if tab_transformer_convs is not None:
            for conv in tab_transformer_convs:
                conv.eval()
        decoder.eval()
        
        # 設置 GNN 相關模組為評估模式（對齐 ExcelFormer）
        if gnn is not None:
            gnn.eval()
        if dgm_module is not None:
            dgm_module.eval()
        
        # encoding 階段組件
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
        
        # decoding 階段組件
        elif gnn_stage == 'decoding':
            if self_attn is not None:
                self_attn.eval()
            if attn_norm is not None:
                attn_norm.eval()
        
        # columnwise 階段組件
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
    
    # 初始化最佳指標 (參考 ExcelFormer 風格)
    if is_classification:
        best_val_metric = 0
        best_test_metric = 0
    else:
        best_val_metric = float('inf')
        best_test_metric = float('inf')
    
    # 記錄訓練過程
    best_epoch = 0
    early_stop_counter = 0
    early_stop_epochs = 0
    train_losses = []
    train_metrics = []
    val_metrics = []
    
    # 訓練循環
    epochs = config.get('epochs', 200)
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        train_metric = test(train_loader)
        val_metric = test(val_loader)
        
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        
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
    gnn_early_stop_epochs = 0

    if gnn_stage == 'decoding':
        # 專門為 decoding 階段優化的配置
        config.update({
            'gnn_max_samples': 5000,     # 增加樣本數
            'gnn_knn': 5,               # 稍微增加 k 值
            'gnn_lr': 0.001,            # 提高學習率
            'gnn_epochs': 100,          # 適中的 epoch 數
            'gnn_patience': 10,         # 增加 patience
            'gnn_hidden': 128           # 增加隱藏層大小
        })
        print(f"[TabTransformer] Updated config for decoding: {config}")
        best_val_metric, test_metric, gnn_early_stop_epochs = gnn_decoding_eval(
            train_loader, val_loader, test_loader, config, task_type, metric_computer,
            cat_encoder, num_encoder, num_norm, pad_embedding, tab_transformer_convs
        ,device)
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
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'metric_computer': metric_computer,
        'metric': metric,
        'early_stop_epochs': early_stop_epochs,
        'gnn_early_stop_epochs': material_outputs.get('gnn_early_stop_epochs', 0),
    }


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    主函數：按順序調用四個階段函數
    """
    print("TabTransformer - 五階段執行")
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
            config.update({
                'gnn_max_samples': 3000,  # 每個分割的最大樣本數
                'gnn_knn': 3,            # 更小的 k 值
                'gnn_lr': 0.0001,        # 更小的學習率
                'gnn_epochs': 50,        # 更少的 epoch
                'gnn_patience': 5        # 更短的 patience
            })
        results=tabtransformer_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)
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
#  python main.py --dataset kaggle_Audit_Data --models tabtransformer --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models tabtransformer --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models tabtransformer --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models tabtransformer --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models tabtransformer --gnn_stages all --epochs 2