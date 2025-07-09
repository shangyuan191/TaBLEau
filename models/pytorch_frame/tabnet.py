from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GLU, BatchNorm1d, Identity, Linear, Module, ModuleList

import torch_frame
from torch_frame import stype
from torch_frame.data import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    StackEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame.typing import NAStrategy
from torch_frame.datasets.yandex import Yandex

"""Reported (reproduced) results of of TabNet model in the original paper
https://arxiv.org/abs/1908.07442.

Forest Cover Type: 96.99 (96.53)
KDD Census Income: 95.5 (95.41)
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import ForestCoverType, KDDCensusIncome
from torch_frame.nn import TabNet
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

    return 




class TabNet(Module):
    r"""The TabNet model introduced in the
    `"TabNet: Attentive Interpretable Tabular Learning"
    <https://arxiv.org/abs/1908.07442>`_ paper.

    .. note::

        For an example of using TabNet, see `examples/tabnet.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        tabnet.py>`_.

    Args:
        out_channels (int): Output dimensionality
        num_layers (int): Number of TabNet layers.
        split_feat_channels (int): Dimensionality of feature channels.
        split_attn_channels (int): Dimensionality of attention channels.
        gamma (float): The gamma value for updating the prior for the attention
            mask.
        col_stats (Dict[str,Dict[torch_frame.data.stats.StatType,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[torch_frame.stype, List[str]]): A
            dictionary that maps :class:`~torch_frame.stype` to a list of
            column names. The column names are sorted based on the ordering
            that appear in :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`StackEncoder()` for
            numerical feature)
        num_shared_glu_layers (int): Number of GLU layers shared across the
            :obj:`num_layers` :class:`FeatureTransformer`s. (default: :obj:`2`)
        num_dependent_glu_layers (int, optional): Number of GLU layers to use
            in each of :obj:`num_layers` :class:`FeatureTransformer`s.
            (default: :obj:`2`)
        cat_emb_channels (int, optional): The categorical embedding
            dimensionality.
    """
    def __init__(
        self,
        out_channels: int,
        num_layers: int,
        split_feat_channels: int,
        split_attn_channels: int,
        gamma: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        num_shared_glu_layers: int = 2,
        num_dependent_glu_layers: int = 2,
        cat_emb_channels: int = 2,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.split_feat_channels = split_feat_channels
        self.split_attn_channels = split_attn_channels
        self.num_layers = num_layers
        self.gamma = gamma

        num_cols = sum([len(v) for v in col_names_dict.values()])
        # if there is no categorical feature, we just set cat_emb_channels to 1
        cat_emb_channels = (cat_emb_channels if torch_frame.categorical
                            in col_names_dict else 1)
        in_channels = cat_emb_channels * num_cols

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical:
                EmbeddingEncoder(na_strategy=NAStrategy.MOST_FREQUENT),
                stype.numerical:
                StackEncoder(na_strategy=NAStrategy.MEAN),
            }

        # Map input tensor frame into (batch_size, num_cols, cat_emb_channels),
        # which is flattened into (batch_size, in_channels)
        self.feature_encoder = StypeWiseFeatureEncoder(
            out_channels=cat_emb_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        # Batch norm applied to input feature.
        self.bn = BatchNorm1d(in_channels)

        shared_glu_block: Module
        if num_shared_glu_layers > 0:
            shared_glu_block = GLUBlock(
                in_channels=in_channels,
                out_channels=split_feat_channels + split_attn_channels,
                no_first_residual=True,
            )
        else:
            shared_glu_block = Identity()

        self.feat_transformers = ModuleList()
        for _ in range(self.num_layers + 1):
            self.feat_transformers.append(
                FeatureTransformer(
                    in_channels,
                    split_feat_channels + split_attn_channels,
                    num_dependent_glu_layers=num_dependent_glu_layers,
                    shared_glu_block=shared_glu_block,
                ))

        self.attn_transformers = ModuleList()
        for _ in range(self.num_layers):
            self.attn_transformers.append(
                AttentiveTransformer(
                    in_channels=split_attn_channels,
                    out_channels=in_channels,
                ))

        self.lin = Linear(self.split_feat_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.feature_encoder.reset_parameters()
        self.bn.reset_parameters()
        for feat_transformer in self.feat_transformers:
            feat_transformer.reset_parameters()
        for attn_transformer in self.attn_transformers:
            attn_transformer.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        tf: TensorFrame,
        return_reg: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Transform :class:`TensorFrame` object into output embeddings.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.
            return_reg (bool): Whether to return the entropy regularization.

        Returns:
            Union[torch.Tensor, (torch.Tensor, torch.Tensor)]: The output
                embeddings of size :obj:`[batch_size, out_channels]`.
                If :obj:`return_reg` is :obj:`True`, return the entropy
                regularization as well.
        """
        # [batch_size, num_cols, cat_emb_channels]
        x, _ = self.feature_encoder(tf)
        batch_size = x.shape[0]
        # [batch_size, num_cols * cat_emb_channels]
        x = x.view(batch_size, math.prod(x.shape[1:]))
        x = self.bn(x)

        # [batch_size, num_cols * cat_emb_channels]
        prior = torch.ones_like(x)
        reg = torch.tensor(0., device=x.device)

        # [batch_size, split_attn_channels]
        attention_x = self.feat_transformers[0](x)
        attention_x = attention_x[:, self.split_feat_channels:]

        outs = []
        for i in range(self.num_layers):
            # [batch_size, num_cols * cat_emb_channels]
            attention_mask = self.attn_transformers[i](attention_x, prior)

            # [batch_size, num_cols * cat_emb_channels]
            masked_x = attention_mask * x
            # [batch_size, split_feat_channels + split_attn_channel]
            out = self.feat_transformers[i + 1](masked_x)

            # Get the split feature
            # [batch_size, split_feat_channels]
            feature_x = F.relu(out[:, :self.split_feat_channels])
            outs.append(feature_x)
            # Get the split attention
            # [batch_size, split_attn_channels]
            attention_x = out[:, self.split_feat_channels:]

            # Update prior
            prior = (self.gamma - attention_mask) * prior

            # Compute entropy regularization
            if return_reg and batch_size > 0:
                entropy = -torch.sum(
                    attention_mask * torch.log(attention_mask + 1e-15),
                    dim=1).mean()
                reg += entropy

        out = sum(outs)
        out = self.lin(out)

        if return_reg:
            return out, reg / self.num_layers
        else:
            return out


class FeatureTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dependent_glu_layers: int,
        shared_glu_block: Module,
    ) -> None:
        super().__init__()

        self.shared_glu_block = shared_glu_block

        self.dependent: Module
        if num_dependent_glu_layers == 0:
            self.dependent = Identity()
        else:
            if not isinstance(self.shared_glu_block, Identity):
                in_channels = out_channels
                no_first_residual = False
            else:
                no_first_residual = True
            self.dependent = GLUBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                no_first_residual=no_first_residual,
                num_glu_layers=num_dependent_glu_layers,
            )
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.shared_glu_block(x)
        x = self.dependent(x)
        return x

    def reset_parameters(self) -> None:
        if not isinstance(self.shared_glu_block, Identity):
            self.shared_glu_block.reset_parameters()
        if not isinstance(self.dependent, Identity):
            self.dependent.reset_parameters()


class GLUBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_glu_layers: int = 2,
        no_first_residual: bool = False,
    ) -> None:
        super().__init__()
        self.no_first_residual = no_first_residual
        self.glu_layers = ModuleList()

        for i in range(num_glu_layers):
            if i == 0:
                glu_layer = GLULayer(in_channels, out_channels)
            else:
                glu_layer = GLULayer(out_channels, out_channels)
            self.glu_layers.append(glu_layer)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for i, glu_layer in enumerate(self.glu_layers):
            if self.no_first_residual and i == 0:
                x = glu_layer(x)
            else:
                x = x * math.sqrt(0.5) + glu_layer(x)
        return x

    def reset_parameters(self) -> None:
        for glu_layer in self.glu_layers:
            glu_layer.reset_parameters()


class GLULayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lin = Linear(in_channels, out_channels * 2, bias=False)
        self.glu = GLU()
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        return self.glu(x)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()


class AttentiveTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bn = GhostBatchNorm1d(out_channels)
        self.reset_parameters()

    def forward(self, x: Tensor, prior: Tensor) -> Tensor:
        x = self.lin(x)
        x = self.bn(x)
        x = prior * x
        # Using softmax instead of sparsemax since softmax performs better.
        x = F.softmax(x, dim=-1)
        return x

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.bn.reset_parameters()


class GhostBatchNorm1d(torch.nn.Module):
    r"""Ghost Batch Normalization https://arxiv.org/abs/1705.08741."""
    def __init__(
        self,
        input_dim: int,
        virtual_batch_size: int = 512,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        if len(x) > 0:
            num_chunks = math.ceil(len(x) / self.virtual_batch_size)
            chunks = torch.chunk(x, num_chunks, dim=0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

    def reset_parameters(self) -> None:
        self.bn.reset_parameters()



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

    batch_size = config.get('batch_size', 4096)  # TabNet通常使用較大的批次
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
def encoding_fn(material_outputs, config):
    """
    階段2: Encoding - 創建特徵編碼器並對特徵進行初始處理
    
    輸入:
    - material_outputs: materialize_fn的輸出
    - config: 配置參數
    
    輸出:
    - 包含編碼器和處理後特徵的字典，可傳給columnwise_fn或自定義GNN
    """
    print("Executing encoding_fn")
    
    # 從上一階段獲取數據
    train_tensor_frame = material_outputs['train_tensor_frame']
    val_tensor_frame = material_outputs['val_tensor_frame']
    test_tensor_frame = material_outputs['test_tensor_frame']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    col_stats = material_outputs['col_stats']
    col_names_dict = train_tensor_frame.col_names_dict
    device = material_outputs['device']
    out_channels = material_outputs['out_channels']
    
    # 獲取TabNet的參數
    cat_emb_channels = config.get('cat_emb_channels', 2)
    print(f"Encoding with cat_emb_channels: {cat_emb_channels}")
    
    # 設置編碼器字典
    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(na_strategy=NAStrategy.MOST_FREQUENT),
        stype.numerical: StackEncoder(na_strategy=NAStrategy.MEAN),
    }
    
    # 創建TabNet的特徵編碼器
    feature_encoder = StypeWiseFeatureEncoder(
        out_channels=cat_emb_channels,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
    
    # 計算輸入通道數
    num_cols = sum([len(v) for v in col_names_dict.values()])
    in_channels = cat_emb_channels * num_cols
    print(f"Input channels: {in_channels}")
    
    # 創建批次標準化層
    bn = BatchNorm1d(in_channels).to(device)
    
    # 對批次數據進行編碼的函數
    def encode_batch(tf):
        # 編碼特徵
        x, _ = feature_encoder(tf)
        batch_size = x.shape[0]
        # 展平特徵
        x_flat = x.view(batch_size, math.prod(x.shape[1:]))
        # 應用批次標準化
        x_norm = bn(x_flat)
        return x_norm
    
    # 處理樣本數據以便檢查形狀
    with torch.no_grad():
        if len(train_loader) > 0:
            sample_tf = next(iter(train_loader))
            sample_tf = sample_tf.to(device)
            sample_encoded = encode_batch(sample_tf)
            print(f"Encoded shape: {sample_encoded.shape}")
    
    # 返回編碼器和相關信息
    return {
        'feature_encoder': feature_encoder,
        'bn': bn,
        'encode_batch': encode_batch,
        'in_channels': in_channels,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_tensor_frame': train_tensor_frame,
        'val_tensor_frame': val_tensor_frame,
        'test_tensor_frame': test_tensor_frame,
        'cat_emb_channels': cat_emb_channels,
        'out_channels': out_channels,
        'is_classification': material_outputs['is_classification'],
        'is_binary_class': material_outputs['is_binary_class'],
        'metric_computer': material_outputs['metric_computer'],
        'metric': material_outputs['metric'],
        'device': device
    }
def columnwise_fn(encoding_outputs, config):
    """
    階段3: Column-wise Interaction - 創建TabNet的特徵和注意力變換器
    
    輸入:
    - encoding_outputs: encoding_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 包含特徵變換器和注意力變換器的字典，可傳給decoding_fn或自定義GNN
    """
    print("Executing columnwise_fn")
    
    # 從上一階段獲取數據
    feature_encoder = encoding_outputs['feature_encoder']
    bn = encoding_outputs['bn']
    encode_batch = encoding_outputs['encode_batch']
    in_channels = encoding_outputs['in_channels']
    device = encoding_outputs['device']
    
    # 獲取TabNet的參數
    split_feat_channels = config.get('channels', 128)
    split_attn_channels = config.get('channels', 128)
    num_layers = config.get('num_layers', 6)
    gamma = config.get('gamma', 1.2)
    num_shared_glu_layers = config.get('num_shared_glu_layers', 2)
    num_dependent_glu_layers = config.get('num_dependent_glu_layers', 2)
    
    print(f"Building TabNet with {num_layers} layers, split channels: {split_feat_channels}")
    
    # 創建共享的GLU塊
    shared_glu_block = None
    if num_shared_glu_layers > 0:
        shared_glu_block = GLUBlock(
            in_channels=in_channels,
            out_channels=split_feat_channels + split_attn_channels,
            num_glu_layers=num_shared_glu_layers,
            no_first_residual=True,
        ).to(device)
    else:
        shared_glu_block = Identity().to(device)
    
    # 創建特徵變換器
    feat_transformers = ModuleList([
        FeatureTransformer(
            in_channels,
            split_feat_channels + split_attn_channels,
            num_dependent_glu_layers=num_dependent_glu_layers,
            shared_glu_block=shared_glu_block,
        ).to(device)
        for _ in range(num_layers + 1)
    ])
    
    # 創建注意力變換器
    attn_transformers = ModuleList([
        AttentiveTransformer(
            in_channels=split_attn_channels,
            out_channels=in_channels,
        ).to(device)
        for _ in range(num_layers)
    ])
    
    # 定義列間交互處理函數
    def process_batch_interaction(x, return_reg=False):
        """
        處理一個批次的特徵通過特徵變換器和注意力變換器
        
        參數:
        - x: 編碼後的特徵張量 [batch_size, in_channels]
        - return_reg: 是否返回熵正則化
        
        返回:
        - feature_outputs: 列表，包含每層的輸出特徵
        - reg: 熵正則化值 (如果return_reg=True)
        """
        batch_size = x.shape[0]
        prior = torch.ones_like(x)
        reg = torch.tensor(0., device=x.device)
        
        # 第一層特徵變換器處理，分離注意力部分
        attention_x = feat_transformers[0](x)
        attention_x = attention_x[:, split_feat_channels:]
        
        feature_outputs = []
        for i in range(num_layers):
            # 應用注意力變換器生成掩碼
            attention_mask = attn_transformers[i](attention_x, prior)
            
            # 應用掩碼到輸入特徵
            masked_x = attention_mask * x
            
            # 應用特徵變換器
            out = feat_transformers[i + 1](masked_x)
            
            # 分離特徵和注意力部分
            feature_x = F.relu(out[:, :split_feat_channels])
            feature_outputs.append(feature_x)
            attention_x = out[:, split_feat_channels:]
            
            # 更新prior
            prior = (gamma - attention_mask) * prior
            
            # 計算熵正則化
            if return_reg and batch_size > 0:
                entropy = -torch.sum(
                    attention_mask * torch.log(attention_mask + 1e-15),
                    dim=1).mean()
                reg += entropy
        
        if return_reg:
            return feature_outputs, reg / num_layers
        else:
            return feature_outputs
    
    # 返回特徵變換器、注意力變換器和相關信息
    return {
        'feature_encoder': feature_encoder,
        'bn': bn,
        'encode_batch': encode_batch,
        'feat_transformers': feat_transformers,
        'attn_transformers': attn_transformers,
        'process_batch_interaction': process_batch_interaction,
        'shared_glu_block': shared_glu_block,
        'in_channels': in_channels,
        'split_feat_channels': split_feat_channels,
        'split_attn_channels': split_attn_channels,
        'gamma': gamma,
        'num_layers': num_layers,
        'train_loader': encoding_outputs['train_loader'],
        'val_loader': encoding_outputs['val_loader'],
        'test_loader': encoding_outputs['test_loader'],
        'out_channels': encoding_outputs['out_channels'],
        'is_classification': encoding_outputs['is_classification'],
        'is_binary_class': encoding_outputs['is_binary_class'],
        'metric_computer': encoding_outputs['metric_computer'],
        'metric': encoding_outputs['metric'],
        'device': device
    }

def decoding_fn(columnwise_outputs, config):
    """
    階段4: Decoding - 創建輸出層並訓練模型
    
    輸入:
    - columnwise_outputs: columnwise_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 訓練結果和最終模型
    """
    print("Executing decoding_fn")
    
    # 從上一階段獲取數據
    feature_encoder = columnwise_outputs['feature_encoder']
    bn = columnwise_outputs['bn']
    encode_batch = columnwise_outputs['encode_batch']
    feat_transformers = columnwise_outputs['feat_transformers']
    attn_transformers = columnwise_outputs['attn_transformers']
    process_batch_interaction = columnwise_outputs['process_batch_interaction']
    split_feat_channels = columnwise_outputs['split_feat_channels']
    num_layers = columnwise_outputs['num_layers']
    train_loader = columnwise_outputs['train_loader']
    val_loader = columnwise_outputs['val_loader']
    test_loader = columnwise_outputs['test_loader']
    out_channels = columnwise_outputs['out_channels']
    device = columnwise_outputs['device']
    is_classification = columnwise_outputs['is_classification']
    is_binary_class = columnwise_outputs['is_binary_class']
    metric_computer = columnwise_outputs['metric_computer']
    metric = columnwise_outputs['metric']
    
    # 創建輸出層
    lin = Linear(split_feat_channels, out_channels).to(device)
    
    # 重置參數
    lin.reset_parameters()
    
    # 定義完整的TabNet前向傳播函數
    def forward(tf, return_reg=False):
        # 編碼特徵
        x = encode_batch(tf)
        
        # 通過特徵變換器和注意力變換器處理
        if return_reg:
            feature_outputs, reg = process_batch_interaction(x, return_reg=True)
        else:
            feature_outputs = process_batch_interaction(x, return_reg=False)
        
        # 合併所有層的特徵輸出
        out = sum(feature_outputs)
        
        # 應用輸出層
        out = lin(out)
        
        if return_reg:
            return out, reg
        else:
            return out
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.005)  # TabNet默認學習率
    
    # 收集所有參數
    all_params = list(feature_encoder.parameters()) + \
                 list(bn.parameters()) + \
                 [p for ft in feat_transformers for p in ft.parameters()] + \
                 [p for at in attn_transformers for p in at.parameters()] + \
                 list(lin.parameters())
    
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        feature_encoder.train()
        bn.train()
        for ft in feat_transformers:
            ft.train()
        for at in attn_transformers:
            at.train()
        lin.train()
        
        loss_accum = total_count = 0
        
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            pred, reg = forward(tf, return_reg=True)
            
            if is_classification:
                loss = F.cross_entropy(pred, tf.y)
            else:
                loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
            
            # 添加熵正則化
            loss = loss + 0.01 * reg  # 熵正則化係數
            
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
        
        return loss_accum / total_count
    
    # 定義測試函數
    @torch.no_grad()
    def test(loader):
        # 設置為評估模式
        feature_encoder.eval()
        bn.eval()
        for ft in feat_transformers:
            ft.eval()
        for at in attn_transformers:
            at.eval()
        lin.eval()
        
        metric_computer.reset()
        
        for tf in loader:
            tf = tf.to(device)
            pred = forward(tf)
            
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
    
    # 記錄訓練過程
    train_losses = []
    train_metrics = []
    val_metrics = []
    test_metrics = []
    
    # 訓練循環
    epochs = config.get('epochs', 50)
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        train_metric = test(train_loader)
        val_metric = test(val_loader)
        test_metric = test(test_loader)
        
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        test_metrics.append(test_metric)
        
        if is_classification and val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        elif not is_classification and val_metric < best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        
        print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
              f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')
        
        # 學習率調整
        lr_scheduler.step()
    
    print(f'Best Val {metric}: {best_val_metric:.4f}, '
          f'Best Test {metric}: {best_test_metric:.4f}')
    
    # 返回訓練結果和模型組件
    return {
        'train_losses': train_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'feature_encoder': feature_encoder,
        'bn': bn,
        'feat_transformers': feat_transformers,
        'attn_transformers': attn_transformers,
        'lin': lin,
        'forward': forward  # 返回完整模型的前向函數
    }
def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    主函數：按順序調用四個階段函數
    """
    print("TabNet - 五階段執行")
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    try:
        # 階段0: 開始
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        if gnn_stage=='start':
            # 在 start_fn 和 materialize_fn 之間插入 GNN
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)
        # 階段1: Materialization
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
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
                'mutual_info_sort': mutual_info_sort,
                'dataset': dataset,
                'train_tensor_frame': train_tensor_frame,
                'val_tensor_frame': val_tensor_frame,
                'test_tensor_frame': test_tensor_frame,
            })
            material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs

        results=excelformer_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)
    
        # # 階段2: Encoding
        # encoding_outputs = encoding_fn(material_outputs, config)
        # # 這裡可以插入GNN處理編碼後的數據
        # # encoding_outputs = gnn_process(encoding_outputs, config)
        # # 階段3: Column-wise Interaction
        # columnwise_outputs = columnwise_fn(encoding_outputs, config)
        # # 這裡可以插入GNN處理列間交互後的數據
        # # columnwise_outputs = gnn_process(columnwise_outputs, config)
        # # 階段4: Decoding
        # results = decoding_fn(columnwise_outputs, config)
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