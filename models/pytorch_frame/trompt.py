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
import math

# GNN 相關類和函數
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


def gnn_decoding_eval(train_loader, val_loader, test_loader, config, task_type, metric_computer,
                      encoders, trompt_convs, trompt_decoder, x_prompt, num_layers, device):
    """
    TromPT 的 GNN decoding 評估函數 - 參考 ExcelFormer 模式
    """
    print("[TromPT GNN Decoder] Starting evaluation...")
    
    def get_all_embeddings_and_targets_trompt(loader, encoders, trompt_convs, x_prompt, num_layers):
        """提取所有 embeddings 和 targets"""
        all_embeds, all_targets = [], []
        
        # 設置為評估模式
        for encoder in encoders:
            encoder.eval()
        for conv in trompt_convs:
            conv.eval()
        
        with torch.no_grad():
            for tf in loader:
                tf = tf.to(device)
                batch_size = len(tf)
                
                # 編碼特徵 - 獲取所有層的輸出
                layer_outputs = []
                for encoder in encoders:
                    x, _ = encoder(tf)  # [batch_size, num_cols, channels]
                    layer_outputs.append(x)
                
                # 拓展提示向量以匹配批次大小
                x_prompt_batch = x_prompt.repeat(batch_size, 1, 1)
                
                # 通過 TromptConv 層進行處理
                prompts_outputs = []
                for i in range(num_layers):
                    x = layer_outputs[i]
                    if i == 0:
                        prompt = x_prompt_batch
                    else:
                        prompt = prompts_outputs[-1]
                    
                    updated_prompt = trompt_convs[i](x, prompt)
                    prompts_outputs.append(updated_prompt)
                
                # 使用最後一層的提示向量作為 embedding
                final_prompt = prompts_outputs[-1]  # [batch_size, num_prompts, channels]
                # 展平為 [batch_size, num_prompts * channels]
                embeddings = final_prompt.reshape(batch_size, -1)
                
                all_embeds.append(embeddings.cpu())
                all_targets.append(tf.y.cpu())
        
        if all_embeds:
            return torch.cat(all_embeds, dim=0), torch.cat(all_targets, dim=0)
        else:
            return torch.empty(0, 1), torch.empty(0)
    
    # 提取所有 embeddings
    train_emb, train_y = get_all_embeddings_and_targets_trompt(
        train_loader, encoders, trompt_convs, x_prompt, num_layers)
    val_emb, val_y = get_all_embeddings_and_targets_trompt(
        val_loader, encoders, trompt_convs, x_prompt, num_layers)
    test_emb, test_y = get_all_embeddings_and_targets_trompt(
        test_loader, encoders, trompt_convs, x_prompt, num_layers)
    
    # 合併所有數據
    all_emb = torch.cat([train_emb, val_emb, test_emb], dim=0)
    all_y = torch.cat([train_y, val_y, test_y], dim=0)
    
    print(f"[TromPT GNN Decoder] All embeddings shape: {all_emb.shape}, targets: {all_y.shape}")
    
    # 轉換為 DataFrame 並建立 GNN
    all_df = pd.DataFrame(all_emb.numpy())
    all_df['target'] = all_y.numpy()
    
    feature_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
    y = all_df['target'].values
    
    # 自動計算 num_classes
    if task_type in ['binclass', 'multiclass']:
        num_classes = len(pd.unique(y))
        print(f"[TromPT GNN Decoder] Detected num_classes: {num_classes}")
    else:
        num_classes = 1
    
    # 標籤處理
    if task_type == 'binclass':
        y = torch.tensor(y, dtype=torch.float32, device=device)
    elif task_type == 'multiclass':
        y = torch.tensor(y, dtype=torch.long, device=device)
    else:  # regression
        y = torch.tensor(y, dtype=torch.float32, device=device)
    
    # 建立 KNN 圖
    k = config.get('gnn_knn', 5)
    print(f"[TromPT GNN Decoder] Building KNN graph with k={k}")
    edge_index = knn_graph(x, k).to(device)
    
    # 數據分割 mask
    n_train = len(train_emb)
    n_val = len(val_emb)
    n_test = len(test_emb)
    N = n_train + n_val + n_test
    
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    
    # 創建 GNN 模型
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    in_dim = x.shape[1]
    out_dim = 1 if (task_type == 'regression' or task_type == "binclass") else num_classes
    
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    
    # 訓練 GNN
    gnn.train()
    best_val_metric = None
    best_test_metric = None
    best_val = -float('inf') if (task_type == "binclass" or task_type == "multiclass") else float('inf')
    is_binary_class = task_type == 'binclass'
    is_classification = task_type in ['binclass', 'multiclass']
    best_epoch = 0
    early_stop_counter = 0
    patience = config.get('gnn_patience', 10)
    
    print(f"[TromPT GNN Decoder] Training GNN for {gnn_epochs} epochs...")
    
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(x, edge_index)
        
        # 計算損失
        if task_type == 'binclass':
            loss = F.binary_cross_entropy_with_logits(out[train_mask].squeeze(), y[train_mask])
        elif task_type == 'multiclass':
            loss = F.cross_entropy(out[train_mask], y[train_mask])
        else:  # regression
            loss = F.mse_loss(out[train_mask].squeeze(), y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # 每10個epoch評估一次
        if (epoch + 1) % 10 == 0:
            gnn.eval()
            with torch.no_grad():
                pred = gnn(x, edge_index)
                
                # 計算驗證指標
                if is_binary_class:
                    val_probs = torch.sigmoid(pred[val_mask].squeeze())
                    val_metric = roc_auc_score(y[val_mask].cpu(), val_probs.cpu())
                    
                    test_probs = torch.sigmoid(pred[test_mask].squeeze())
                    test_metric = roc_auc_score(y[test_mask].cpu(), test_probs.cpu())
                elif is_classification:
                    val_pred = pred[val_mask].argmax(dim=1)
                    val_metric = (val_pred == y[val_mask]).float().mean().item()
                    
                    test_pred = pred[test_mask].argmax(dim=1)
                    test_metric = (test_pred == y[test_mask]).float().mean().item()
                else:
                    val_metric = F.mse_loss(pred[val_mask].squeeze(), y[val_mask]).sqrt().item()
                    test_metric = F.mse_loss(pred[test_mask].squeeze(), y[test_mask]).sqrt().item()
                
                # 更新最佳指標
                improved = (val_metric > best_val) if is_classification else (val_metric < best_val)
                if improved:
                    best_val = val_metric
                    best_val_metric = val_metric
                    best_test_metric = test_metric
                    best_epoch = epoch + 1
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                print(f'[TromPT GNN Decoder] Epoch {epoch + 1}: Loss: {loss:.4f}, '
                      f'Val Metric: {val_metric:.4f}, Test Metric: {test_metric:.4f}')
                
                # 早停檢查
                if early_stop_counter >= patience:
                    print(f"[TromPT GNN Decoder] Early stopping at epoch {epoch + 1}")
                    break
            
            gnn.train()
    
    print(f"[TromPT GNN Decoder] Best Val Metric: {best_val_metric:.4f}")
    print(f"[TromPT GNN Decoder] Best Test Metric: {best_test_metric:.4f}")
    
    return best_val_metric, best_test_metric, best_epoch


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
    
    # 根據 GNN 階段創建 GNN 模型 - 參考 ExcelFormer 模式
    gnn = None
    if gnn_stage in ['encoding', 'columnwise']:
        class SimpleGCN_INTERNAL(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, out_channels)
            def forward(self, x, edge_index):
                return torch.relu(self.conv1(x, edge_index))
        
        gnn = SimpleGCN_INTERNAL(channels, channels).to(device)
        print(f"[TromPT Core] Created GNN for {gnn_stage} stage")
    
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
            
            # Stage: encoding GNN 處理 - 參考 ExcelFormer 模式
            if gnn_stage == 'encoding' and gnn is not None and i == 0:  # 只在第一層應用 encoding GNN
                if x.dim() == 3 and batch_size > 1:
                    batch_size_inner, num_cols_inner, channels_inner = x.shape
                    # 參考 ExcelFormer: 重塑為 [batch_size * num_cols, channels]
                    x_reshape = x.view(-1, channels_inner)
                    
                    # 建立全連接邊 (每個批次內所有節點互連)
                    row = torch.arange(num_cols_inner).repeat(num_cols_inner, 1).view(-1)
                    col = torch.arange(num_cols_inner).repeat(1, num_cols_inner).view(-1)
                    edge_index_single = torch.stack([row, col], dim=0)
                    edge_index = []
                    for b in range(batch_size_inner):
                        offset = b * num_cols_inner
                        edge_index.append(edge_index_single + offset)
                    edge_index = torch.cat(edge_index, dim=1).to(x.device)
                    
                    # 應用 GNN
                    try:
                        x_gnn = gnn(x_reshape, edge_index)
                        x = x_gnn.view(batch_size_inner, num_cols_inner, channels_inner)
                        if debug:
                            print(f"[TromPT] After encoding GNN (layer {i}): {x.shape}")
                    except Exception as e:
                        if debug:
                            print(f"[TromPT] GNN failed for layer {i}: {e}")
            
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
            
            # Stage: columnwise GNN 處理 - 參考 ExcelFormer 模式  
            if gnn_stage == 'columnwise' and gnn is not None:
                # 對提示向量應用 GNN (將提示向量看作節點特徵)
                if updated_prompt.dim() == 3 and batch_size > 1:
                    batch_size_inner, num_prompts_inner, channels_inner = updated_prompt.shape
                    # 重塑為 [batch_size * num_prompts, channels]
                    prompt_reshape = updated_prompt.view(-1, channels_inner)
                    
                    # 建立全連接邊
                    row = torch.arange(num_prompts_inner).repeat(num_prompts_inner, 1).view(-1)
                    col = torch.arange(num_prompts_inner).repeat(1, num_prompts_inner).view(-1)
                    edge_index_single = torch.stack([row, col], dim=0)
                    edge_index = []
                    for b in range(batch_size_inner):
                        offset = b * num_prompts_inner
                        edge_index.append(edge_index_single + offset)
                    edge_index = torch.cat(edge_index, dim=1).to(updated_prompt.device)
                    
                    # 應用 GNN
                    try:
                        prompt_gnn = gnn(prompt_reshape, edge_index)
                        updated_prompt = prompt_gnn.view(batch_size_inner, num_prompts_inner, channels_inner)
                        if debug:
                            print(f"[TromPT] After columnwise GNN (layer {i}): {updated_prompt.shape}")
                    except Exception as e:
                        if debug:
                            print(f"[TromPT] Columnwise GNN failed for layer {i}: {e}")
            
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
        前向傳播 - 返回所有層預測結果的平均
        """
        stacked_out = forward_stacked(tf, debug=debug)
        return stacked_out.mean(dim=1)
    
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
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        for encoder in encoders:
            encoder.train()
        for conv in trompt_convs:
            conv.train()
        trompt_decoder.train()
        
        # 設置 GNN 為訓練模式
        if gnn is not None:
            gnn.train()
        
        loss_accum = total_count = 0
        first_batch = True
        
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            
            # 只在第一個epoch的第一個batch啟用調試
            debug = (epoch == 1 and first_batch and gnn_stage in ['encoding', 'columnwise'])
            
            # 使用分層前向傳播
            out = forward_stacked(tf, debug=debug)
            first_batch = False
            
            # 準備標籤和預測值
            batch_size = len(tf)
            # 展平為[batch_size * num_layers, out_channels]
            pred = out.view(-1, out_channels)
            # 對標籤進行重複以匹配每一層的預測
            y = tf.y.repeat_interleave(num_layers)
            
            # 計算多層邏輯損失
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
        
        # 設置 GNN 為評估模式
        if gnn is not None:
            gnn.eval()
        
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
    
    # 如果是 decoding 階段，使用 GNN decoding 評估
    if gnn_stage == 'decoding':
        print("[TromPT Core] Training standard TromPT for decoding evaluation...")
        
        # 訓練標準 TromPT 模型
        epochs = config.get('epochs', 300)
        patience = config.get('patience', 10)
        
        # 初始化最佳指標
        if is_classification:
            best_val_metric = 0
            best_test_metric = 0
        else:
            best_val_metric = float('inf')
            best_test_metric = float('inf')
        
        best_epoch = 0
        early_stop_counter = 0
        
        # 簡化的訓練循環
        for epoch in range(1, min(epochs, 20) + 1):  # 減少訓練輪數以便快速提取特徵
            train_loss = train(epoch)
            val_metric = test(val_loader)
            
            improved = (val_metric > best_val_metric) if is_classification else (val_metric < best_val_metric)
            if improved:
                best_val_metric = val_metric
                best_epoch = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            print(f'[TromPT Core] Epoch {epoch}: Train Loss: {train_loss:.4f}, Val {metric}: {val_metric:.4f}')
            
            lr_scheduler.step()
            
            if early_stop_counter >= patience:
                print(f"[TromPT Core] Early stopping at epoch {epoch}")
                break
        
        print(f"[TromPT Core] Standard model trained. Best Val: {best_val_metric:.4f}")
        
        # 使用訓練好的模型進行 GNN decoding 評估
        print("[TromPT Core] Starting GNN decoding evaluation...")
        
        # 為 decoding 階段優化配置
        config.update({
            'gnn_max_samples': 3000,
            'gnn_knn': 5,
            'gnn_lr': 0.001,
            'gnn_epochs': 100,
            'gnn_patience': 10,
            'gnn_hidden': 128
        })
        
        gnn_val_metric, gnn_test_metric, gnn_epochs = gnn_decoding_eval(
            train_loader, val_loader, test_loader, config, task_type, metric_computer,
            encoders, trompt_convs, trompt_decoder, x_prompt, num_layers, device
        )
        
        return {
            'train_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'best_val_metric': gnn_val_metric,
            'final_metric': gnn_val_metric,
            'best_test_metric': gnn_test_metric,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'is_classification': is_classification,
            'is_binary_class': is_binary_class,
            'metric_computer': metric_computer,
            'metric': metric,
            'early_stop_epochs': gnn_epochs,
            'gnn_early_stop_epochs': gnn_epochs,
            'model_type': 'trompt_decoding_gnn',
            'standard_val_metric': best_val_metric,
        }
    
    # 標準訓練流程（用於 none, encoding, columnwise 階段）
    else:
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
        epochs = config.get('epochs', 300)
        patience = config.get('patience', 10)
        early_stop_counter = 0
        best_epoch = 0
        
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
            else:
                early_stop_counter += 1
            
            print(f'[TromPT Core] Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
                  f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')
            
            # 學習率調整
            lr_scheduler.step()
            
            if early_stop_counter >= patience:
                print(f"[TromPT Core] Early stopping at epoch {epoch}")
                break
        
        print(f'[TromPT Core] Best Val {metric}: {best_val_metric:.4f}, '
              f'Best Test {metric}: {best_test_metric:.4f}')
        
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
            'early_stop_epochs': 0,
            'gnn_early_stop_epochs': 0,
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