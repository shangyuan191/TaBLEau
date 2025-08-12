from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import FTTransformerConvs
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

"""Reported (reproduced) results of FT-Transformer
https://arxiv.org/abs/2106.11959.

adult 85.9 (85.5)
helena 39.1 (39.2)
jannis 73.2 (72.2)
california_housing 0.459 (0.537)
--------
Reported (reproduced) results of ResNet
https://arxiv.org/abs/2106.11959

adult 85.7 (85.4)
helena 39.6 (39.1)
jannis 72.8 (72.5)
california_housing 0.486 (0.523)
"""
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from torch_frame.datasets import Yandex
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    ResNet,
)

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


    numerical_encoder_type = 'linear'
    model_type = 'fttransformer'
    channels = config.get('channels', 256)
    num_layers = config.get('num_layers', 4)

    # 7. Yandex 數據集包裝
    dataset = Yandex(train_df_gnn, val_df_gnn, test_df_gnn, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification
    # 8. split tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]
    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    if numerical_encoder_type == 'linear':
        numerical_encoder = LinearEncoder()
    elif numerical_encoder_type == 'linearbucket':
        numerical_encoder = LinearBucketEncoder()
    elif numerical_encoder_type == 'linearperiodic':
        numerical_encoder = LinearPeriodicEncoder()
    else:
        raise ValueError(f'Unsupported encoder type: {numerical_encoder_type}')

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: numerical_encoder,
    }

    if is_classification:
        output_channels = dataset.num_classes
    else:
        output_channels = 1

    is_binary_class = is_classification and output_channels == 2
    if is_binary_class:
        metric_computer = AUROC(task='binary')
        metric = 'AUC'
    elif is_classification:
        metric_computer = Accuracy(task='multiclass', num_classes=output_channels)
        metric = 'Acc'
    else:
        metric_computer = MeanSquaredError()
        metric = 'RMSE'

    metric_computer = metric_computer.to(device)
    return (train_loader, 
            val_loader, 
            test_loader, 
            dataset.col_stats, 
            stype_encoder_dict, 
            dataset, 
            train_tensor_frame, 
            val_tensor_frame, 
            test_tensor_frame,
            gnn_early_stop_epochs)


# 取得所有 row 的 embedding
def get_all_embeddings_and_targets(loader, encoder, backbone):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_embeds, all_targets = [], []
    for tf in loader:
        tf = tf.to(device)
        x, _ = encoder(tf)
        x, x_cls = backbone(x)
        x_pooled = x.mean(dim=2)  # (batch_size, num_cols)
        all_embeds.append(x_pooled.detach().cpu())
        all_targets.append(tf.y.detach().cpu())
    return torch.cat(all_embeds, dim=0), torch.cat(all_targets, dim=0)
def gnn_decoding_eval(train_loader, val_loader, test_loader, config, task_type, metric_computer, encoder, backbone):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_emb, train_y = get_all_embeddings_and_targets(train_loader, encoder, backbone)
    val_emb, val_y = get_all_embeddings_and_targets(val_loader, encoder, backbone)
    test_emb, test_y = get_all_embeddings_and_targets(test_loader, encoder, backbone)
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
    numerical_encoder_type = 'linear'
    model_type = 'fttransformer'
    channels = config.get('channels', 256)
    num_layers = config.get('num_layers', 4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 數據集包裝（直接合併三份 DataFrame，標記 split_col）
    dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    # 根據 split_col 取得三份 tensor_frame
    train_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 0]
    val_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 1]
    test_tensor_frame = dataset.tensor_frame[dataset.df['split_col'] == 2]

    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    if numerical_encoder_type == 'linear':
        numerical_encoder = LinearEncoder()
    elif numerical_encoder_type == 'linearbucket':
        numerical_encoder = LinearBucketEncoder()
    elif numerical_encoder_type == 'linearperiodic':
        numerical_encoder = LinearPeriodicEncoder()
    else:
        raise ValueError(f'Unsupported encoder type: {numerical_encoder_type}')

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: numerical_encoder,
    }

    if is_classification:
        output_channels = dataset.num_classes
    else:
        output_channels = 1

    is_binary_class = is_classification and output_channels == 2
    if is_binary_class:
        metric_computer = AUROC(task='binary')
        metric = 'AUC'
    elif is_classification:
        metric_computer = Accuracy(task='multiclass', num_classes=output_channels)
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
        'col_stats': dataset.col_stats,
        'metric_computer': metric_computer,
        'metric': metric,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'out_channels': output_channels,
        'stype_encoder_dict': stype_encoder_dict,
        'device': device,
        'channels': channels,
        'num_layers': num_layers,
    }




def fttransformer_core_fn(material_outputs, config, task_type, gnn_stage=None):
    """
    FTTransformer核心函數：將materialize_fn、encoding_fn、columnwise_fn和decoding_fn整合
    """
    print("Executing fttransformer_core_fn")
    # 從上一階段獲取數據
    train_tensor_frame = material_outputs['train_tensor_frame']
    val_tensor_frame = material_outputs['val_tensor_frame']
    test_tensor_frame = material_outputs['test_tensor_frame']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    col_stats = material_outputs['col_stats']
    device = material_outputs['device']
    stype_encoder_dict = material_outputs['stype_encoder_dict']
    # 獲取模型參數
    channels = config.get('channels', 256)
    num_layers = config.get('num_layers', 4)
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
    # 創建FTTransformer的編碼器部分
    encoder = StypeWiseFeatureEncoder(
        out_channels=channels,
        col_stats=col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)

    # 創建FTTransformer的列間交互層 - FTTransformerConvs
    backbone = FTTransformerConvs(
        channels=channels,
        num_layers=num_layers
    ).to(device)
    is_classification = material_outputs['is_classification']
    is_binary_class = material_outputs['is_binary_class']
    out_channels = material_outputs['out_channels']
    metric_computer = material_outputs['metric_computer']
    metric = material_outputs['metric']
    decoder = Sequential(
        LayerNorm(channels),
        ReLU(),
        Linear(channels, out_channels),
    ).to(device)

    # 初始化解碼器參數
    for m in decoder:
        if not isinstance(m, ReLU):
            m.reset_parameters()
    
    # 完整模型的前向傳播函數
    def model_forward(tf):
        x, _ = encoder(tf)
        batch_size, num_cols, channels_ = x.shape
        # print(f"Input x shape: {x.shape}, batch_size: {batch_size}, num_cols: {num_cols}, channels_: {channels_}")
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
            # print(f"gnn_stage = encoding, Input x shape: {x.shape}")
        # 列間交互階段
        x, x_cls = backbone(x)
        # print(f"After backbone, x shape: {x.shape}, x_cls shape: {x_cls.shape}")
        if gnn_stage == 'columnwise':
            # x_reshape = x.view(-1, channels_)
            # row = torch.arange(num_cols).repeat(num_cols, 1).view(-1)
            # col = torch.arange(num_cols).repeat(1, num_cols).view(-1)
            # edge_index_single = torch.stack([row, col], dim=0)
            # edge_index = []
            # for i in range(batch_size):
            #     offset = i * num_cols
            #     edge_index.append(edge_index_single + offset)
            # edge_index = torch.cat(edge_index, dim=1).to(x.device)
            # x = gnn(x_reshape, edge_index).view(batch_size, num_cols, channels_)
            x=x

        # 解碼階段
        out = decoder(x_cls)
        return out
    
    # 設置優化器和學習率
    lr = config.get('lr', 0.001)
    all_params = list(encoder.parameters()) + list(gnn.parameters()) + list(backbone.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        encoder.train()
        backbone.train()
        decoder.train()
        
        loss_accum = total_count = 0
        
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            pred = model_forward(tf)
            
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
    
    # 定義測試函數
    @torch.no_grad()
    def test(loader):
        # 設置為評估模式
        encoder.eval()
        backbone.eval()
        decoder.eval()
        
        metric_computer.reset()
        
        for tf in loader:
            tf = tf.to(device)
            pred = model_forward(tf)
            
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
    best_epoch = 0
    early_stop_counter = 0
    
    # 訓練循環
    epochs = config.get('epochs', 200)
    early_stop_epochs = 0
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

        if early_stop_counter >= patience:
            early_stop_epochs = epoch
            print(f"Early stopping at epoch {epoch}")
            break
    # 決定最終 metric 輸出
    final_metric = None
    test_metric = None
    if gnn_stage == 'decoding':
        # 確保gnn在gnn_decoding_eval作用域可見
        best_val_metric, test_metric, gnn_early_stop_epochs = gnn_decoding_eval(train_loader, val_loader, test_loader, config, task_type, metric_computer, encoder, backbone)
        final_metric = best_val_metric
    else:
        final_metric = best_val_metric
        print(f'Best Val {metric}: {final_metric:.4f}')
        test_metric = test(test_loader)

    # 返回訓練結果
    return {
        'train_losses': train_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_metric': best_val_metric,
        'final_metric': final_metric,
        'best_test_metric': test_metric,
        'encoder': encoder,
        'backbone': backbone,
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
    """
    主函數：按順序調用四個階段函數
    
    可用於在階段間插入GNN模型
    """
    print("FTTransformer - 四階段執行")
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    try:
        # 階段0: 開始
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        gnn_early_stop_epochs=0
        if gnn_stage=="start":
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)

        
        # 階段1: Materialization
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        if gnn_stage == 'materialize':
            # 在 materialize_fn 和 encoding_fn 之間插入 GNN
            train_tensor_frame = material_outputs['train_tensor_frame']
            val_tensor_frame = material_outputs['val_tensor_frame']
            test_tensor_frame = material_outputs['test_tensor_frame']
            (train_loader, val_loader, test_loader,col_stats, stype_encoder_dict, dataset, train_tensor_frame, val_tensor_frame, test_tensor_frame, gnn_early_stop_epochs) = gnn_after_materialize_fn(
                train_tensor_frame, val_tensor_frame, test_tensor_frame, config, dataset_results['dataset'], task_type)
            material_outputs.update({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'col_stats': col_stats,
                'stype_encoder_dict': stype_encoder_dict,
                'dataset': dataset,
                'train_tensor_frame': train_tensor_frame,
                'val_tensor_frame': val_tensor_frame,
                'test_tensor_frame': test_tensor_frame,
            })
            material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        results=fttransformer_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)


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


#  python main.py --dataset kaggle_Audit_Data --models fttransformer --gnn_stages all --epochs 2
#  python main.py --dataset eye --models fttransformer --gnn_stages all --epochs 2
#  python main.py --dataset house --models fttransformer --gnn_stages all --epochs 2
#  python main.py --dataset credit --models fttransformer --gnn_stages all --epochs 2
#  python main.py --dataset openml_The_Office_Dataset --models fttransformer --gnn_stages all --epochs 2