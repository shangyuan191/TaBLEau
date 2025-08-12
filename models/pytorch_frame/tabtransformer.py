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
class TabTransformer(Module):
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
        'device': device
    }
def tabtransformer_core_fn(material_outputs, config, task_type, gnn_stage=None):
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
    
    # 修復：有條件地創建 GNN 和投影層
    gnn = None
    num_to_gnn_proj = None
    gnn_to_num_proj = None
    
    if gnn_stage in ['encoding', 'columnwise']:
        from torch_geometric.nn import GCNConv
        import torch.nn as nn
        
        class SimpleGCN_INTERNAL(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, out_channels)
            def forward(self, x, edge_index):
                return torch.relu(self.conv1(x, edge_index))
        
        gnn = SimpleGCN_INTERNAL(channels, channels).to(device)
        
        # 為數值特徵創建持久的投影層
        if numerical_col_len > 0:
            num_to_gnn_proj = torch.nn.Linear(numerical_col_len, channels).to(device)
            gnn_to_num_proj = torch.nn.Linear(channels, numerical_col_len).to(device)
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
            # 僅對類別特徵應用TabTransformerConv
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
    
    # 定義完整的前向傳播函數 (參考 ExcelFormer 風格，添加 GNN 處理)
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
        
            # 修復：正確的 GNN 處理邏輯
        def apply_gnn_to_features(features, stage_name):
            if gnn is None:
                return features
                
            processed = []
            for feature, feature_type in features:
                if feature_type == "categorical" and feature.dim() == 3:
                    # 類別特徵 GNN 處理
                    batch_size_inner, num_cols, channels_inner = feature.shape
                    if batch_size_inner > 1:
                        # 創建批次內的全連接圖
                        edge_index = []
                        for i in range(batch_size_inner):
                            for j in range(batch_size_inner):
                                if i != j:
                                    edge_index.append([i, j])
                        
                        if edge_index:
                            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(feature.device)
                            # 對每個 column 分別應用 GNN
                            feature_gnn_list = []
                            for col_idx in range(num_cols):
                                col_feature = feature[:, col_idx, :]  # [batch_size, channels]
                                try:
                                    col_feature_gnn = gnn(col_feature, edge_index)
                                    feature_gnn_list.append(col_feature_gnn.unsqueeze(1))
                                except Exception as e:
                                    if debug:
                                        print(f"[TabTransformer] GNN failed for categorical column {col_idx}: {e}")
                                    feature_gnn_list.append(col_feature.unsqueeze(1))
                            feature = torch.cat(feature_gnn_list, dim=1)
                        
                        if debug:
                            print(f"[TabTransformer] After {stage_name} GNN ({feature_type}): {feature.shape}")
                            
                elif feature_type == "numerical" and feature.dim() == 2:
                    # 數值特徵 GNN 處理
                    batch_size_inner, num_features = feature.shape
                    if batch_size_inner > 1 and num_to_gnn_proj is not None and gnn_to_num_proj is not None:
                        # 創建批次內的全連接圖
                        edge_index = []
                        for i in range(batch_size_inner):
                            for j in range(batch_size_inner):
                                if i != j:
                                    edge_index.append([i, j])
                        
                        if edge_index:
                            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(feature.device)
                            try:
                                # 使用持久的投影層
                                feature_proj = num_to_gnn_proj(feature)  # [batch_size, channels]
                                feature_gnn = gnn(feature_proj, edge_index)
                                feature = gnn_to_num_proj(feature_gnn)  # [batch_size, num_features]
                            except Exception as e:
                                if debug:
                                    print(f"[TabTransformer] GNN failed for numerical features: {e}")
                        
                        if debug:
                            print(f"[TabTransformer] After {stage_name} GNN ({feature_type}): {feature.shape}")
                
                processed.append((feature, feature_type))
            return processed
    
        # Stage 2: GNN 處理 (在 Transformer 之前)
        if gnn_stage == 'encoding':
            encoded_features = apply_gnn_to_features(encoded_features, "encoding")
        
        # Stage 3: 列間交互階段 (Transformer)
        processed_features = process_batch_interaction(encoded_features)
        
        # Stage 4: GNN 處理 (在 Transformer 之後)
        if gnn_stage == 'columnwise':
            processed_features = apply_gnn_to_features(processed_features, "columnwise")
        # Stage 5: 準備解碼輸入
        xs = []
        for feature, feature_type in processed_features:
            if feature_type == "categorical":
                # 展平類別特徵: [batch_size, num_cols, channels] -> [batch_size, num_cols * channels]
                x_cat_flat = feature.reshape(batch_size, -1)
                xs.append(x_cat_flat)
            else:
                # 數值特徵已經是展平的: [batch_size, num_features]
                xs.append(feature)
        
        if debug:
            print(f"[TabTransformer] Before concatenation: {len(xs)} tensors")
            for i, x in enumerate(xs):
                print(f"[TabTransformer]   Tensor {i}: {x.shape}")
        
        # Stage 6: 連接所有特徵並解碼
        if xs:
            x_concat = torch.cat(xs, dim=1)
            if debug:
                print(f"[TabTransformer] After concatenation: {x_concat.shape}")
            out = decoder(x_concat)
            if debug:
                print(f"[TabTransformer] Final output: {out.shape}")
            return out
        else:
            # 處理無特徵的極端情況
            return torch.zeros(batch_size, out_channels, device=device)
        
    # 設置優化器和學習率調度器 (參考 ExcelFormer 風格)
    lr = config.get('lr', 0.0001)
    gamma = config.get('gamma', 0.95)
    
    # 修復：只收集實際使用的參數
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
    
    # 只在需要時添加 GNN 參數
    if gnn is not None:
        all_params.extend(list(gnn.parameters()))
    if num_to_gnn_proj is not None:
        all_params.extend(list(num_to_gnn_proj.parameters()))
    if gnn_to_num_proj is not None:
        all_params.extend(list(gnn_to_num_proj.parameters()))
    
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    # 定義訓練函數 (參考 ExcelFormer 風格)
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
        
        # 修復：設置 GNN 相關模組為訓練模式
        if gnn is not None:
            gnn.train()
        if num_to_gnn_proj is not None:
            num_to_gnn_proj.train()
        if gnn_to_num_proj is not None:
            gnn_to_num_proj.train()
        # if gnn is not None:
        #     gnn.train()
        
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
    
    # 定義測試函數 (參考 ExcelFormer 風格)
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
        
        # 修復：設置 GNN 相關模組為評估模式
        if gnn is not None:
            gnn.eval()
        if num_to_gnn_proj is not None:
            num_to_gnn_proj.eval()
        if gnn_to_num_proj is not None:
            gnn_to_num_proj.eval()
        # if gnn is not None:
        #     gnn.eval()
        
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
        'gnn_early_stop_epochs': 0 if gnn_stage != 'decoding' else gnn_early_stop_epochs,
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