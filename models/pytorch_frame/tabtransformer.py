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

def encoding_fn(material_outputs, config):
    """
    階段2: Encoding - 創建編碼器並處理類別特徵和數值特徵
    
    輸入:
    - material_outputs: materialize_fn的輸出
    - config: 配置參數
    
    輸出:
    - 包含編碼器和相關設置的字典，可傳給columnwise_fn或自定義GNN
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
    
    # 獲取模型參數
    channels = config.get('channels', 32)
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
    
    # 返回編碼器和相關信息
    return {
        'cat_encoder': cat_encoder,
        'pad_embedding': pad_embedding,
        'num_encoder': num_encoder,
        'num_norm': num_norm,
        'categorical_col_len': categorical_col_len,
        'numerical_col_len': numerical_col_len,
        'col_names_dict': col_names_dict,
        'col_stats': col_stats,
        'encode_batch': encode_batch,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_tensor_frame': train_tensor_frame,
        'val_tensor_frame': val_tensor_frame,
        'test_tensor_frame': test_tensor_frame,
        'channels': channels,
        'encoder_pad_size': encoder_pad_size,
        'out_channels': material_outputs['out_channels'],
        'is_classification': material_outputs['is_classification'],
        'is_binary_class': material_outputs['is_binary_class'],
        'metric_computer': material_outputs['metric_computer'],
        'metric': material_outputs['metric'],
        'device': device
    }
def columnwise_fn(encoding_outputs, config):
    """
    階段3: Column-wise Interaction - 創建Transformer層處理類別特徵的列間交互
    
    輸入:
    - encoding_outputs: encoding_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 包含列間交互處理層的字典，可傳給decoding_fn或自定義GNN
    """
    print("Executing columnwise_fn")
    
    # 從上一階段獲取數據
    cat_encoder = encoding_outputs['cat_encoder']
    pad_embedding = encoding_outputs['pad_embedding']
    num_encoder = encoding_outputs['num_encoder']
    num_norm = encoding_outputs['num_norm']
    categorical_col_len = encoding_outputs['categorical_col_len']
    numerical_col_len = encoding_outputs['numerical_col_len']
    col_names_dict = encoding_outputs['col_names_dict']
    channels = encoding_outputs['channels']
    device = encoding_outputs['device']
    
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
    
    # 返回列間交互層和相關信息
    return {
        'tab_transformer_convs': tab_transformer_convs,
        'cat_encoder': cat_encoder,
        'pad_embedding': pad_embedding,
        'num_encoder': num_encoder,
        'num_norm': num_norm,
        'categorical_col_len': categorical_col_len,
        'numerical_col_len': numerical_col_len,
        'col_names_dict': col_names_dict,
        'encode_batch': encoding_outputs['encode_batch'],
        'process_batch_interaction': process_batch_interaction,
        'train_loader': encoding_outputs['train_loader'],
        'val_loader': encoding_outputs['val_loader'],
        'test_loader': encoding_outputs['test_loader'],
        'train_tensor_frame': encoding_outputs['train_tensor_frame'],
        'val_tensor_frame': encoding_outputs['val_tensor_frame'],
        'test_tensor_frame': encoding_outputs['test_tensor_frame'],
        'channels': channels,
        'encoder_pad_size': encoding_outputs['encoder_pad_size'],
        'out_channels': encoding_outputs['out_channels'],
        'is_classification': encoding_outputs['is_classification'],
        'is_binary_class': encoding_outputs['is_binary_class'],
        'metric_computer': encoding_outputs['metric_computer'],
        'metric': encoding_outputs['metric'],
        'device': device
    }
def decoding_fn(columnwise_outputs, config):
    """
    階段4: Decoding - 創建解碼器並訓練模型
    
    輸入:
    - columnwise_outputs: columnwise_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 訓練結果和最終模型
    """
    print("Executing decoding_fn")
    
    # 從上一階段獲取數據
    cat_encoder = columnwise_outputs['cat_encoder']
    pad_embedding = columnwise_outputs['pad_embedding']
    num_encoder = columnwise_outputs['num_encoder']
    num_norm = columnwise_outputs['num_norm']
    tab_transformer_convs = columnwise_outputs['tab_transformer_convs']
    categorical_col_len = columnwise_outputs['categorical_col_len']
    numerical_col_len = columnwise_outputs['numerical_col_len']
    col_names_dict = columnwise_outputs['col_names_dict']
    encode_batch = columnwise_outputs['encode_batch']
    process_batch_interaction = columnwise_outputs['process_batch_interaction']
    train_loader = columnwise_outputs['train_loader']
    val_loader = columnwise_outputs['val_loader']
    test_loader = columnwise_outputs['test_loader']
    channels = columnwise_outputs['channels']
    out_channels = columnwise_outputs['out_channels']
    device = columnwise_outputs['device']
    is_classification = columnwise_outputs['is_classification']
    is_binary_class = columnwise_outputs['is_binary_class']
    metric_computer = columnwise_outputs['metric_computer']
    metric = columnwise_outputs['metric']
    
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
    
    # 定義完整的前向傳播函數
    def forward(tf):
        # 編碼階段
        encoded_features = encode_batch(tf)
        
        # 列間交互階段
        processed_features = process_batch_interaction(encoded_features)
        
        # 準備解碼輸入
        xs = []
        batch_size = len(tf)
        
        for feature, feature_type in processed_features:
            if feature_type == "categorical":
                # 展平類別特徵
                x_cat_flat = feature.reshape(batch_size, -1)
                xs.append(x_cat_flat)
            else:
                # 數值特徵已經是展平的
                xs.append(feature)
        
        # 連接所有特徵
        if xs:
            x = torch.cat(xs, dim=1)
            # 解碼階段
            out = decoder(x)
            return out
        else:
            # 處理無特徵的極端情況
            return torch.zeros(batch_size, out_channels, device=device)
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.0001)
    
    # 收集所有參數
    model_params = []
    if cat_encoder is not None:
        model_params.extend(list(cat_encoder.parameters()))
    if pad_embedding is not None:
        model_params.extend(list(pad_embedding.parameters()))
    if num_encoder is not None:
        model_params.extend(list(num_encoder.parameters()))
    if num_norm is not None:
        model_params.extend(list(num_norm.parameters()))
    if tab_transformer_convs is not None:
        for conv in tab_transformer_convs:
            model_params.extend(list(conv.parameters()))
    model_params.extend(list(decoder.parameters()))
    
    optimizer = torch.optim.Adam(model_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
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
        
        loss_accum = total_count = 0
        
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            pred = forward(tf)
            
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
        'cat_encoder': cat_encoder,
        'pad_embedding': pad_embedding,
        'num_encoder': num_encoder,
        'num_norm': num_norm,
        'tab_transformer_convs': tab_transformer_convs,
        'decoder': decoder,
        'forward': forward  # 返回完整模型的前向函數
    }

def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    主函數：按順序調用四個階段函數
    """
    print("TabTransformer - 四階段執行")
    print(f"gnn_stage: {gnn_stage}")
    try:
        # 階段0: 開始
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        # 階段1: Materialization
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        # 階段2: Encoding
        encoding_outputs = encoding_fn(material_outputs, config)
        # 這裡可以插入GNN處理編碼後的數據
        # encoding_outputs = gnn_process(encoding_outputs, config)
        # 階段3: Column-wise Interaction
        columnwise_outputs = columnwise_fn(encoding_outputs, config)
        # 這裡可以插入GNN處理列間交互後的數據
        # columnwise_outputs = gnn_process(columnwise_outputs, config)
        # 階段4: Decoding
        results = decoding_fn(columnwise_outputs, config)
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