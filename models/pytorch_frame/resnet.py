from __future__ import annotations

import math
from typing import Any

from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
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
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import Yandex
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    ResNet,
)
class FCResidualBlock(Module):
    r"""Fully connected residual block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str, optional): The type of normalization to use.
            :obj:`layer_norm`, :obj:`batch_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.0`, i.e.,
            no dropout).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_prob)

        self.norm1: BatchNorm1d | LayerNorm | None
        self.norm2: BatchNorm1d | LayerNorm | None
        if normalization == "batch_norm":
            self.norm1 = BatchNorm1d(out_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == "layer_norm":
            self.norm1 = LayerNorm(out_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

        self.shortcut: Linear | None
        if in_channels != out_channels:
            self.shortcut = Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.shortcut is not None:
            self.shortcut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out = self.lin1(x)
        out = self.norm1(out) if self.norm1 else out
        out = self.relu(out)
        out = self.dropout(out)

        out = self.lin2(out)
        out = self.norm2(out) if self.norm2 else out
        out = self.relu(out)
        out = self.dropout(out)

        if self.shortcut is not None:
            x = self.shortcut(x)

        out = out + x

        return out


class ResNet(Module):
    r"""The ResNet model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using ResNet, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): The number of channels in the backbone layers.
        out_channels (int): The number of output channels in the decoder.
        num_layers (int): The number of layers in the backbone.
        col_stats(dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`LinearEncoder()` for
            numerical feature)
        normalization (str, optional): The type of normalization to use.
            :obj:`batch_norm`, :obj:`layer_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.2`).
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])
        in_channels = channels * num_cols
        self.backbone = Sequential(*[
            FCResidualBlock(
                in_channels if i == 0 else channels,
                channels,
                normalization=normalization,
                dropout_prob=dropout_prob,
            ) for i in range(num_layers)
        ])

        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for block in self.backbone:
            block.reset_parameters()
        self.decoder[0].reset_parameters()
        self.decoder[-1].reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        # Flattening the encoder output
        x = x.view(x.size(0), math.prod(x.shape[1:]))

        x = self.backbone(x)
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

    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    numerical_encoder_type = config.get('numerical_encoder_type', 'linear')
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
        out_channels = dataset.num_classes
    else:
        out_channels = 1

    is_binary_class = is_classification and out_channels == 2

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
        'col_stats': dataset.col_stats,
        'stype_encoder_dict': stype_encoder_dict,
        'metric_computer': metric_computer,
        'metric': metric,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'out_channels': out_channels,
        'device': device
    }


def encoding_fn(material_outputs, config):
    """
    階段2: Encoding - 將張量框架編碼為嵌入向量
    
    輸入:
    - material_outputs: materialize_fn的輸出
    - config: 配置參數
    
    輸出:
    - 編碼後的嵌入表示，可傳給columnwise_fn或自定義GNN
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
    stype_encoder_dict = material_outputs['stype_encoder_dict']
    device = material_outputs['device']
    
    # 獲取模型參數
    channels = config.get('channels', 256)
    print(f"Encoding with channels: {channels}")
    
    # 創建ResNet的編碼器部分
    encoder = StypeWiseFeatureEncoder(
        out_channels=channels,
        col_stats=col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
    
    # 對訓練、驗證和測試數據進行編碼處理
    train_embeddings = []
    val_embeddings = []
    test_embeddings = []
    train_labels = []
    val_labels = []
    test_labels = []
    
    # 控制批次大小，避免GPU內存不足
    batch_size = config.get('batch_size', 512)
    
    with torch.no_grad():
        # 處理訓練數據
        for tf in train_loader:
            tf = tf.to(device)
            x, _ = encoder(tf)  # 獲取編碼後的嵌入
            
            # 在ResNet中，嵌入會被展平處理
            batch_size, num_cols, embed_dim = x.shape
            x_flattened = x.reshape(batch_size, -1)  # 展平為[batch_size, num_cols*embed_dim]
            
            train_embeddings.append(x_flattened.cpu())  # 移到CPU以節省GPU內存
            train_labels.append(tf.y.cpu())
        
        # 處理驗證數據
        for tf in val_loader:
            tf = tf.to(device)
            x, _ = encoder(tf)
            
            batch_size, num_cols, embed_dim = x.shape
            x_flattened = x.reshape(batch_size, -1)
            
            val_embeddings.append(x_flattened.cpu())
            val_labels.append(tf.y.cpu())
        
        # 處理測試數據
        for tf in test_loader:
            tf = tf.to(device)
            x, _ = encoder(tf)
            
            batch_size, num_cols, embed_dim = x.shape
            x_flattened = x.reshape(batch_size, -1)
            
            test_embeddings.append(x_flattened.cpu())
            test_labels.append(tf.y.cpu())
    
    # 合併所有批次的嵌入和標籤
    if train_embeddings:
        all_train_embeddings = torch.cat(train_embeddings, dim=0)
        all_train_labels = torch.cat(train_labels, dim=0)
    else:
        all_train_embeddings = None
        all_train_labels = None
        
    if val_embeddings:
        all_val_embeddings = torch.cat(val_embeddings, dim=0)
        all_val_labels = torch.cat(val_labels, dim=0)
    else:
        all_val_embeddings = None
        all_val_labels = None
        
    if test_embeddings:
        all_test_embeddings = torch.cat(test_embeddings, dim=0)
        all_test_labels = torch.cat(test_labels, dim=0)
    else:
        all_test_embeddings = None
        all_test_labels = None
    
    # 計算嵌入的特徵數量，用於後續的backbone層
    embed_dim = all_train_embeddings.shape[1] if all_train_embeddings is not None else 0
    
    # 返回編碼結果和相關信息 - 這些都是columnwise_fn的輸入
    return {
        'encoder': encoder,
        'train_embeddings': all_train_embeddings,
        'val_embeddings': all_val_embeddings, 
        'test_embeddings': all_test_embeddings,
        'train_labels': all_train_labels,
        'val_labels': all_val_labels,
        'test_labels': all_test_labels,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_tensor_frame': train_tensor_frame,
        'embed_dim': embed_dim,  # 嵌入特徵數
        'channels': channels,
        'out_channels': material_outputs['out_channels'],
        'is_classification': material_outputs['is_classification'],
        'is_binary_class': material_outputs['is_binary_class'],
        'metric_computer': material_outputs['metric_computer'],
        'metric': material_outputs['metric'],
        'device': device
    }


def columnwise_fn(encoding_outputs, config):
    """
    階段3: Column-wise Interaction - 處理列間交互
    
    輸入:
    - encoding_outputs: encoding_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 處理後的嵌入，可傳給decoding_fn或自定義GNN
    """
    print("Executing columnwise_fn")
    
    # 從上一階段獲取數據
    train_embeddings = encoding_outputs['train_embeddings']
    val_embeddings = encoding_outputs['val_embeddings']
    test_embeddings = encoding_outputs['test_embeddings']
    train_labels = encoding_outputs['train_labels']
    val_labels = encoding_outputs['val_labels']
    test_labels = encoding_outputs['test_labels']
    channels = encoding_outputs['channels']
    embed_dim = encoding_outputs['embed_dim']
    device = encoding_outputs['device']
    
    # 獲取ResNet的參數
    num_layers = config.get('num_layers', 4)
    normalization = config.get('normalization', 'layer_norm')
    dropout_prob = config.get('dropout_prob', 0.2)
    
    print(f"Building ResNet backbone with {num_layers} layers")
    
    # 創建ResNet的骨幹網絡 - FCResidualBlock的堆疊
    backbone = Sequential(*[
        FCResidualBlock(
            embed_dim if i == 0 else channels,  # 第一層使用原始嵌入維度
            channels,
            normalization=normalization,
            dropout_prob=dropout_prob,
        ).to(device)
        for i in range(num_layers)
    ])
    
    # 對嵌入數據應用骨幹網絡進行處理
    # 使用小批次處理以避免GPU內存不足
    batch_size = 128
    
    train_backbone_outputs = []
    val_backbone_outputs = []
    test_backbone_outputs = []
    
    # 將數據移回GPU進行處理
    with torch.no_grad():
        # 處理訓練數據
        for i in range(0, train_embeddings.size(0), batch_size):
            end_idx = min(i + batch_size, train_embeddings.size(0))
            batch = train_embeddings[i:end_idx].to(device)
            output = backbone(batch)
            train_backbone_outputs.append(output.cpu())
        
        # 處理驗證數據
        for i in range(0, val_embeddings.size(0), batch_size):
            end_idx = min(i + batch_size, val_embeddings.size(0))
            batch = val_embeddings[i:end_idx].to(device)
            output = backbone(batch)
            val_backbone_outputs.append(output.cpu())
        
        # 處理測試數據
        for i in range(0, test_embeddings.size(0), batch_size):
            end_idx = min(i + batch_size, test_embeddings.size(0))
            batch = test_embeddings[i:end_idx].to(device)
            output = backbone(batch)
            test_backbone_outputs.append(output.cpu())
    
    # 合併處理後的結果
    all_train_backbone_outputs = torch.cat(train_backbone_outputs, dim=0) if train_backbone_outputs else None
    all_val_backbone_outputs = torch.cat(val_backbone_outputs, dim=0) if val_backbone_outputs else None
    all_test_backbone_outputs = torch.cat(test_backbone_outputs, dim=0) if test_backbone_outputs else None
    
    # 返回骨幹網絡處理結果和相關信息 - 這些都是decoding_fn的輸入
    return {
        'backbone': backbone,
        'train_backbone_outputs': all_train_backbone_outputs,
        'val_backbone_outputs': all_val_backbone_outputs,
        'test_backbone_outputs': all_test_backbone_outputs,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'train_embeddings': train_embeddings,  # 保留原始嵌入，以便可能的需要
        'val_embeddings': val_embeddings,
        'test_embeddings': test_embeddings,
        'train_loader': encoding_outputs['train_loader'],
        'val_loader': encoding_outputs['val_loader'],
        'test_loader': encoding_outputs['test_loader'],
        'encoder': encoding_outputs['encoder'],
        'channels': channels,
        'out_channels': encoding_outputs['out_channels'],
        'is_classification': encoding_outputs['is_classification'],
        'is_binary_class': encoding_outputs['is_binary_class'],
        'metric_computer': encoding_outputs['metric_computer'],
        'metric': encoding_outputs['metric'],
        'device': device
    }

def decoding_fn(columnwise_outputs, config):
    """
    階段4: Decoding - 解碼預測並訓練模型
    
    輸入:
    - columnwise_outputs: columnwise_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 訓練結果和最終模型
    """
    print("Executing decoding_fn")
    
    # 從上一階段獲取數據
    backbone = columnwise_outputs['backbone']
    train_backbone_outputs = columnwise_outputs['train_backbone_outputs']
    val_backbone_outputs = columnwise_outputs['val_backbone_outputs']
    test_backbone_outputs = columnwise_outputs['test_backbone_outputs']
    train_labels = columnwise_outputs['train_labels']
    val_labels = columnwise_outputs['val_labels']
    test_labels = columnwise_outputs['test_labels']
    train_loader = columnwise_outputs['train_loader']
    val_loader = columnwise_outputs['val_loader']
    test_loader = columnwise_outputs['test_loader']
    encoder = columnwise_outputs['encoder']
    channels = columnwise_outputs['channels']
    out_channels = columnwise_outputs['out_channels']
    device = columnwise_outputs['device']
    is_classification = columnwise_outputs['is_classification']
    is_binary_class = columnwise_outputs['is_binary_class']
    metric_computer = columnwise_outputs['metric_computer']
    metric = columnwise_outputs['metric']
    
    # 創建ResNet的解碼器部分
    decoder = Sequential(
        LayerNorm(channels),
        ReLU(),
        Linear(channels, out_channels),
    ).to(device)
    
    # 初始化解碼器參數
    decoder[0].reset_parameters()  # LayerNorm
    decoder[2].reset_parameters()  # Linear
    
    # 實現完整的ResNet前向傳播函數
    def model_forward(tf):
        x, _ = encoder(tf)
        batch_size, num_cols, embed_dim = x.shape
        x = x.reshape(batch_size, -1)  # 展平
        x = backbone(x)
        out = decoder(x)
        return out
    
    # 設置優化器
    lr = config.get('lr', 0.0001)
    all_params = list(encoder.parameters()) + list(backbone.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr)
    
    # 創建數據集
    class EmbeddingDataset:
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]
    
    # 創建數據加載器
    batch_size = config.get('batch_size', 512)
    train_embed_dataset = EmbeddingDataset(train_backbone_outputs.to(device), train_labels.to(device))
    val_embed_dataset = EmbeddingDataset(val_backbone_outputs.to(device), val_labels.to(device))
    test_embed_dataset = EmbeddingDataset(test_backbone_outputs.to(device), test_labels.to(device))
    
    train_embed_loader = torch.utils.data.DataLoader(train_embed_dataset, batch_size=batch_size, shuffle=True)
    val_embed_loader = torch.utils.data.DataLoader(val_embed_dataset, batch_size=batch_size)
    test_embed_loader = torch.utils.data.DataLoader(test_embed_dataset, batch_size=batch_size)
    
    # 定義訓練函數 - 使用預處理後的嵌入進行快速訓練
    def train_on_embeddings(epoch):
        decoder.train()
        loss_accum = total_count = 0
        
        for embeddings, labels in tqdm(train_embed_loader, desc=f'Epoch: {epoch}'):
            out = decoder(embeddings)
            
            if is_classification:
                loss = F.cross_entropy(out, labels)
            else:
                loss = F.mse_loss(out.view(-1), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(labels)
            total_count += len(labels)
            optimizer.step()
        
        return loss_accum / total_count
    
    # 定義完整模型的訓練函數
    def train_full_model(epoch):
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
    
    # 定義測試函數 - 使用預處理後的嵌入進行快速評估
    @torch.no_grad()
    def test_on_embeddings(loader):
        decoder.eval()
        metric_computer.reset()
        
        for embeddings, labels in loader:
            pred = decoder(embeddings)
            
            if is_binary_class:
                metric_computer.update(pred[:, 1], labels)
            elif is_classification:
                pred_class = pred.argmax(dim=-1)
                metric_computer.update(pred_class, labels)
            else:
                metric_computer.update(pred.view(-1), labels.view(-1))
        
        if is_classification:
            return metric_computer.compute().item()
        else:
            return metric_computer.compute().item()**0.5
    
    # 定義完整模型的測試函數
    @torch.no_grad()
    def test_full_model(loader):
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
    test_metrics = []
    
    # 訓練循環
    epochs = config.get('epochs', 100)
    use_embeddings = config.get('use_embeddings', True)  # 是否使用預處理嵌入訓練
    
    for epoch in range(1, epochs + 1):
        # 使用預處理嵌入或完整模型訓練
        if use_embeddings:
            train_loss = train_on_embeddings(epoch)
            train_metric = test_on_embeddings(train_embed_loader)
            val_metric = test_on_embeddings(val_embed_loader)
            test_metric = test_on_embeddings(test_embed_loader)
        else:
            train_loss = train_full_model(epoch)
            train_metric = test_full_model(train_loader)
            val_metric = test_full_model(val_loader)
            test_metric = test_full_model(test_loader)
        
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
    
    print(f'Best Val {metric}: {best_val_metric:.4f}, '
          f'Best Test {metric}: {best_test_metric:.4f}')
    
    # 返回訓練結果
    return {
        'train_losses': train_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'encoder': encoder,
        'backbone': backbone,
        'decoder': decoder,
        'model_forward': model_forward  # 返回完整模型的前向函數
    }

def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    主函數：按順序調用四個階段函數
    """
    print("ResNet - 四階段執行")
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