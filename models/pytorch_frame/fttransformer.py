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

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='adult')
# parser.add_argument('--numerical_encoder_type', type=str, default='linear',
#                     choices=['linear', 'linearbucket', 'linearperiodic'])
# parser.add_argument('--model_type', type=str, default='fttransformer',
#                     choices=['fttransformer'])
# parser.add_argument('--channels', type=int, default=256)
# parser.add_argument('--num_layers', type=int, default=4)
# parser.add_argument('--batch_size', type=int, default=512)
# parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--compile', action='store_true')
# args = parser.parse_args()

# torch.manual_seed(args.seed)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Prepare datasets
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
#                 args.dataset)
# dataset = Yandex(root=path, name=args.dataset)
# dataset.materialize()
# is_classification = dataset.task_type.is_classification

# train_dataset, val_dataset, test_dataset = dataset.split()

# # Set up data loaders
# train_tensor_frame = train_dataset.tensor_frame
# val_tensor_frame = val_dataset.tensor_frame
# test_tensor_frame = test_dataset.tensor_frame
# train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
#                           shuffle=True)
# val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
# test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

# if args.numerical_encoder_type == 'linear':
#     numerical_encoder = LinearEncoder()
# elif args.numerical_encoder_type == 'linearbucket':
#     numerical_encoder = LinearBucketEncoder()
# elif args.numerical_encoder_type == 'linearperiodic':
#     numerical_encoder = LinearPeriodicEncoder()
# else:
#     raise ValueError(
#         f'Unsupported encoder type: {args.numerical_encoder_type}')

# stype_encoder_dict = {
#     stype.categorical: EmbeddingEncoder(),
#     stype.numerical: numerical_encoder,
# }

# if is_classification:
#     output_channels = dataset.num_classes
# else:
#     output_channels = 1

# if args.model_type == 'fttransformer':
#     model = FTTransformer(
#         channels=args.channels,
#         out_channels=output_channels,
#         num_layers=args.num_layers,
#         col_stats=dataset.col_stats,
#         col_names_dict=train_tensor_frame.col_names_dict,
#         stype_encoder_dict=stype_encoder_dict,
#     ).to(device)
# elif args.model_type == 'resnet':
#     model = ResNet(
#         channels=args.channels,
#         out_channels=output_channels,
#         num_layers=args.num_layers,
#         col_stats=dataset.col_stats,
#         col_names_dict=train_tensor_frame.col_names_dict,
#     ).to(device)
# else:
#     raise ValueError(f'Unsupported model type: {args.model_type}')

# model = torch.compile(model, dynamic=True) if args.compile else model
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


# def train(epoch: int) -> float:
#     model.train()
#     loss_accum = total_count = 0

#     for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
#         tf = tf.to(device)
#         pred = model(tf)
#         if is_classification:
#             loss = F.cross_entropy(pred, tf.y)
#         else:
#             loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         loss_accum += float(loss) * len(tf.y)
#         total_count += len(tf.y)
#         optimizer.step()
#     return loss_accum / total_count


# @torch.no_grad()
# def test(loader: DataLoader) -> float:
#     model.eval()
#     accum = total_count = 0

#     for tf in loader:
#         tf = tf.to(device)
#         pred = model(tf)
#         if is_classification:
#             pred_class = pred.argmax(dim=-1)
#             accum += float((tf.y == pred_class).sum())
#         else:
#             accum += float(
#                 F.mse_loss(pred.view(-1), tf.y.view(-1), reduction='sum'))
#         total_count += len(tf.y)

#     if is_classification:
#         accuracy = accum / total_count
#         return accuracy
#     else:
#         rmse = (accum / total_count)**0.5
#         return rmse


# if is_classification:
#     metric = 'Acc'
#     best_val_metric = 0
#     best_test_metric = 0
# else:
#     metric = 'RMSE'
#     best_val_metric = float('inf')
#     best_test_metric = float('inf')

# for epoch in range(1, args.epochs + 1):
#     train_loss = train(epoch)
#     train_metric = test(train_loader)
#     val_metric = test(val_loader)
#     test_metric = test(test_loader)

#     if is_classification and val_metric > best_val_metric:
#         best_val_metric = val_metric
#         best_test_metric = test_metric
#     elif not is_classification and val_metric < best_val_metric:
#         best_val_metric = val_metric
#         best_test_metric = test_metric

#     print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
#           f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')

# print(f'Best Val {metric}: {best_val_metric:.4f}, '
#       f'Best Test {metric}: {best_test_metric:.4f}')


def start_fn(df, dataset_results, config):
    return df


def materialize_fn(df, dataset_results, config):
    """
    階段1: Materialization - 將原始表格數據轉換為張量格式
    
    輸入:
    - df: 輸入數據框
    - dataset_results: 數據集信息
    - config: 配置參數
    
    輸出:
    - 包含資料集和張量框架的字典，可直接傳給encoding_fn或自定義GNN
    """
    print("Executing materialize_fn")
    print(f"Input DataFrame shape: {df.shape}")
    # 獲取配置參數
    dataset_name = dataset_results['dataset']
    dataset_size = dataset_results['info']['size']
    task_type = dataset_results['info']['task_type']
    train_val_test_split_ratio = config['train_val_test_split_ratio']
    numerical_encoder_type='linear'
    model_type='fttransformer'
    channels=256
    num_layers=4

    # 設備設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 數據集加載和物化
    dataset = Yandex(df=df, name=dataset_name, 
                     train_val_test_split_ratio=train_val_test_split_ratio, 
                     task_type=task_type, DS=False)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    train_dataset, val_dataset, test_dataset = dataset.split()

    # Set up data loaders
    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    print(f"Train TensorFrame shape: {train_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Val TensorFrame shape: {val_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Test TensorFrame shape: {test_tensor_frame.feat_dict[stype.numerical].shape}")

    # 創建數據加載器
    batch_size = config.get('batch_size', 512)
    print(f"Batch size: {batch_size}")
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size,
                            shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)
    if numerical_encoder_type == 'linear':
        numerical_encoder = LinearEncoder()
    elif numerical_encoder_type == 'linearbucket':
        numerical_encoder = LinearBucketEncoder()
    elif numerical_encoder_type == 'linearperiodic':
        numerical_encoder = LinearPeriodicEncoder()
    else:
        raise ValueError(
            f'Unsupported encoder type: {numerical_encoder_type}')

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: numerical_encoder,
    }

    if is_classification:
        output_channels = dataset.num_classes
    else:
        output_channels = 1

    is_binary_class = is_classification and output_channels == 2
    # 設置評估指標
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
    # 返回所有需要的信息 - 這些都是encoding_fn的輸入
    return {
        'dataset': dataset,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
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
    device = material_outputs['device']
    stype_encoder_dict = material_outputs['stype_encoder_dict']
    print(f"Input Train TensorFrame shape: {train_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Input Val TensorFrame shape: {val_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Input Test TensorFrame shape: {test_tensor_frame.feat_dict[stype.numerical].shape}")
    # 獲取模型參數
    channels = config.get('channels', 256)
    
    # 創建FTTransformer的編碼器部分
    encoder = StypeWiseFeatureEncoder(
        out_channels=channels,
        col_stats=col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
    
    # 對訓練、驗證和測試數據進行編碼處理
    # 這裡我們將預先處理每個批次，生成編碼後的嵌入
    train_embeddings = []
    val_embeddings = []
    test_embeddings = []
    train_labels = []
    val_labels = []
    test_labels = []
    
    with torch.no_grad():
        # 處理訓練數據
        for tf in train_loader:
            tf = tf.to(device)
            x, _ = encoder(tf)  # 獲取編碼後的嵌入
            train_embeddings.append(x)
            train_labels.append(tf.y)
        
        # 處理驗證數據
        for tf in val_loader:
            tf = tf.to(device)
            x, _ = encoder(tf)
            val_embeddings.append(x)
            val_labels.append(tf.y)
        
        # 處理測試數據
        for tf in test_loader:
            tf = tf.to(device)
            x, _ = encoder(tf)
            test_embeddings.append(x)
            test_labels.append(tf.y)
    
    # 合併所有批次的嵌入和標籤
    all_train_embeddings = torch.cat(train_embeddings, dim=0) if train_embeddings else None
    all_val_embeddings = torch.cat(val_embeddings, dim=0) if val_embeddings else None
    all_test_embeddings = torch.cat(test_embeddings, dim=0) if test_embeddings else None
    all_train_labels = torch.cat(train_labels, dim=0) if train_labels else None
    all_val_labels = torch.cat(val_labels, dim=0) if val_labels else None
    all_test_labels = torch.cat(test_labels, dim=0) if test_labels else None
    print(f"Train Embeddings shape: {all_train_embeddings.shape if all_train_embeddings is not None else None}")
    print(f"Val Embeddings shape: {all_val_embeddings.shape if all_val_embeddings is not None else None}")
    print(f"Test Embeddings shape: {all_test_embeddings.shape if all_test_embeddings is not None else None}")
    print(f"Train Labels shape: {all_train_labels.shape if all_train_labels is not None else None}")
    print(f"Val Labels shape: {all_val_labels.shape if all_val_labels is not None else None}")
    print(f"Test Labels shape: {all_test_labels.shape if all_test_labels is not None else None}")
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
        'val_tensor_frame': val_tensor_frame,
        'test_tensor_frame': test_tensor_frame,
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
    device = encoding_outputs['device']
    print(f"Train Embeddings shape: {train_embeddings.shape}")
    print(f"Val Embeddings shape: {val_embeddings.shape}")
    print(f"Test Embeddings shape: {test_embeddings.shape}")
    print(f"Train Labels shape: {train_labels.shape}")
    print(f"Val Labels shape: {val_labels.shape}")
    print(f"Test Labels shape: {test_labels.shape}")
    # 獲取列間交互參數
    num_layers = config.get('num_layers', 4)
    
    # 創建FTTransformer的列間交互層 - FTTransformerConvs
    backbone = FTTransformerConvs(
        channels=channels,
        num_layers=num_layers
    ).to(device)
    
    # 對編碼後的嵌入應用列間交互層
    train_conv_outputs = []
    val_conv_outputs = []
    test_conv_outputs = []
    train_cls_outputs = []
    val_cls_outputs = []
    test_cls_outputs = []
    
    with torch.no_grad():
        # 處理訓練數據
        x, x_cls = backbone(train_embeddings)
        train_conv_outputs.append(x)
        train_cls_outputs.append(x_cls)
        
        # 處理驗證數據
        x, x_cls = backbone(val_embeddings)
        val_conv_outputs.append(x)
        val_cls_outputs.append(x_cls)
        
        # 處理測試數據
        x, x_cls = backbone(test_embeddings)
        test_conv_outputs.append(x)
        test_cls_outputs.append(x_cls)
    
    # 合併結果
    all_train_conv_outputs = torch.cat(train_conv_outputs, dim=0) if train_conv_outputs else None
    all_val_conv_outputs = torch.cat(val_conv_outputs, dim=0) if val_conv_outputs else None
    all_test_conv_outputs = torch.cat(test_conv_outputs, dim=0) if test_conv_outputs else None
    all_train_cls_outputs = torch.cat(train_cls_outputs, dim=0) if train_cls_outputs else None
    all_val_cls_outputs = torch.cat(val_cls_outputs, dim=0) if val_cls_outputs else None
    all_test_cls_outputs = torch.cat(test_cls_outputs, dim=0) if test_cls_outputs else None
    print(f"Train Conv Outputs shape: {all_train_conv_outputs.shape if all_train_conv_outputs is not None else None}")
    print(f"Val Conv Outputs shape: {all_val_conv_outputs.shape if all_val_conv_outputs is not None else None}")
    print(f"Test Conv Outputs shape: {all_test_conv_outputs.shape if all_test_conv_outputs is not None else None}")
    print(f"Train CLS Outputs shape: {all_train_cls_outputs.shape if all_train_cls_outputs is not None else None}")
    print(f"Val CLS Outputs shape: {all_val_cls_outputs.shape if all_val_cls_outputs is not None else None}")
    print(f"Test CLS Outputs shape: {all_test_cls_outputs.shape if all_test_cls_outputs is not None else None}")
    # 返回列間交互結果和相關信息 - 這些都是decoding_fn的輸入
    return {
        'backbone': backbone,
        'train_conv_outputs': all_train_conv_outputs,
        'val_conv_outputs': all_val_conv_outputs,
        'test_conv_outputs': all_test_conv_outputs,
        'train_cls_outputs': all_train_cls_outputs,
        'val_cls_outputs': all_val_cls_outputs,
        'test_cls_outputs': all_test_cls_outputs,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'train_loader': encoding_outputs['train_loader'],
        'val_loader': encoding_outputs['val_loader'],
        'test_loader': encoding_outputs['test_loader'],
        'encoder': encoding_outputs['encoder'],
        'out_channels': encoding_outputs['out_channels'],
        'is_classification': encoding_outputs['is_classification'],
        'is_binary_class': encoding_outputs['is_binary_class'],
        'metric_computer': encoding_outputs['metric_computer'],
        'metric': encoding_outputs['metric'],
        'channels': channels,
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
    train_cls_outputs = columnwise_outputs['train_cls_outputs']
    val_cls_outputs = columnwise_outputs['val_cls_outputs']
    test_cls_outputs = columnwise_outputs['test_cls_outputs']
    train_labels = columnwise_outputs['train_labels']
    val_labels = columnwise_outputs['val_labels']
    test_labels = columnwise_outputs['test_labels']
    train_loader = columnwise_outputs['train_loader']
    val_loader = columnwise_outputs['val_loader']
    test_loader = columnwise_outputs['test_loader']
    encoder = columnwise_outputs['encoder']
    backbone = columnwise_outputs['backbone']
    channels = columnwise_outputs['channels']
    out_channels = columnwise_outputs['out_channels']
    device = columnwise_outputs['device']
    is_classification = columnwise_outputs['is_classification']
    is_binary_class = columnwise_outputs['is_binary_class']
    metric_computer = columnwise_outputs['metric_computer']
    metric = columnwise_outputs['metric']
    print(f"Train CLS Outputs shape: {train_cls_outputs.shape}")
    print(f"Val CLS Outputs shape: {val_cls_outputs.shape}")
    print(f"Test CLS Outputs shape: {test_cls_outputs.shape}")
    print(f"Train Labels shape: {train_labels.shape}")
    print(f"Val Labels shape: {val_labels.shape}")
    print(f"Test Labels shape: {test_labels.shape}")
    # 創建解碼器 - 對應FTTransformer中的decoder部分
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
        x, x_cls = backbone(x)
        out = decoder(x_cls)
        return out
    
    # 設置優化器和學習率
    lr = config.get('lr', 0.0001)
    all_params = list(encoder.parameters()) + list(backbone.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr)
    
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
    test_metrics = []
    
    # 訓練循環
    epochs = config.get('epochs', 100)
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
        'model_forward': model_forward  # 返回完整模型的前向傳播函數
    }
def main(df, dataset_results, config):
    """
    主函數：按順序調用四個階段函數
    
    可用於在階段間插入GNN模型
    """
    print("FTTransformer - 四階段執行")
    try:
        # 階段0: 開始
        df = start_fn(df, dataset_results, config)
        # 階段1: Materialization
        material_outputs = materialize_fn(df, dataset_results, config)
        
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
        # 返回一個基本值的結果
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