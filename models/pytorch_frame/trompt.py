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
class Trompt(Module):
    r"""The Trompt model introduced in the
    `"Trompt: Towards a Better Deep Neural Network for Tabular Data"
    <https://arxiv.org/abs/2305.18446>`_ paper.

    .. note::

        For an example of using Trompt, see `examples/trompt.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        trompt.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_prompts (int): Number of prompt columns.
        num_layers (int, optional): Number of :class:`TromptConv` layers.
            (default: :obj:`6`)
        col_stats(Dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (Dict[:obj:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dicts
            (list[dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]], optional):
            A list of :obj:`num_layers` dictionaries that each dictionary maps
            stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`LinearEncoder()` for
            numerical feature)
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_prompts: int,
        num_layers: int,
        # kwargs for encoder
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dicts: list[dict[torch_frame.stype, StypeEncoder]]
        | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])

        self.x_prompt = Parameter(torch.empty(num_prompts, channels))
        self.encoders = ModuleList()
        self.trompt_convs = ModuleList()
        for i in range(num_layers):
            if stype_encoder_dicts is None:
                stype_encoder_dict_layer = {
                    stype.categorical:
                    EmbeddingEncoder(
                        post_module=LayerNorm(channels),
                        na_strategy=NAStrategy.MOST_FREQUENT,
                    ),
                    stype.numerical:
                    LinearEncoder(
                        post_module=Sequential(
                            ReLU(),
                            LayerNorm(channels),
                        ),
                        na_strategy=NAStrategy.MEAN,
                    ),
                }
            else:
                stype_encoder_dict_layer = stype_encoder_dicts[i]

            self.encoders.append(
                StypeWiseFeatureEncoder(
                    out_channels=channels,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict_layer,
                ))
            self.trompt_convs.append(
                TromptConv(channels, num_cols, num_prompts))
        # Decoder is shared across layers.
        self.trompt_decoder = TromptDecoder(channels, out_channels,
                                            num_prompts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.x_prompt, std=0.01)
        for encoder in self.encoders:
            encoder.reset_parameters()
        for trompt_conv in self.trompt_convs:
            trompt_conv.reset_parameters()
        self.trompt_decoder.reset_parameters()

    def forward_stacked(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into a series of output
        predictions at each layer. Used during training to compute layer-wise
        loss.

        Args:
            tf (:class:`torch_frame.TensorFrame`):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output predictions stacked across layers. The
                shape is :obj:`[batch_size, num_layers, out_channels]`.
        """
        batch_size = len(tf)
        outs = []
        # [batch_size, num_prompts, channels]
        x_prompt = self.x_prompt.repeat(batch_size, 1, 1)
        for i in range(self.num_layers):
            # [batch_size, num_cols, channels]
            x, _ = self.encoders[i](tf)
            # [batch_size, num_prompts, channels]
            x_prompt = self.trompt_convs[i](x, x_prompt)
            # [batch_size, out_channels]
            out = self.trompt_decoder(x_prompt)
            # [batch_size, 1, out_channels]
            out = out.view(batch_size, 1, self.out_channels)
            outs.append(out)
        # [batch_size, num_layers, out_channels]
        stacked_out = torch.cat(outs, dim=1)
        return stacked_out

    def forward(self, tf: TensorFrame) -> Tensor:
        return self.forward_stacked(tf).mean(dim=1)

# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, default="california")
# parser.add_argument("--channels", type=int, default=128)
# parser.add_argument("--num_prompts", type=int, default=128)
# parser.add_argument("--num_layers", type=int, default=6)
# parser.add_argument("--batch_size", type=int, default=256)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--epochs", type=int, default=50)
# parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--compile", action="store_true")
# args = parser.parse_args()

# torch.manual_seed(args.seed)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Prepare datasets
# path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",
#                 args.dataset)
# dataset = TabularBenchmark(root=path, name=args.dataset)
# dataset.materialize()
# # Only support classification training/eval for now.
# # TODO: support regression tasks.
# assert dataset.task_type.is_classification
# dataset = dataset.shuffle()
# # Split ratio following https://arxiv.org/abs/2207.08815
# # 70% is used for training. 30% of the remaining is used for validation.
# # The final reminder is used for testing.
# train_dataset, val_dataset, test_dataset = (
#     dataset[:0.7],
#     dataset[0.7:0.79],
#     dataset[0.79:],
# )

# # Set up data loaders
# train_tensor_frame = train_dataset.tensor_frame
# val_tensor_frame = val_dataset.tensor_frame
# test_tensor_frame = test_dataset.tensor_frame
# train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
#                           shuffle=True)
# val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
# test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

# # Set up model and optimizer
# model = Trompt(
#     channels=args.channels,
#     out_channels=dataset.num_classes,
#     num_prompts=args.num_prompts,
#     num_layers=args.num_layers,
#     col_stats=dataset.col_stats,
#     col_names_dict=train_tensor_frame.col_names_dict,
# ).to(device)
# model = torch.compile(model, dynamic=True) if args.compile else model
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# lr_scheduler = ExponentialLR(optimizer, gamma=0.95)


# def train(epoch: int) -> float:
#     model.train()
#     loss_accum = total_count = 0

#     for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
#         tf = tf.to(device)
#         # [batch_size, num_layers, num_classes]
#         out = model.forward_stacked(tf)
#         num_layers = out.size(1)
#         # [batch_size * num_layers, num_classes]
#         pred = out.view(-1, dataset.num_classes)
#         y = tf.y.repeat_interleave(num_layers)
#         # Layer-wise logit loss
#         loss = F.cross_entropy(pred, y)
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
#         pred_class = pred.argmax(dim=-1)
#         accum += float((tf.y == pred_class).sum())
#         total_count += len(tf.y)

#     return accum / total_count


# best_val_acc = 0
# best_test_acc = 0
# for epoch in range(1, args.epochs + 1):
#     train_loss = train(epoch)
#     train_acc = test(train_loader)
#     val_acc = test(val_loader)
#     test_acc = test(test_loader)
#     if best_val_acc < val_acc:
#         best_val_acc = val_acc
#         best_test_acc = test_acc
#     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
#           f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
#     lr_scheduler.step()

# print(f"Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}")


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
    
    # 設備設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 數據集加載和物化
    from torch_frame.datasets import Yandex
    dataset = Yandex(df=df, name=dataset_name, 
                     train_val_test_split_ratio=train_val_test_split_ratio, 
                     task_type=task_type, DS=False)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification
    
    # 數據集分割
    train_dataset, val_dataset, test_dataset = dataset.split()
    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    
    # 創建數據加載器
    from torch_frame.data import DataLoader
    batch_size = config.get('batch_size', 256)  # Trompt默認批次大小
    print(f"Batch size: {batch_size}")
    
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)
    
    # 確定輸出通道數
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
        'out_channels': out_channels,
        'device': device
    }

def encoding_fn(material_outputs, config):
    """
    階段2: Encoding - 創建多層編碼器
    
    輸入:
    - material_outputs: materialize_fn的輸出
    - config: 配置參數
    
    輸出:
    - 包含多層編碼器和提示向量的字典，可傳給columnwise_fn或自定義GNN
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
    
    # 獲取Trompt模型參數
    channels = config.get('channels', 128)
    num_prompts = config.get('num_prompts', 128)
    num_layers = config.get('num_layers', 6)
    print(f"Encoding with channels: {channels}, num_prompts: {num_prompts}, num_layers: {num_layers}")
    
    # 計算總列數
    num_cols = sum([len(col_names) for col_names in col_names_dict.values()])
    print(f"Total number of columns: {num_cols}")
    
    # 創建提示向量參數
    x_prompt = Parameter(torch.empty(num_prompts, channels, device=device).normal_(std=0.01))
    torch.nn.init.normal_(x_prompt, std=0.01)
    
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
    
    # 對批次數據進行編碼的函數
    def encode_batch(tf):
        batch_size = len(tf)
        layer_outputs = []
        
        # 為每一層執行編碼
        for encoder in encoders:
            # 編碼特徵 [batch_size, num_cols, channels]
            x, _ = encoder(tf)
            layer_outputs.append(x)
        
        return layer_outputs, batch_size
    
    # 返回編碼器、提示向量和相關信息
    return {
        'encoders': encoders,
        'x_prompt': x_prompt,
        'encode_batch': encode_batch,
        'num_cols': num_cols,
        'num_prompts': num_prompts,
        'channels': channels,
        'num_layers': num_layers,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_tensor_frame': train_tensor_frame,
        'val_tensor_frame': val_tensor_frame,
        'test_tensor_frame': test_tensor_frame,
        'out_channels': material_outputs['out_channels'],
        'is_classification': material_outputs['is_classification'],
        'is_binary_class': material_outputs['is_binary_class'],
        'metric_computer': material_outputs['metric_computer'],
        'metric': material_outputs['metric'],
        'device': device
    }



def columnwise_fn(encoding_outputs, config):
    """
    階段3: Column-wise Interaction - 創建TromptConv層處理列間交互
    
    輸入:
    - encoding_outputs: encoding_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 包含TromptConv層的字典，可傳給decoding_fn或自定義GNN
    """
    print("Executing columnwise_fn")
    
    # 從上一階段獲取數據
    encoders = encoding_outputs['encoders']
    x_prompt = encoding_outputs['x_prompt']
    encode_batch = encoding_outputs['encode_batch']
    num_cols = encoding_outputs['num_cols']
    num_prompts = encoding_outputs['num_prompts']
    channels = encoding_outputs['channels']
    num_layers = encoding_outputs['num_layers']
    device = encoding_outputs['device']
    
    # 創建TromptConv層列表
    trompt_convs = ModuleList([
        TromptConv(channels, num_cols, num_prompts).to(device)
        for _ in range(num_layers)
    ])
    
    # 所有TromptConv層重置參數
    for trompt_conv in trompt_convs:
        trompt_conv.reset_parameters()
    
    # 定義列間交互處理函數
    def process_batch_interaction(encoded_features, batch_size):
        """
        處理編碼後的特徵通過TromptConv進行列間交互
        
        參數:
        - encoded_features: 編碼後的特徵列表，每層一個張量
        - batch_size: 批次大小
        
        返回:
        - prompts_outputs: 各層處理後的提示向量列表
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
            prompts_outputs.append(updated_prompt)
        
        return prompts_outputs
    
    # 返回TromptConv層和相關信息
    return {
        'encoders': encoders,
        'trompt_convs': trompt_convs,
        'x_prompt': x_prompt,
        'encode_batch': encode_batch,
        'process_batch_interaction': process_batch_interaction,
        'num_cols': num_cols,
        'num_prompts': num_prompts,
        'channels': channels,
        'num_layers': num_layers,
        'train_loader': encoding_outputs['train_loader'],
        'val_loader': encoding_outputs['val_loader'],
        'test_loader': encoding_outputs['test_loader'],
        'train_tensor_frame': encoding_outputs['train_tensor_frame'],
        'val_tensor_frame': encoding_outputs['val_tensor_frame'],
        'test_tensor_frame': encoding_outputs['test_tensor_frame'],
        'out_channels': encoding_outputs['out_channels'],
        'is_classification': encoding_outputs['is_classification'],
        'is_binary_class': encoding_outputs['is_binary_class'],
        'metric_computer': encoding_outputs['metric_computer'],
        'metric': encoding_outputs['metric'],
        'device': encoding_outputs['device']
    }


def decoding_fn(columnwise_outputs, config):
    """
    階段4: Decoding - 創建TromptDecoder並訓練模型
    
    輸入:
    - columnwise_outputs: columnwise_fn的輸出或GNN的輸出
    - config: 配置參數
    
    輸出:
    - 訓練結果和最終模型
    """
    print("Executing decoding_fn")
    
    # 從上一階段獲取數據
    encoders = columnwise_outputs['encoders']
    trompt_convs = columnwise_outputs['trompt_convs']
    x_prompt = columnwise_outputs['x_prompt']
    encode_batch = columnwise_outputs['encode_batch']
    process_batch_interaction = columnwise_outputs['process_batch_interaction']
    num_prompts = columnwise_outputs['num_prompts']
    channels = columnwise_outputs['channels']
    num_layers = columnwise_outputs['num_layers']
    out_channels = columnwise_outputs['out_channels']
    train_loader = columnwise_outputs['train_loader']
    val_loader = columnwise_outputs['val_loader']
    test_loader = columnwise_outputs['test_loader']
    device = columnwise_outputs['device']
    is_classification = columnwise_outputs['is_classification']
    is_binary_class = columnwise_outputs['is_binary_class']
    metric_computer = columnwise_outputs['metric_computer']
    metric = columnwise_outputs['metric']
    
    # 創建TromptDecoder
    trompt_decoder = TromptDecoder(
        channels, 
        out_channels, 
        num_prompts
    ).to(device)
    trompt_decoder.reset_parameters()
    
    # 定義前向傳播函數 - 分層版本
    def forward_stacked(tf):
        """
        前向傳播 - 返回每一層的預測結果
        
        參數:
        - tf: 輸入TensorFrame
        
        返回:
        - stacked_out: 形狀為[batch_size, num_layers, out_channels]的堆疊預測結果
        """
        # 編碼特徵
        encoded_features, batch_size = encode_batch(tf)
        
        # 通過TromptConv進行列間交互
        prompts_outputs = process_batch_interaction(encoded_features, batch_size)
        
        # 對每層的提示向量應用解碼器生成預測
        outs = []
        for i in range(num_layers):
            # 獲取當前層的提示向量
            prompt = prompts_outputs[i]
            
            # 應用解碼器
            out = trompt_decoder(prompt)
            
            # 調整形狀為[batch_size, 1, out_channels]
            out = out.view(batch_size, 1, out_channels)
            outs.append(out)
        
        # 堆疊所有層的輸出
        stacked_out = torch.cat(outs, dim=1)
        return stacked_out
    
    # 定義前向傳播函數 - 平均版本
    def forward(tf):
        """
        前向傳播 - 返回所有層預測結果的平均
        
        參數:
        - tf: 輸入TensorFrame
        
        返回:
        - out: 形狀為[batch_size, out_channels]的預測結果
        """
        stacked_out = forward_stacked(tf)
        return stacked_out.mean(dim=1)
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.001)
    
    # 收集所有參數
    all_params = [x_prompt] + \
                 [p for encoder in encoders for p in encoder.parameters()] + \
                 [p for conv in trompt_convs for p in conv.parameters()] + \
                 list(trompt_decoder.parameters())
    
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
        
        loss_accum = total_count = 0
        
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            
            # 使用分層前向傳播
            out = forward_stacked(tf)
            
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
        
        metric_computer.reset()
        
        for tf in loader:
            tf = tf.to(device)
            # 使用平均前向傳播
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
        'encoders': encoders,
        'trompt_convs': trompt_convs,
        'trompt_decoder': trompt_decoder,
        'x_prompt': x_prompt,
        'forward': forward,
        'forward_stacked': forward_stacked
    }


def main(df, dataset_results, config):
    """
    主函數：按順序調用四個階段函數
    
    可用於在階段間插入GNN模型
    """
    print("Trompt - 四階段執行")
    
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