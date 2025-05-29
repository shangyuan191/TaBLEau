from __future__ import annotations
"""Reported (reproduced) accuracy (for multi-classification task), auc
(for binary classification task) and rmse (for regression task)
based on Table 1 of the paper https://arxiv.org/abs/2301.02819.
ExcelFormer uses the same train-validation-test split as the Yandex paper.
The reproduced results are based on Z-score Normalization, and the
reported ones are based on :class:`QuantileTransformer` preprocessing
in the Sklearn Python package. The above preprocessing is applied
to numerical features.

california_housing: 0.4587 (0.4550) mixup: feature, num_layers: 3,
gamma: 1.00, epochs: 300
jannis : 72.51 (72.80) mixup: feature
covtype: 97.17 (97.02) mixup: hidden
helena: 38.20 (37.68) mixup: feature
higgs_small: 80.75 (79.27) mixup: hidden
"""
import argparse
import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame.data.loader import DataLoader
from torch_frame.datasets.yandex import Yandex
from torch_frame.nn import ExcelFormer
from torch_frame.transforms import CatToNumTransform, MutualInformationSort
from tqdm import tqdm
import json
import os
import pandas as pd
import xlsxwriter
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList


from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList
import torch_frame
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder
from torch_frame.nn.encoder.stype_encoder import ExcelFormerEncoder
from torch_frame.nn.encoder.stypewise_encoder import (
    StypeEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.typing import NAStrategy
# classification_datasets = {
#     'adult', 'aloi', 'covtype', 'helena', 'higgs_small', 'jannis'
# }
# regression_datasets = {'california_housing', 'microsoft', 'yahoo', 'year'}


# dataset_strs = [
#     'adult', 'aloi', 'covtype', 'helena', 
#     'higgs_small', 'jannis', 
#     'california_housing', 'microsoft', 
#     'yahoo', 'year'
# ]
# def gen_best_result_report(filename):
#     with open(filename,'r',encoding='utf-8') as f:
#         lines=f.readlines()
#     datasets=[]
#     current_dataset=[]

#     for line in lines:
#         if line.strip()=="" and len(current_dataset)==7:
#             datasets.append(current_dataset)
#             current_dataset=[]
#         elif line.strip():
#             current_dataset.append(line.strip())

#     if len(current_dataset)==7:
#         datasets.append(current_dataset)

#     dataset_names=[]
#     dataset_sizes=[]
#     task_types=[]
#     train_val_test_split_ratios_1=[]
#     best_val_result_names_1=[]
#     best_val_result_metrics_1=[]
#     best_test_result_names_1=[]
#     best_test_result_metrics_1=[]
#     train_val_test_split_ratios_2=[]
#     best_val_result_names_2=[]
#     best_val_result_metrics_2=[]
#     best_test_result_names_2=[]
#     best_test_result_metrics_2=[]

#     for dataset in datasets:
#         dataset_name=dataset[0].split(":")[1].strip()
#         dataset_size=dataset[1].split(":")[1].strip()
#         task_type=dataset[2].split(":")[1].strip()
#         train_val_test_split_ratio_1=dataset[3].split(":")[1].strip()
#         best_val_result_1,best_test_result_1=dataset[4].split(",")[0],dataset[4].split(",")[1].strip()
#         best_val_result_name_1,best_val_result_metric_1=best_val_result_1.split(":")[0],best_val_result_1.split(":")[1]
#         best_test_result_name_1,best_test_result_metric_1=best_test_result_1.split(":")[0],best_test_result_1.split(":")[1]
#         train_val_test_split_ratio_2=dataset[5].split(":")[1].strip()
#         best_val_result_2,best_test_result_2=dataset[6].split(",")[0],dataset[6].split(",")[1].strip()
#         best_val_result_name_2,best_val_result_metric_2=best_val_result_2.split(":")[0],best_val_result_2.split(":")[1]
#         best_test_result_name_2,best_test_result_metric_2=best_test_result_2.split(":")[0],best_test_result_2.split(":")[1]
#         dataset_names.append(dataset_name)
#         dataset_sizes.append(dataset_size)
#         task_types.append(task_type)
#         train_val_test_split_ratios_1.append(train_val_test_split_ratio_1)
#         best_val_result_names_1.append(best_val_result_name_1)
#         best_val_result_metrics_1.append(best_val_result_metric_1)
#         best_test_result_names_1.append(best_test_result_name_1)
#         best_test_result_metrics_1.append(best_test_result_metric_1)
#         train_val_test_split_ratios_2.append(train_val_test_split_ratio_2)
#         best_val_result_names_2.append(best_val_result_name_2)
#         best_val_result_metrics_2.append(best_val_result_metric_2)
#         best_test_result_names_2.append(best_test_result_name_2)
#         best_test_result_metrics_2.append(best_test_result_metric_2)


#     data = {
#         'Dataset Name': dataset_names,
#         'Dataset Size': dataset_sizes,
#         'Task Type': task_types,
#         'Train/Val/Test Split Ratio 1': train_val_test_split_ratios_1,
#         'Best Val Result Name 1': best_val_result_names_1,
#         'Best Val Result Metric 1': best_val_result_metrics_1,
#         'Best Test Result Name 1': best_test_result_names_1,
#         'Best Test Result Metric 1': best_test_result_metrics_1,
#         'Train/Val/Test Split Ratio 2': train_val_test_split_ratios_2,
#         'Best Val Result Name 2': best_val_result_names_2,
#         'Best Val Result Metric 2': best_val_result_metrics_2,
#         'Best Test Result Name 2': best_test_result_names_2,
#         'Best Test Result Metric 2': best_test_result_metrics_2,
#         'Full sample -> FewShot(val)': ["+"+str(round(float(best_val_result_metric_2)-float(best_val_result_metric_1),4)) if float(best_val_result_metric_2)-float(best_val_result_metric_1)>=0 else float(best_val_result_metric_2)-float(best_val_result_metric_1) for best_val_result_metric_1,best_val_result_metric_2 in zip(best_val_result_metrics_1,best_val_result_metrics_2)],
#         'Full sample -> FewShot(test)': ["+"+str(round(float(best_test_result_metric_2)-float(best_test_result_metric_1),4)) if float(best_test_result_metric_2)-float(best_test_result_metric_1)>=0 else float(best_test_result_metric_2)-float(best_test_result_metric_1) for best_test_result_metric_1,best_test_result_metric_2 in zip(best_test_result_metrics_1,best_test_result_metrics_2)]
#     }

#     # 創建 DataFrame
#     df = pd.DataFrame(data)

#     # 使用 XlsxWriter 進行格式化並儲存為 Excel 檔案
#     excel_filename = './result/Best_result.xlsx'
#     with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
#         df.to_excel(writer, sheet_name='Best Results', index=False)

#         # 獲取 XlsxWriter workbook 和 worksheet 物件
#         workbook = writer.book
#         worksheet = writer.sheets['Best Results']

#         # 設置欄位寬度自動調整
#         for col_num, col in enumerate(df.columns):
#             column_len = max(df[col].astype(str).map(len).max(), len(col))
#             worksheet.set_column(col_num, col_num, column_len)

#         # 設置全部置中
#         center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
#         worksheet.set_column(0, len(df.columns) - 1, None, center_format)

#     print("Excel file created successfully.")
#     print(df)




# if __name__ == '__main__':
#     # gen_best_result_report('./result/Best_result.txt')

#     train_val_test_split_ratios=[[0.64,0.16,0.2],[0.05,0.15,0.8]]
#     dataset_info={}
#     with open(f'dataset_info.json','r',encoding='utf-8') as f:
#         dataset_info=json.load(f)
#     with open(f'./result/Best_result.txt','w') as f:
#         for dataset_name in dataset_info.keys():
#             dataset_type=dataset_info[dataset_name]['dataset_type']
#             task_type=dataset_info[dataset_name]['task_type']
#             f.write(f"Dataset name : {dataset_name}\n")
#             f.write(f"Dataset size : {dataset_type}\n")
#             f.write(f"Task type : {task_type}\n")
#             all_train_loss=[]
#             all_train_metric=[]
#             all_val_metric=[]
#             all_test_metric=[]
#             for train_val_test_split_ratio in train_val_test_split_ratios:
#                 parser = argparse.ArgumentParser()
#                 parser.add_argument('--dataset_size',type=str,default=f'{dataset_type}_datasets')
#                 parser.add_argument('--dataset', type=str, default=f"{dataset_name}")
#                 parser.add_argument('--mixup', type=str, default=None,
#                                     choices=[None, 'feature', 'hidden'])
#                 parser.add_argument('--channels', type=int, default=256)
#                 parser.add_argument('--batch_size', type=int, default=512)
#                 parser.add_argument('--num_heads', type=int, default=4)
#                 parser.add_argument('--num_layers', type=int, default=5)
#                 parser.add_argument('--lr', type=float, default=0.001)
#                 parser.add_argument('--gamma', type=float, default=0.95)
#                 parser.add_argument('--epochs', type=int, default=200)
#                 parser.add_argument('--compile', action='store_true')
#                 args = parser.parse_args()

#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                 print(device)
#                 print(args.dataset)
#                 path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',args.dataset_size,
#                                 args.dataset)
#                 print(path)
#                 dataset = Yandex(root=path, name=args.dataset,train_val_test_split_ratio=train_val_test_split_ratio)
#                 dataset.materialize()
#                 print(len(dataset))
#                 print(type(dataset))
#                 train_dataset, val_dataset, test_dataset = dataset.split()
#                 print(len(train_dataset))
#                 print(type(train_dataset))
#                 print(len(val_dataset))
#                 print(type(val_dataset))
#                 print(len(test_dataset))
#                 print(type(test_dataset))
#                 train_tensor_frame = train_dataset.tensor_frame
#                 val_tensor_frame = val_dataset.tensor_frame
#                 test_tensor_frame = test_dataset.tensor_frame

#                 # CategoricalCatBoostEncoder encodes the categorical features
#                 # into numerical features with CatBoostEncoder.
#                 categorical_transform = CatToNumTransform()
#                 categorical_transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)

#                 train_tensor_frame = categorical_transform(train_tensor_frame)
#                 val_tensor_frame = categorical_transform(val_tensor_frame)
#                 test_tensor_frame = categorical_transform(test_tensor_frame)
#                 col_stats = categorical_transform.transformed_stats

#                 # MutualInformationSort sorts the features based on mutual
#                 # information.
#                 mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)

#                 mutual_info_sort.fit(train_tensor_frame, col_stats)
#                 train_tensor_frame = mutual_info_sort(train_tensor_frame)
#                 val_tensor_frame = mutual_info_sort(val_tensor_frame)
#                 test_tensor_frame = mutual_info_sort(test_tensor_frame)

#                 train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
#                                         shuffle=True)
#                 val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
#                 test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

#                 is_classification = dataset.task_type.is_classification

#                 if is_classification:
#                     out_channels = dataset.num_classes
#                 else:
#                     out_channels = 1

#                 is_binary_class = is_classification and out_channels == 2

#                 if is_binary_class:
#                     metric_computer = AUROC(task='binary')
#                 elif is_classification:
#                     metric_computer = Accuracy(task='multiclass', num_classes=out_channels)
#                 else:
#                     metric_computer = MeanSquaredError()
#                 metric_computer = metric_computer.to(device)

#                 model = ExcelFormer(
#                     in_channels=args.channels,
#                     out_channels=out_channels,
#                     num_layers=args.num_layers,
#                     num_cols=train_tensor_frame.num_cols,
#                     num_heads=args.num_heads,
#                     residual_dropout=0.,
#                     diam_dropout=0.3,
#                     aium_dropout=0.,
#                     mixup=args.mixup,
#                     col_stats=mutual_info_sort.transformed_stats,
#                     col_names_dict=train_tensor_frame.col_names_dict,
#                 ).to(device)
#                 model = torch.compile(model, dynamic=True) if args.compile else model
#                 optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#                 lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma)


#                 def train(epoch: int) -> float:
#                     model.train()
#                     loss_accum = total_count = 0

#                     for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
#                         tf = tf.to(device)
#                         # Train with FEAT-MIX or HIDDEN-MIX
#                         pred_mixedup, y_mixedup = model(tf, mixup_encoded=True)
#                         if is_classification:
#                             # Softly mixed one-hot labels
#                             loss = F.cross_entropy(pred_mixedup, y_mixedup)
#                         else:
#                             loss = F.mse_loss(pred_mixedup.view(-1), y_mixedup.view(-1))
#                         optimizer.zero_grad()
#                         loss.backward()
#                         loss_accum += float(loss) * len(y_mixedup)
#                         total_count += len(y_mixedup)
#                         optimizer.step()
#                     return loss_accum / total_count


#                 @torch.no_grad()
#                 def test(loader: DataLoader) -> float:
#                     model.eval()
#                     metric_computer.reset()
#                     for tf in loader:
#                         tf = tf.to(device)
#                         pred = model(tf)
#                         if is_binary_class:
#                             metric_computer.update(pred[:, 1], tf.y)
#                         elif is_classification:
#                             pred_class = pred.argmax(dim=-1)
#                             metric_computer.update(pred_class, tf.y)
#                         else:
#                             metric_computer.update(pred.view(-1), tf.y.view(-1))

#                     if is_classification:
#                         return metric_computer.compute().item()
#                     else:
#                         return metric_computer.compute().item()**0.5


#                 if is_classification:
#                     metric = 'Acc' if not is_binary_class else 'AUC'
#                     best_val_metric = 0
#                     best_test_metric = 0
#                 else:
#                     metric = 'RMSE'
#                     best_val_metric = float('inf')
#                     best_test_metric = float('inf')
#                 now_train_loss=[]
#                 now_train_metric=[]
#                 now_val_metric=[]
#                 now_test_metric=[]

                
#                 epochs=range(1,args.epochs+1)
#                 for epoch in range(1, args.epochs + 1):
#                     train_loss = train(epoch)
#                     train_metric = test(train_loader)
#                     val_metric = test(val_loader)
#                     test_metric = test(test_loader)
                    
#                     now_train_loss.append(train_loss)
#                     now_train_metric.append(train_metric)
#                     now_val_metric.append(val_metric)
#                     now_test_metric.append(test_metric)

#                     if is_classification and val_metric > best_val_metric:
#                         best_val_metric = val_metric
#                         best_test_metric = test_metric
#                     elif not is_classification and val_metric < best_val_metric:
#                         best_val_metric = val_metric
#                         best_test_metric = test_metric

#                     print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
#                         f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')
#                     lr_scheduler.step()

#                 print(f'Best Val {metric}: {best_val_metric:.4f}, '
#                     f'Best Test {metric}: {best_test_metric:.4f}')
                
                
#                 f.write(f"Train_val_test_split_ratio : {train_val_test_split_ratio}\n")
#                 f.write(f'Best Val {metric}: {best_val_metric:.4f}, 'f'Best Test {metric}: {best_test_metric:.4f}\n')
#                 all_train_loss.append(now_train_loss)
#                 all_train_metric.append(now_train_metric)
#                 all_val_metric.append(now_val_metric)
#                 all_test_metric.append(now_test_metric)
#             # 創建圖表，2 行（每行兩張圖），每個比例會有兩個子圖
#             fig, axes = plt.subplots(len(train_val_test_split_ratios), 2, figsize=(12, 8))

#             for i, ratio in enumerate(train_val_test_split_ratios):
#                 # 第一張子圖：訓練損失
#                 axes[i, 0].plot(epochs, all_train_loss[i], label=f'Train Loss (Ratio: {ratio})')
#                 axes[i, 0].set_xlabel('Epoch')
#                 axes[i, 0].set_ylabel('Loss')
#                 axes[i, 0].set_title(f'Training Loss Over Epochs (Ratio: {ratio})')
#                 axes[i, 0].legend()

#                 # 第二張子圖：metric（包括訓練、驗證、測試的 metric）
#                 axes[i, 1].plot(epochs, all_train_metric[i], label=f'Train {metric} (Ratio: {ratio})')
#                 axes[i, 1].plot(epochs, all_val_metric[i], label=f'Val {metric} (Ratio: {ratio})')
#                 axes[i, 1].plot(epochs, all_test_metric[i], label=f'Test {metric} (Ratio: {ratio})')
#                 axes[i, 1].set_xlabel('Epoch')
#                 axes[i, 1].set_ylabel(f'{metric}')
#                 axes[i, 1].set_title(f'{metric} Over Epochs (Ratio: {ratio})')
#                 axes[i, 1].legend()

#             # 調整布局，避免重疊
#             plt.tight_layout()
#             result_path=f"./result/{dataset_type}_datasets/{task_type}/{dataset_name}/"
#             os.makedirs(os.path.dirname(result_path), exist_ok=True)
#             plt.savefig(f"{result_path}comparison_{args.dataset}.png")

#             f.write('\n\n')

    # gen_best_result_report('./result/Best_result.txt')





























def feature_mixup(
    x: Tensor,
    y: Tensor,
    num_classes: int,
    beta: float | Tensor = 0.5,
    mixup_type: str | None = None,
    mi_scores: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Mixup input numerical feature tensor :obj:`x` by swapping some
    feature elements of two shuffled sample samples. The shuffle rates for
    each row is sampled from the Beta distribution. The target `y` is also
    linearly mixed up.

    Args:
        x (Tensor): The input numerical feature.
        y (Tensor): The target.
        num_classes (int): Number of classes.
        beta (float): The concentration parameter of the Beta distribution.
            (default: :obj:`0.5`)
        mixup_type (str, optional): The mixup methods. No mixup if set to
            :obj:`None`, options `feature` and `hidden` are `FEAT-MIX`
            (mixup at feature dimension) and `HIDDEN-MIX` (mixup at
            hidden dimension) proposed in ExcelFormer paper.
            (default: :obj:`None`)
        mi_scores (Tensor, optional): Mutual information scores only used in
            the mixup weight calculation for `FEAT-MIX`.
            (default: :obj:`None`)

    Returns:
        x_mixedup (Tensor): The mixedup numerical feature.
        y_mixedup (Tensor): Transformed target of size
            :obj:`[batch_size, num_classes]`
    """
    assert num_classes > 0
    assert mixup_type in [None, 'feature', 'hidden']

    beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
    beta_distribution = torch.distributions.beta.Beta(beta, beta)
    shuffle_rates = beta_distribution.sample(torch.Size((len(x), 1)))
    shuffled_idx = torch.randperm(len(x), device=x.device)
    assert x.ndim == 3, """
    FEAT-MIX or HIDDEN-MIX is for encoded numerical features
    of size [batch_size, num_cols, in_channels]."""
    b, f, d = x.shape
    if mixup_type == 'feature':
        assert mi_scores is not None
        mi_scores = mi_scores.to(x.device)
        # Hard mask (feature dimension)
        mixup_mask = torch.rand(torch.Size((b, f)),
                                device=x.device) < shuffle_rates
        # L1 normalized mutual information scores
        norm_mi_scores = mi_scores / mi_scores.sum()
        # Mixup weights
        lam = torch.sum(
            norm_mi_scores.unsqueeze(0) * mixup_mask, dim=1, keepdim=True)
        mixup_mask = mixup_mask.unsqueeze(2)
    elif mixup_type == 'hidden':
        # Hard mask (hidden dimension)
        mixup_mask = torch.rand(torch.Size((b, d)),
                                device=x.device) < shuffle_rates
        mixup_mask = mixup_mask.unsqueeze(1)
        # Mixup weights
        lam = shuffle_rates
    else:
        # No mixup
        mixup_mask = torch.ones_like(x, dtype=torch.bool)
        # Fake mixup weights
        lam = torch.ones_like(shuffle_rates)
    x_mixedup = mixup_mask * x + ~mixup_mask * x[shuffled_idx]

    y_shuffled = y[shuffled_idx]
    if num_classes == 1:
        # Regression task or binary classification
        lam = lam.squeeze(1)
        y_mixedup = lam * y + (1 - lam) * y_shuffled
    else:
        # Classification task
        one_hot_y = F.one_hot(y, num_classes=num_classes)
        one_hot_y_shuffled = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mixedup = (lam * one_hot_y + (1 - lam) * one_hot_y_shuffled)
    return x_mixedup, y_mixedup


# class ExcelFormer(Module):
#     r"""The ExcelFormer model introduced in the
#     `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
#     <https://arxiv.org/abs/2301.02819>`_ paper.

#     ExcelFormer first converts the categorical features with a target
#     statistics encoder (i.e., :class:`CatBoostEncoder` in the paper)
#     into numerical features. Then it sorts the numerical features
#     with mutual information sort. So the model itself limits to
#     numerical features.

#     .. note::

#         For an example of using ExcelFormer, see `examples/excelformer.py
#         <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
#         excelformer.py>`_.

#     Args:
#         in_channels (int): Input channel dimensionality
#         out_channels (int): Output channels dimensionality
#         num_cols (int): Number of columns
#         num_layers (int): Number of
#             :class:`torch_frame.nn.conv.ExcelFormerConv` layers.
#         num_heads (int): Number of attention heads used in :class:`DiaM`
#         col_stats(dict[str,dict[:class:`torch_frame.data.stats.StatType`,Any]]):
#              A dictionary that maps column name into stats.
#              Available as :obj:`dataset.col_stats`.
#         col_names_dict (dict[:obj:`torch_frame.stype`, list[str]]): A
#             dictionary that maps stype to a list of column names. The column
#             names are sorted based on the ordering that appear in
#             :obj:`tensor_frame.feat_dict`. Available as
#             :obj:`tensor_frame.col_names_dict`.
#         stype_encoder_dict
#             (dict[:class:`torch_frame.stype`,
#             :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
#             A dictionary mapping stypes into their stype encoders.
#             (default: :obj:`None`, will call :obj:`ExcelFormerEncoder()`
#             for numerical feature)
#         diam_dropout (float, optional): diam_dropout. (default: :obj:`0.0`)
#         aium_dropout (float, optional): aium_dropout. (default: :obj:`0.0`)
#         residual_dropout (float, optional): residual dropout.
#             (default: :obj:`0.0`)
#         mixup (str, optional): mixup type.
#             :obj:`None`, :obj:`feature`, or :obj:`hidden`.
#             (default: :obj:`None`)
#         beta (float, optional): Shape parameter for beta distribution to
#                 calculate shuffle rate in mixup. Only useful when `mixup` is
#                 not :obj:`None`. (default: :obj:`0.5`)
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         num_cols: int,
#         num_layers: int,
#         num_heads: int,
#         col_stats: dict[str, dict[StatType, Any]],
#         col_names_dict: dict[torch_frame.stype, list[str]],
#         stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
#         | None = None,
#         diam_dropout: float = 0.0,
#         aium_dropout: float = 0.0,
#         residual_dropout: float = 0.0,
#         mixup: str | None = None,
#         beta: float = 0.5,
#     ) -> None:
#         super().__init__()
#         if num_layers <= 0:
#             raise ValueError(
#                 f"num_layers must be a positive integer (got {num_layers})")

#         assert mixup in [None, 'feature', 'hidden']

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if col_names_dict.keys() != {stype.numerical}:
#             raise ValueError("ExcelFormer only accepts numerical "
#                              "features.")

#         if stype_encoder_dict is None:
#             stype_encoder_dict = {
#                 stype.numerical:
#                 ExcelFormerEncoder(out_channels, na_strategy=NAStrategy.MEAN)
#             }

#         self.excelformer_encoder = StypeWiseFeatureEncoder(
#             out_channels=self.in_channels,
#             col_stats=col_stats,
#             col_names_dict=col_names_dict,
#             stype_encoder_dict=stype_encoder_dict,
#         )
#         self.excelformer_convs = ModuleList([
#             ExcelFormerConv(in_channels, num_cols, num_heads, diam_dropout,
#                             aium_dropout, residual_dropout)
#             for _ in range(num_layers)
#         ])
#         self.excelformer_decoder = ExcelFormerDecoder(in_channels,
#                                                       out_channels, num_cols)
#         self.reset_parameters()
#         self.mixup = mixup
#         self.beta = beta

#     def reset_parameters(self) -> None:
#         self.excelformer_encoder.reset_parameters()
#         for excelformer_conv in self.excelformer_convs:
#             excelformer_conv.reset_parameters()
#         self.excelformer_decoder.reset_parameters()

#     def forward(
#         self,
#         tf: TensorFrame,
#         mixup_encoded: bool = False,
#     ) -> Tensor | tuple[Tensor, Tensor]:
#         r"""Transform :class:`TensorFrame` object into output embeddings. If
#         :obj:`mixup_encoded` is :obj:`True`, it produces the output embeddings
#         together with the mixed-up targets in :obj:`self.mixup` manner.

#         Args:
#             tf (:class:`torch_frame.TensorFrame`): Input :class:`TensorFrame`
#                 object.
#             mixup_encoded (bool): Whether to mixup on encoded numerical
#                 features, i.e., `FEAT-MIX` and `HIDDEN-MIX`.
#                 (default: :obj:`False`)

#         Returns:
#             torch.Tensor | tuple[Tensor, Tensor]: The output embeddings of size
#                 [batch_size, out_channels]. If :obj:`mixup_encoded` is
#                 :obj:`True`, return the mixed-up targets of size
#                 [batch_size, num_classes] as well.
#         """
#         # print(f"tf.col_names_dict")
#         # print(f"len of col_names_dict.keys():{len(tf.col_names_dict.keys())}")
#         # print(tf.col_names_dict.keys())
#         # for key in tf.col_names_dict.keys():
#         #     print(f"key:{key}")
#         #     print("type of key:",type(key))
#         #     print(f"type of train_tensor_frame.col_names_dict[{key}]:{type(tf.col_names_dict[key])}")
#         #     print(f"len of train_tensor_frame.col_names_dict[{key}]:{len(tf.col_names_dict[key])}")
#         #     print(f"train_tensor_frame.col_names_dict[{key}]:{tf.col_names_dict[key]}")
#         # print(f"feat_dict")
#         # print(f"len of feat_dict.keys():{len(tf.feat_dict.keys())}")
#         # print(tf.feat_dict.keys())
#         # for key in tf.feat_dict.keys():
#         #     print(f"key:{key}")
#         #     print("type of key:",type(key))
#         #     print(f"type of tf.feat_dict[{key}]:{type(tf.feat_dict[key])}")
#         #     print(f"len of tf.feat_dict[{key}]:{len(tf.feat_dict[key])}")
#         #     print(f"shape of tf.feat_dict[{key}]:{tf.feat_dict[key].shape}")

#         # print(f"y.shape:{tf.y.shape}")
#         x, _ = self.excelformer_encoder(tf)
#         # print(f"x.shape: {x.shape}")
#         # FEAT-MIX or HIDDEN-MIX is compatible with `torch.compile`
#         if mixup_encoded:
#             assert tf.y is not None
#             x, y_mixedup = feature_mixup(
#                 x,
#                 tf.y,
#                 num_classes=self.out_channels,
#                 beta=self.beta,
#                 mixup_type=self.mixup,
#                 mi_scores=getattr(tf, 'mi_scores', None),
#             )
#             # print(f"type(x): {type(x)}")
#             # print(f"x.shape: {x.shape}")
#             # print(f"type(y_mixedup): {type(y_mixedup)}")
#             # print(f"y_mixedup.shape: {y_mixedup.shape}\n\n")
#         # print(f"len of excelformer_convs:{len(self.excelformer_convs)}")
#         for excelformer_conv in self.excelformer_convs:
#             x = excelformer_conv(x)
#         out = self.excelformer_decoder(x)

#         if mixup_encoded:
#             return out, y_mixedup
#         return out
















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
    # 數據集加載和物化
    dataset = Yandex(df=df, name=dataset_name, 
                     train_val_test_split_ratio=train_val_test_split_ratio, 
                     task_type=task_type, DS=False)
    dataset.materialize()
    
    # 數據集分割
    train_dataset, val_dataset, test_dataset = dataset.split()
    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    
    # 類別特徵轉換為數值特徵 - 保留在materialize階段確保數據準備完整
    categorical_transform = CatToNumTransform()
    categorical_transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)
    
    train_tensor_frame = categorical_transform(train_tensor_frame)
    val_tensor_frame = categorical_transform(val_tensor_frame)
    test_tensor_frame = categorical_transform(test_tensor_frame)
    col_stats = categorical_transform.transformed_stats
    
    # 基於互信息的特徵排序 - 保留在materialize階段確保數據準備完整
    mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
    mutual_info_sort.fit(train_tensor_frame, col_stats)
    
    train_tensor_frame = mutual_info_sort(train_tensor_frame)
    val_tensor_frame = mutual_info_sort(val_tensor_frame)
    test_tensor_frame = mutual_info_sort(test_tensor_frame)
    print(f"Train TensorFrame shape: {train_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Val TensorFrame shape: {val_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Test TensorFrame shape: {test_tensor_frame.feat_dict[stype.numerical].shape}")

    # 獲取分類任務信息
    is_classification = dataset.task_type.is_classification
    if is_classification:
        out_channels = dataset.num_classes
    else:
        out_channels = 1
    is_binary_class = is_classification and out_channels == 2
    
    # 創建數據加載器
    batch_size = config.get('batch_size', 512)
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)
    
    # 設置評估指標
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
        'col_stats': col_stats,
        'mutual_info_sort': mutual_info_sort,
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
    mutual_info_sort = material_outputs['mutual_info_sort']
    device = material_outputs['device']
    print(f"Input Train TensorFrame shape: {train_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Input Val TensorFrame shape: {val_tensor_frame.feat_dict[stype.numerical].shape}")
    print(f"Input Test TensorFrame shape: {test_tensor_frame.feat_dict[stype.numerical].shape}")
    # 獲取模型參數
    channels = config.get('channels', 256)
    mixup = config.get('mixup', None)
    beta = config.get('beta', 0.5)
    
    # 創建ExcelFormer的編碼器部分
    stype_encoder_dict = {
        stype.numerical: ExcelFormerEncoder(channels, na_strategy=NAStrategy.MEAN)
    }
    
    encoder = StypeWiseFeatureEncoder(
        out_channels=channels,
        col_stats=mutual_info_sort.transformed_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
    
    # 對訓練、驗證和測試數據進行編碼處理
    # 這裡我們將預先處理每個批次，生成編碼後的嵌入
    # 這不是必需的，但對於想在此階段插入GNN的情況很有用
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
            x, _ = encoder(tf)  # 只獲取編碼後的嵌入
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
    
    # 合併以便後續處理
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
        'mixup': mixup,
        'beta': beta,
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
    train_tensor_frame = encoding_outputs['train_tensor_frame']
    channels = encoding_outputs['channels']
    device = encoding_outputs['device']
    print(f"Train Embeddings shape: {train_embeddings.shape}")
    print(f"Val Embeddings shape: {val_embeddings.shape}")
    print(f"Test Embeddings shape: {test_embeddings.shape}")
    print(f"Train Labels shape: {train_labels.shape}")
    print(f"Val Labels shape: {val_labels.shape}")
    print(f"Test Labels shape: {test_labels.shape}")
    # 獲取列間交互參數
    num_layers = config.get('num_layers', 5)
    num_heads = config.get('num_heads', 4)
    
    # 創建ExcelFormer的列間交互層
    convs = ModuleList([
        ExcelFormerConv(
            channels, 
            train_tensor_frame.num_cols, 
            num_heads, 
            diam_dropout=0.3,
            aium_dropout=0.0,
            residual_dropout=0.0
        ).to(device)
        for _ in range(num_layers)
    ])
    
    # 對編碼後的嵌入應用列間交互層
    # 處理訓練、驗證和測試數據
    train_conv_outputs = train_embeddings
    val_conv_outputs = val_embeddings
    test_conv_outputs = test_embeddings
    
    with torch.no_grad():
        # 處理訓練數據
        for conv in convs:
            train_conv_outputs = conv(train_conv_outputs)
        
        # 處理驗證數據
        for conv in convs:
            val_conv_outputs = conv(val_conv_outputs)
        
        # 處理測試數據
        for conv in convs:
            test_conv_outputs = conv(test_conv_outputs)
    print(f"Train Conv Outputs shape: {train_conv_outputs.shape}")
    print(f"Val Conv Outputs shape: {val_conv_outputs.shape}")
    print(f"Test Conv Outputs shape: {test_conv_outputs.shape}")

    # 返回列間交互結果和相關信息 - 這些都是decoding_fn的輸入
    return {
        'convs': convs,
        'train_conv_outputs': train_conv_outputs,
        'val_conv_outputs': val_conv_outputs,
        'test_conv_outputs': test_conv_outputs,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'train_loader': encoding_outputs['train_loader'],
        'val_loader': encoding_outputs['val_loader'],
        'test_loader': encoding_outputs['test_loader'],
        'train_tensor_frame': train_tensor_frame,
        'encoder': encoding_outputs['encoder'],
        'mixup': encoding_outputs['mixup'],
        'beta': encoding_outputs['beta'],
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
    train_conv_outputs = columnwise_outputs['train_conv_outputs']
    val_conv_outputs = columnwise_outputs['val_conv_outputs']
    test_conv_outputs = columnwise_outputs['test_conv_outputs']
    train_labels = columnwise_outputs['train_labels']
    val_labels = columnwise_outputs['val_labels']
    test_labels = columnwise_outputs['test_labels']
    train_tensor_frame = columnwise_outputs['train_tensor_frame']
    train_loader = columnwise_outputs['train_loader']
    val_loader = columnwise_outputs['val_loader']
    test_loader = columnwise_outputs['test_loader']
    encoder = columnwise_outputs['encoder']
    convs = columnwise_outputs['convs']
    channels = columnwise_outputs['channels']
    out_channels = columnwise_outputs['out_channels']
    device = columnwise_outputs['device']
    mixup = columnwise_outputs['mixup']
    beta = columnwise_outputs['beta']
    is_classification = columnwise_outputs['is_classification']
    is_binary_class = columnwise_outputs['is_binary_class']
    metric_computer = columnwise_outputs['metric_computer']
    metric = columnwise_outputs['metric']
    print(f"Train Conv Outputs shape: {train_conv_outputs.shape}")
    print(f"Val Conv Outputs shape: {val_conv_outputs.shape}")
    print(f"Test Conv Outputs shape: {test_conv_outputs.shape}")
    # 創建解碼器
    decoder = ExcelFormerDecoder(
        channels, 
        out_channels, 
        train_tensor_frame.num_cols
    ).to(device)
    
    # 模擬完整的ExcelFormer前向傳播
    def model_forward(tf, mixup_encoded=False):
        # 編碼階段
        x, _ = encoder(tf)
        
        # 如需mixup，則在編碼後應用
        if mixup_encoded and mixup is not None:
            assert tf.y is not None
            x, y_mixedup = feature_mixup(
                x,
                tf.y,
                num_classes=out_channels,
                beta=beta,
                mixup_type=mixup,
                mi_scores=getattr(tf, 'mi_scores', None),
            )
        else:
            y_mixedup = tf.y  # 如果沒有 mixup，使用原始標籤
        
        # 列間交互階段
        for conv in convs:
            x = conv(x)
        
        # 解碼階段
        out = decoder(x)
        
        return out, y_mixedup  # 始終返回兩個值
    
    # 設置優化器和學習率調度器
    lr = config.get('lr', 0.001)
    gamma = config.get('gamma', 0.95)
    
    # 收集所有參數
    all_params = list(encoder.parameters()) + [p for conv in convs for p in conv.parameters()] + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    # 定義訓練函數
    def train(epoch):
        # 設置為訓練模式
        encoder.train()
        for conv in convs:
            conv.train()
        decoder.train()
        
        loss_accum = total_count = 0
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            # 使用FEAT-MIX或HIDDEN-MIX進行訓練
            pred_mixedup, y_mixedup = model_forward(tf, mixup_encoded=True)
            # print(f"pred_mixedup.shape: {pred_mixedup.shape}")
            # print(f"y_mixedup.shape: {y_mixedup.shape}")
            # print(f"type(pred_mixedup): {type(pred_mixedup)}")
            # print(f"type(y_mixedup): {type(y_mixedup)}")
            # print(f"len of pred_mixedup: {len(pred_mixedup)}")
            # print(f"len of y_mixedup: {len(y_mixedup)}")
            # print(f"pred_mixedup: {pred_mixedup}")
            # print(f"y_mixedup: {y_mixedup}")
            if is_classification:
                # 軟混合的one-hot標籤
                loss = F.cross_entropy(pred_mixedup, y_mixedup)
            else:
                loss = F.mse_loss(pred_mixedup.view(-1), y_mixedup.view(-1))
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(y_mixedup)
            total_count += len(y_mixedup)
            optimizer.step()
        return loss_accum / total_count
    
    # 定義測試函數
    @torch.no_grad()
    def test(loader):
        # 設置為評估模式
        encoder.eval()
        for conv in convs:
            conv.eval()
        decoder.eval()
        
        metric_computer.reset()
        
        for tf in loader:
            tf = tf.to(device)
            pred, _ = model_forward(tf)
            
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
    epochs = config.get('epochs', 200)
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
        
        lr_scheduler.step()
    
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
        'convs': convs,
        'decoder': decoder,
        'model_forward': model_forward  # 返回模型前向函數以便後續使用
    }

def main(df, dataset_results, config):
    """
    主函數：按順序調用四個階段函數
    
    該函數也可以作為在階段間插入GNN的範例
    """
    print("ExcelFormer - 四階段執行")
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
        # print(f"results: {results}")
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