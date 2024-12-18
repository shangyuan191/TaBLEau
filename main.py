import os
import csv
import sys
import numpy as np
import pandas as pd
import argparse
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
import xlsxwriter
import sys
import os
from sklearn.model_selection import train_test_split
# 將 project_root 路徑插入到 sys.path 的最前面
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'torch_frame')))
def create_dir(path):
    """創建資料夾路徑"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_metrics_to_csv(csv_file_path, row_data, headers):
    """將數據儲存到 CSV 檔案"""
    write_headers = not os.path.exists(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_headers:
            writer.writeheader()
        writer.writerow(row_data)
if __name__=="__main__":
    dataset_sizes=['small','large']
    task_types=['binclass','multiclass','regression']
    # dataset_sizes=['small']
    # task_types=['binclass']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_val_test_split_ratio=[0.64,0.16,0.2]
    # train_val_test_split_ratio = [0.05, 0.15, 0.8]

    train_val_test_split_ratio_str = "_".join(map(str, train_val_test_split_ratio))
    # 統一的 CSV 檔路徑
    csv_result_path = f'./result/ExcelFormer/{train_val_test_split_ratio_str}.csv'
    headers = ['dataset_name', 'dataset_size', 'task_type', 'best_val_metric', 'test_result']

    # dataset_size='small'
    # task_type='binclass'
    for dataset_size in dataset_sizes:
        for task_type in task_types:
            for dataset_name in os.listdir(f'./datasets/{dataset_size}_datasets/{task_type}'):
                print(f"\nDataset_name : {dataset_name}")
                parser = argparse.ArgumentParser()
                parser.add_argument('--dataset', type=str, default=f"{dataset_name}")
                parser.add_argument('--mixup', type=str, default=None,
                                    choices=[None, 'feature', 'hidden'])
                parser.add_argument('--channels', type=int, default=256)
                parser.add_argument('--batch_size', type=int, default=512)
                parser.add_argument('--num_heads', type=int, default=4)
                parser.add_argument('--num_layers', type=int, default=5)
                parser.add_argument('--lr', type=float, default=0.001)
                parser.add_argument('--gamma', type=float, default=0.95)
                parser.add_argument('--epochs', type=int, default=200)
                parser.add_argument('--compile', action='store_true')
                args = parser.parse_args()
                csv_path=f'./datasets/{dataset_size}_datasets/{task_type}/{dataset_name}/{dataset_name}.csv'
                dataset_path=f'./DS/all/{dataset_name}'
                df=pd.read_csv(csv_path)
                dataset = Yandex(df=df, name=args.dataset,train_val_test_split_ratio=train_val_test_split_ratio,task_type=task_type,DS=False)
                dataset.materialize()
                train_dataset, val_dataset, test_dataset = dataset.split()
                train_tensor_frame = train_dataset.tensor_frame
                val_tensor_frame = val_dataset.tensor_frame
                test_tensor_frame = test_dataset.tensor_frame
                # CategoricalCatBoostEncoder encodes the categorical features
                # into numerical features with CatBoostEncoder.
                categorical_transform = CatToNumTransform()
                categorical_transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)

                train_tensor_frame = categorical_transform(train_tensor_frame)
                val_tensor_frame = categorical_transform(val_tensor_frame)
                test_tensor_frame = categorical_transform(test_tensor_frame)
                col_stats = categorical_transform.transformed_stats

                # MutualInformationSort sorts the features based on mutual
                # information.
                mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)

                mutual_info_sort.fit(train_tensor_frame, col_stats)
                train_tensor_frame = mutual_info_sort(train_tensor_frame)
                val_tensor_frame = mutual_info_sort(val_tensor_frame)
                test_tensor_frame = mutual_info_sort(test_tensor_frame)

                train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                                        shuffle=True)
                val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
                test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

                is_classification = dataset.task_type.is_classification

                if is_classification:
                    out_channels = dataset.num_classes
                else:
                    out_channels = 1

                is_binary_class = is_classification and out_channels == 2

                if is_binary_class:
                    metric_computer = AUROC(task='binary')
                elif is_classification:
                    metric_computer = Accuracy(task='multiclass', num_classes=out_channels)
                else:
                    metric_computer = MeanSquaredError()
                metric_computer = metric_computer.to(device)

                model = ExcelFormer(
                    in_channels=args.channels,
                    out_channels=out_channels,
                    num_layers=args.num_layers,
                    num_cols=train_tensor_frame.num_cols,
                    num_heads=args.num_heads,
                    residual_dropout=0.,
                    diam_dropout=0.3,
                    aium_dropout=0.,
                    mixup=args.mixup,
                    col_stats=mutual_info_sort.transformed_stats,
                    col_names_dict=train_tensor_frame.col_names_dict,
                ).to(device)
                model = torch.compile(model, dynamic=True) if args.compile else model
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma)

                result_dir = f'./result/ExcelFormer/{dataset_size}_datasets/{task_type}/{dataset_name}/{train_val_test_split_ratio_str}'
                create_dir(result_dir)
                train_losses, train_metrics, val_metrics = [], [], []


                def train(epoch: int) -> float:
                    model.train()
                    loss_accum = total_count = 0

                    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
                        tf = tf.to(device)
                        # Train with FEAT-MIX or HIDDEN-MIX
                        pred_mixedup, y_mixedup = model(tf, mixup_encoded=True)
                        if is_classification:
                            # Softly mixed one-hot labels
                            loss = F.cross_entropy(pred_mixedup, y_mixedup)
                        else:
                            loss = F.mse_loss(pred_mixedup.view(-1), y_mixedup.view(-1))
                        optimizer.zero_grad()
                        loss.backward()
                        loss_accum += float(loss) * len(y_mixedup)
                        total_count += len(y_mixedup)
                        optimizer.step()
                    return loss_accum / total_count


                @torch.no_grad()
                def test(loader: DataLoader) -> float:
                    model.eval()
                    probabilities = []
                    metric_computer.reset()
                    for tf in loader:
                        tf = tf.to(device)
                        pred = model(tf)
                        if is_binary_class:
                            metric_computer.update(pred[:, 1], tf.y)
                            prob_class_1 = torch.sigmoid(pred[:, 1])
                            probabilities.extend(prob_class_1.cpu().numpy())
                        elif is_classification:
                            pred_class = pred.argmax(dim=-1)
                            metric_computer.update(pred_class, tf.y)
                        else:
                            metric_computer.update(pred.view(-1), tf.y.view(-1))

                    if is_classification:
                        return metric_computer.compute().item()
                    else:
                        return metric_computer.compute().item()**0.5


                if is_classification:
                    metric = 'Acc' if not is_binary_class else 'AUC'
                    best_val_metric = 0
                    best_test_metric = 0
                else:
                    metric = 'RMSE'
                    best_val_metric = float('inf')
                    best_test_metric = float('inf')
                # now_train_loss=[]
                # now_train_metric=[]
                # now_val_metric=[]

                
                epochs=range(1,args.epochs+1)
                for epoch in range(1, args.epochs + 1):
                    train_loss = train(epoch)
                    train_metric= test(train_loader)
                    val_metric= test(val_loader)
                    train_losses.append(train_loss)
                    train_metrics.append(train_metric)
                    val_metrics.append(val_metric)


                    if is_classification and val_metric > best_val_metric:
                        best_val_metric = val_metric
                        # best_test_metric = test_metric
                    elif not is_classification and val_metric < best_val_metric:
                        best_val_metric = val_metric
                        # best_test_metric = test_metric

                    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
                        f'Val {metric}: {val_metric:.4f}')
                    lr_scheduler.step()

                print(f'Best Val {metric}: {best_val_metric:.4f}')
                print("Evaluating on the test set...")
                test_metric= test(test_loader)
                print(f'Final Test {metric}: {test_metric:.4f}')

                # 繪圖並保存
                plt.figure()
                plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
                plt.plot(range(1, args.epochs + 1), train_metrics, label="Train Metric")
                plt.plot(range(1, args.epochs + 1), val_metrics, label="Val Metric")
                plt.xlabel("Epochs")
                plt.ylabel("Metric")
                plt.legend()
                plt.title(f"Training Progress - {dataset_name}")
                plt.savefig(os.path.join(result_dir, "training_progress.png"))
                plt.close()
                # 保存結果到 CSV
                save_metrics_to_csv(csv_result_path, {
                    'dataset_name': dataset_name,
                    'dataset_size': dataset_size,
                    'task_type': task_type,
                    'best_val_metric': best_val_metric,
                    'test_result': test_metric
                }, headers)

                print(f"Best Val Metric: {best_val_metric:.4f}, Final Test Metric: {test_metric:.4f}")
            