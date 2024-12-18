import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
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
from sklearn.metrics import root_mean_squared_error
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
# ======== 1. 數據預處理與圖構建 ========
def preprocess_and_build_graph(df, task_type, train_val_test_split_ratio, knn_k=10):
    """
    根據表格數據構建圖結構，並生成 Train/Val/Test Mask。
    """
    # 識別特徵和目標欄位
    numerical_cols = [col for col in df.columns if col.startswith('N_feature')]
    categorical_cols = [col for col in df.columns if col.startswith('C_feature')]
    target_col = 'target'

    # 數值特徵處理
    num_data = df[numerical_cols].fillna(0).values
    scaler = StandardScaler()
    num_data = scaler.fit_transform(num_data)

    # 類別特徵處理
    cat_data = []
    for col in categorical_cols:
        encoder = LabelEncoder()
        encoded_col = encoder.fit_transform(df[col].fillna('missing'))
        cat_data.append(encoded_col)
    cat_data = np.array(cat_data).T if categorical_cols else None

    # 節點特徵拼接
    if cat_data is not None:
        node_features = np.hstack([num_data, cat_data])
    else:
        node_features = num_data

    # KNN 構建邊
    knn_graph = kneighbors_graph(node_features, knn_k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)

    # 目標處理
    target = df[target_col].values
    if task_type == 'binclass' or task_type == 'multiclass':
        target = LabelEncoder().fit_transform(target)  # 分類任務需要 Label 編碼

    # Train/Val/Test Mask
    num_nodes = df.shape[0]
    train_size, val_size, test_size = train_val_test_split_ratio
    train_mask, temp_mask = train_test_split(np.arange(num_nodes), train_size=train_size, random_state=42)
    val_mask, test_mask = train_test_split(temp_mask, test_size=test_size/(val_size + test_size), random_state=42)

    # PyG Graph Data
    graph_data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(target, dtype=torch.float),
        train_mask=torch.tensor(np.isin(np.arange(num_nodes), train_mask), dtype=torch.bool),
        val_mask=torch.tensor(np.isin(np.arange(num_nodes), val_mask), dtype=torch.bool),
        test_mask=torch.tensor(np.isin(np.arange(num_nodes), test_mask), dtype=torch.bool)
    )
    graph_data.task_type = task_type
    return graph_data


# ======== 2. 定義 GCN 模型 ========
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task_type):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.task_type = task_type

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        if self.task_type == 'binclass' or self.task_type == 'multiclass':
            return F.log_softmax(x, dim=1)
        else:
            return x


# ======== 3. 訓練與評估函數 ========
def train_and_evaluate(graph_data, epochs, output_dir, dataset_name, task_type):
    input_dim = graph_data.x.shape[1]
    output_dim = len(torch.unique(graph_data.y)) if task_type in ['binclass', 'multiclass'] else 1
    model = GCN(input_dim, hidden_dim=16, output_dim=output_dim, task_type=task_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 紀錄每個 epoch 的 loss 和 metric
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    best_val_metric, best_epoch = -1, 0

    for epoch in range(epochs):
        # 訓練
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)

        if task_type in ['binclass', 'multiclass']:
            loss = F.nll_loss(out[graph_data.train_mask], graph_data.y[graph_data.train_mask].long())
        else:
            loss = F.mse_loss(out[graph_data.train_mask].squeeze(), graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()

        # 計算訓練和驗證的 metric
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index)
            preds = out.argmax(dim=1) if task_type in ['binclass', 'multiclass'] else out.squeeze()
            y_true = graph_data.y.cpu().numpy()

            # 計算 metric
            if task_type == 'binclass':
                # 檢查是否為單一類別
                y_true = graph_data.y.cpu().numpy()
                preds_train = preds[graph_data.train_mask]
                preds_val = preds[graph_data.val_mask]

                # 若是 logits，轉換為概率
                preds_train_prob = torch.sigmoid(preds_train).squeeze().cpu().numpy()
                preds_val_prob = torch.sigmoid(preds_val).squeeze().cpu().numpy()

                # 轉換標籤為 NumPy 格式
                y_true_train = y_true[graph_data.train_mask]
                y_true_val = y_true[graph_data.val_mask]

                # 檢查 train_mask 上類別數量
                if len(np.unique(y_true_train)) < 2:
                    print("Warning: Only one class present in train set. Using accuracy instead of AUC.")
                    # 二分類只有一類時，計算 accuracy
                    train_preds = (preds_train_prob > 0.5).astype(int)
                    val_preds = (preds_val_prob > 0.5).astype(int)
                    
                    train_metric = accuracy_score(y_true_train, train_preds)
                    val_metric = accuracy_score(y_true_val, val_preds)
                else:
                    # 計算 AUC（基於概率值）
                    train_metric = roc_auc_score(y_true_train, preds_train_prob)
                    val_metric = roc_auc_score(y_true_val, preds_val_prob)
            elif task_type == 'multiclass':
                train_metric = accuracy_score(y_true[graph_data.train_mask], preds[graph_data.train_mask].cpu())
                val_metric = accuracy_score(y_true[graph_data.val_mask], preds[graph_data.val_mask].cpu())
            else:
                train_metric = np.sqrt(mean_squared_error(y_true[graph_data.train_mask], preds[graph_data.train_mask].cpu()))
                val_metric = np.sqrt(mean_squared_error(y_true[graph_data.val_mask], preds[graph_data.val_mask].cpu()))

        # 記錄結果
        train_losses.append(loss.item())
        val_losses.append(loss.item())
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)

        # 儲存最佳 val metric
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch

    # 測試階段
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        preds = out.argmax(dim=1) if task_type in ['binclass', 'multiclass'] else out.squeeze()
        y_true = graph_data.y.cpu().numpy()

        if task_type == 'binclass':
            test_result = roc_auc_score(y_true[graph_data.test_mask], preds[graph_data.test_mask].cpu())
        elif task_type == 'multiclass':
            test_result = accuracy_score(y_true[graph_data.test_mask], preds[graph_data.test_mask].cpu())
        else:
            test_result = np.sqrt(mean_squared_error(y_true[graph_data.test_mask], preds[graph_data.test_mask].cpu()))

    # 繪製折線圖 - 包含 train_losses, train_metrics, val_metrics
    plt.figure(figsize=(10, 6))  # 設定圖表大小

    # 繪製 Train Losses
    plt.plot(range(epochs), train_losses, label='Train Loss', color='tab:blue', linestyle='--')

    # 繪製 Train Metrics
    plt.plot(range(epochs), train_metrics, label='Train Metric', color='tab:orange')

    # 繪製 Val Metrics
    plt.plot(range(epochs), val_metrics, label='Val Metric', color='tab:green')

    # 設定圖表細節
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title(f'{dataset_name} Training Curve')
    plt.legend()
    plt.grid(True)  # 加上網格以便觀察
    plt.tight_layout()

    # 儲存圖表
    plt.savefig(os.path.join(output_dir, 'training_curve.png'))
    plt.close()

    return best_val_metric, test_result
if __name__=="__main__":
    dataset_sizes=['small','large']
    task_types=['binclass','multiclass','regression']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # train_val_test_split_ratio = [0.64, 0.16, 0.2]
    train_val_test_split_ratio = [0.05, 0.15, 0.8]
    ratio_str = "_".join(map(str, train_val_test_split_ratio))
    result_dir = './result/GCN'
    summary_csv = os.path.join(result_dir, f'{ratio_str}.csv')
    # 準備 CSV 檔案
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(summary_csv):
        with open(summary_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['dataset_name', 'dataset_size', 'task_type', 'best_val_metric', 'test_result'])


    for dataset_size in dataset_sizes:
        for task_type in task_types:
            for dataset_name in os.listdir(f'./datasets/{dataset_size}_datasets/{task_type}'):
                print(f"\nDataset_name : {dataset_name}")
                csv_path=f'./datasets/{dataset_size}_datasets/{task_type}/{dataset_name}/{dataset_name}.csv'
                df=pd.read_csv(csv_path)
                print(len(df))

                # 預處理並構建圖
                graph_data = preprocess_and_build_graph(df, task_type, train_val_test_split_ratio, knn_k=5)
                # 建立輸出目錄
                dataset_output_dir = os.path.join(result_dir, f"{dataset_size}_datasets", task_type, dataset_name, ratio_str)
                os.makedirs(dataset_output_dir, exist_ok=True)

                # 訓練與評估
                best_val_metric, test_result = train_and_evaluate(
                    graph_data, epochs=200, output_dir=dataset_output_dir, dataset_name=dataset_name, task_type=task_type
                )

                # 寫入結果至 CSV
                with open(summary_csv, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([dataset_name, dataset_size, task_type, best_val_metric, test_result])
                print(f"Best Val Metric: {best_val_metric:.4f}, Test Result: {test_result:.4f}")