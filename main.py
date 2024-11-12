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

# 將 project_root 路徑插入到 sys.path 的最前面
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'torch_frame')))

if __name__=="__main__":
    dataset_sizes=['small','large']
    task_types=['binclass','multiclass','regression']
    dataset_sizes=['small']
    task_types=['binclass']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    for dataset_size in dataset_sizes:
        for task_type in task_types:
            for dataset_name in os.listdir(f'./datasets/{dataset_size}_datasets/{task_type}'):
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
                train_val_test_split_ratio=[0.64,0.16,0.2]
                csv_path=f'./datasets/{dataset_size}_datasets/{task_type}/{dataset_name}/{dataset_name}.csv'
                df=pd.read_csv(csv_path)
                dataset = Yandex(df=df, name=args.dataset,train_val_test_split_ratio=train_val_test_split_ratio,task_type=task_type)
                dataset.materialize()
                print(dataset.df)
                print(len(dataset))
                print(type(dataset))
                train_dataset, val_dataset, test_dataset = dataset.split()
                print(len(train_dataset))
                print(type(train_dataset))
                print(len(val_dataset))
                print(type(val_dataset))
                print(len(test_dataset))
                print(type(test_dataset))
                


                

                