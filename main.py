import os
import csv
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
                df=pd.read_csv(f'./datasets/{dataset_size}_datasets/{task_type}/{dataset_name}/{dataset_name}.csv')

                

                