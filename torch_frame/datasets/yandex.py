from __future__ import annotations

import os.path as osp
import os
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import json
import torch_frame
from torch_frame.utils.split import SPLIT_TO_NUM
# SPLIT_TO_NUM = {'train': 0, 'val': 1, 'test': 2}

SPLIT_COL = 'split_col'
TARGET_COL = 'target'




def col2stype(df):
    col_to_stype: dict[str, torch_frame.stype] = {}
    c_col_names=[]
    n_col_names=[]
    for col_name in df.columns:
        if col_name == TARGET_COL:
            continue
        if col_name.startswith('C_feature'):
            col_to_stype[col_name] = torch_frame.categorical
            c_col_names.append(col_name)
        elif col_name.startswith('N_feature'):
            col_to_stype[col_name] = torch_frame.numerical
            n_col_names.append(col_name)

    if n_col_names is not None:
        for n_col in n_col_names:
            df[n_col] = df[n_col].astype('float64')
    return df, col_to_stype



class Yandex(torch_frame.data.Dataset):
    def train_val_test_split(self, df: pd.DataFrame, train_val_test_split_ratio: list) -> pd.DataFrame:
        # 新增 split_col 欄位，預設值為 -1 表示尚未分配
        df = df.assign(split_col=-1)
        
        # 打亂資料順序
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 計算每個分割的樣本數量
        train_size = int(len(df_shuffled) * train_val_test_split_ratio[0])
        val_size = int(len(df_shuffled) * train_val_test_split_ratio[1])
        test_size = len(df_shuffled) - train_size - val_size  # 剩下的就是 test 的大小

        # 填入對應的標籤到 split_col
        df_shuffled.loc[:train_size - 1, 'split_col'] = 0  # Train 標記為 0
        df_shuffled.loc[train_size:train_size + val_size - 1, 'split_col'] = 1  # Validation 標記為 1
        df_shuffled.loc[train_size + val_size:, 'split_col'] = 2  # Test 標記為 2

        return df_shuffled

    def __init__(self, df, name: str,train_val_test_split_ratio:list, task_type:str,DS:bool) -> None:
        self.df=df
        self.dataset_name = name
        
        self.df, col_to_stype = col2stype(self.df)
        if not DS:
            self.df=self.train_val_test_split(self.df,train_val_test_split_ratio)
        if task_type=="regression":
            col_to_stype[TARGET_COL] = torch_frame.numerical
        else:
            col_to_stype[TARGET_COL] = torch_frame.categorical
        super().__init__(self.df, col_to_stype, target_col=TARGET_COL,
                         split_col=SPLIT_COL)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}')")