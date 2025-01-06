import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from collections import defaultdict

if __name__ == '__main__':
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # Load data
    dataset_sizes=['small','large']
    task_types=['binclass','multiclass','regression']
    for dataset_size in dataset_sizes:
        for task_type in task_types:
            for dataset_name in os.listdir(f'./datasets/{dataset_size}_datasets/{task_type}'):
                if dataset_name in ["categorical","numerical","balanced"]:
                    continue
                dataset_path=f'./datasets/{dataset_size}_datasets/{task_type}/{dataset_name}/{dataset_name}.csv'
                dataset=pd.read_csv(dataset_path)
                column_names = dataset.columns

                # 確認類別型與數值型特徵
                numerical_features = [col for col in column_names if col.startswith('N_feature')]
                categorical_features = [col for col in column_names if col.startswith('C_feature')]

                num_numerical = len(numerical_features)
                num_categorical = len(categorical_features)
                total_features = num_numerical + num_categorical

                numerical_ratio = num_numerical / total_features
                categorical_ratio = num_categorical / total_features

                if numerical_ratio>=2/3:
                    category='numerical'
                elif categorical_ratio>=2/3:
                    category='categorical'
                else:
                    category='balanced'
                print(f"num_numerical: {num_numerical}")
                print(f"num_categorical: {num_categorical}")
                print(f"total_features: {total_features}")
                print(f"numerical_ratio: {numerical_ratio*100}%")
                print(f"categorical_ratio: {categorical_ratio*100}%")
                print(f'{dataset_name} is {category} dataset')
                target_folder=f'./datasets/{dataset_size}_datasets/{task_type}/{category}/{dataset_name}'
                print(f"target_folder: {target_folder}")
                print("\n")
                

                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    print(f"  Created target folder: {target_folder}")

                # 複製檔案
                target_file = os.path.join(target_folder, f'{dataset_name}.csv')
                shutil.copy(dataset_path, target_file)
                stats[dataset_size][task_type][category] += 1
                print(f"  Copied {dataset_path} to {target_file}\n")
        # 輸出統計結果
    output_file = "dataset_statistics.txt"

    with open(output_file, "w") as f:
        f.write("\nDataset Statistics:\n")
        for dataset_size, task_stats in stats.items():
            total_datasets = sum(
                sum(category_counts.values()) for category_counts in task_stats.values()
            )
            f.write(f"  {dataset_size} datasets: {total_datasets} total\n")
            for task_type, category_counts in task_stats.items():
                task_total = sum(category_counts.values())
                f.write(f"    {task_type}: {task_total} datasets\n")
                for category, count in category_counts.items():
                    f.write(f"      {category}: {count} datasets\n")