import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def compare_models(model_result_df):
    # 初始化結果字典
    results = {"best_val_metric": {}, "test_result": {}}

    # 假設所有模型的資料格式相同，並以 dataset_name 對齊
    base_df = list(model_result_df.values())[0]
    base_df['dataset_type'] = base_df['dataset_size'] + "+" + base_df['task_type']

    # 提取所有 dataset_type
    dataset_types = base_df['dataset_type'].unique()
    # 依照 dataset_type 統計每個模型的勝出次數
    dataset_counts = {}  # 用來記錄每種類型的資料集總數
    for dataset_type in dataset_types:
        # 確保字典結構正確
        if dataset_type not in results["best_val_metric"]:
            results["best_val_metric"][dataset_type] = {}
        if dataset_type not in results["test_result"]:
            results["test_result"][dataset_type] = {}
        
        # 過濾出當前 dataset_type 的資料
        filtered_dfs = {
            model_name: df[df['dataset_size'] + "+" + df['task_type'] == dataset_type]
            for model_name, df in model_result_df.items()
        }

        # 統計該 dataset_type 的資料集總數
        dataset_names = base_df[base_df['dataset_type'] == dataset_type]['dataset_name'].unique()
        dataset_counts[dataset_type] = len(dataset_names)
        # 初始化每個模型的勝出次數計數
        metric_counts = {model_name: 0 for model_name in model_result_df.keys()}
        result_counts = {model_name: 0 for model_name in model_result_df.keys()}

        for dataset_name in dataset_names:
            # 提取該資料集在所有模型中的 metric 和 test_result
            best_val_metrics = {
                model_name: df[df['dataset_name'] == dataset_name]['best_val_metric'].max()
                for model_name, df in filtered_dfs.items()
                if not df[df['dataset_name'] == dataset_name].empty
            }
            test_results = {
                model_name: df[df['dataset_name'] == dataset_name]['test_result'].max()
                for model_name, df in filtered_dfs.items()
                if not df[df['dataset_name'] == dataset_name].empty
            }

            # 特別處理 regression 類型，metric 和 result 都選最小值
            if "regression" in dataset_type:
                if best_val_metrics:
                    best_metric_model = min(best_val_metrics, key=best_val_metrics.get)  # 最小 RMSE 模型
                    metric_counts[best_metric_model] += 1
                if test_results:
                    best_result_model = min(test_results, key=test_results.get)  # 最小 RMSE 模型
                    result_counts[best_result_model] += 1
            else:
                # 非 regression 類型，metric 和 result 都選最大值
                if best_val_metrics:
                    best_metric_model = max(best_val_metrics, key=best_val_metrics.get)  # 最大分數模型
                    metric_counts[best_metric_model] += 1
                if test_results:
                    best_result_model = max(test_results, key=test_results.get)  # 最大分數模型
                    result_counts[best_result_model] += 1

        # 將結果存入字典
        results["best_val_metric"][dataset_type] = metric_counts
        results["test_result"][dataset_type] = result_counts

    return results, dataset_counts
def save_comparison_results_to_txt(train_val_test_split_ratio_str, comparison_results, dataset_counts):
    """
    將比較結果保存到 .txt 檔案中。
    
    :param file_path: 保存結果的 .txt 檔案路徑
    :param train_val_test_split_ratio_str: 當前訓練、驗證、測試比例字串
    :param comparison_results: 比較結果字典，包含 `best_val_metric` 和 `test_result`
    :param dataset_counts: 每個資料集類型的總數
    """
    file_path = "./result/comparison_results.txt"
    with open(file_path, 'a') as f:  # 使用 'a' 模式以附加方式保存內容
        f.write(f"Comparison for train_val_test_split_ratio: {train_val_test_split_ratio_str}\n\n")
        
        for dataset_type, metric_scores in comparison_results["best_val_metric"].items():
            # 先寫入資料集總數
            total_datasets = dataset_counts[dataset_type]
            f.write(f"  Dataset Type: {dataset_type} ({total_datasets} datasets)\n")
            
            # 寫入 `best_val_metric` 的最佳表現次數
            f.write("    Best Val Metric:\n")
            for model_name, count in metric_scores.items():
                f.write(f"      {model_name}: {count} best performances\n")
            
            # 寫入 `test_result` 的最佳表現次數
            f.write("    Test Result:\n")
            for model_name, count in comparison_results["test_result"][dataset_type].items():
                f.write(f"      {model_name}: {count} best performances\n")
        
        f.write("\n")

def plot_comparison_bar_chart(comparison_results, dataset_counts, train_val_test_split_ratio_str):
    """
    根據不同的 train-val-test 分割比例繪製每個模型在各資料集類型上贏得資料集數量的長條圖。

    :param comparison_results: 包含每個資料集類型的 `best_val_metric` 和 `test_result`
    :param dataset_counts: 每個資料集類型的總數
    :param train_val_test_split_ratio_str: 當前訓練、驗證、測試比例字串
    """
    # 獲取所有資料集類型
    dataset_types = list(comparison_results["best_val_metric"].keys())
    
    # 計算每個模型的贏得次數
    model_names = list(comparison_results["best_val_metric"][dataset_types[0]].keys())
    model_win_counts = {model_name: [0] * len(dataset_types) for model_name in model_names}
    
    for i, dataset_type in enumerate(dataset_types):
        for model_name in model_names:
            model_win_counts[model_name][i] = comparison_results["test_result"][dataset_type].get(model_name, 0)

    # 設定圖表大小
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 設定每個模型的長條圖
    bar_width = 0.15
    index = np.arange(len(dataset_types))
    
    # 根據模型數量來調整位置
    for i, model_name in enumerate(model_names):
        ax.bar(index + i * bar_width, model_win_counts[model_name], bar_width, label=model_name)
    
    # 設定標籤
    ax.set_xlabel('Dataset Type', fontsize=12)
    ax.set_ylabel('Number of Wins', fontsize=12)
    ax.set_title(f'Model Performance Comparison by Dataset Type\n(split ratio: {train_val_test_split_ratio_str})', fontsize=14)
    ax.set_xticks(index + (len(model_names) - 1) * bar_width / 2)
    ax.set_xticklabels(dataset_types, rotation=45, ha='right', fontsize=10)
    ax.legend()

    # 顯示圖表
    plt.tight_layout()
    plt.show()
    plt.savefig(f'./result/comparison_results_{train_val_test_split_ratio_str}.png')

if __name__ == '__main__':
    # Read the data from the CSV file
    train_val_test_split_ratios=[[0.64,0.16,0.2],[0.05,0.15,0.8]]
    for train_val_test_split_ratio in train_val_test_split_ratios:
        train_val_test_split_ratio_str = "_".join(map(str, train_val_test_split_ratio))
        model_result_df={}
        for model_name in os.listdir('./result'):
            model_path = os.path.join('./result', model_name)
            if os.path.isdir(model_path):
                df=pd.read_csv(f'./result/{model_name}/{train_val_test_split_ratio_str}.csv')
                model_result_df[model_name]=df


        # 比較模型表現
        comparison_results, dataset_counts = compare_models(model_result_df)
        
        # 保存結果到 .txt 檔案
        save_comparison_results_to_txt(train_val_test_split_ratio_str, comparison_results, dataset_counts)
        # 繪製比較長條圖
        plot_comparison_bar_chart(comparison_results, dataset_counts, train_val_test_split_ratio_str)
