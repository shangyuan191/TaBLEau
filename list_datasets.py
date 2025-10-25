#!/usr/bin/env python3
"""
列出資料集分類統計
按照 dataset_size / task_type / feature_type 分類統計所有資料集
"""

import os
from pathlib import Path


def list_datasets_by_classification(base_path=None):
    """
    列出所有資料集按照分類統計
    
    Args:
        base_path: 資料集根目錄路徑，預設為當前目錄下的 datasets
    """
    if base_path is None:
        base_path = Path(__file__).parent / 'datasets'
    else:
        base_path = Path(base_path)
    
    # 定義分類
    dataset_sizes = ['small_datasets', 'large_datasets']
    task_types = ['binclass', 'multiclass', 'regression']
    feature_types = ['numerical', 'categorical', 'balanced']
    
    # 存儲結果
    results = {}
    
    # 遍歷所有組合
    for size in dataset_sizes:
        for task in task_types:
            for feature in feature_types:
                path = base_path / size / task / feature
                if path.exists() and path.is_dir():
                    # 獲取所有資料集（排除隱藏檔案和非目錄項目）
                    datasets = sorted([d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')])
                    if datasets:  # 只記錄有資料集的分類
                        key = f"{size}+{task}+{feature}"
                        results[key] = datasets
    
    return results


def print_statistics(results):
    """
    輸出統計結果
    
    Args:
        results: 資料集分類字典
    """
    # 按照特定順序輸出結果
    print("=" * 80)
    print("資料集分類統計")
    print("=" * 80)
    print()
    
    dataset_sizes = ['small_datasets', 'large_datasets']
    task_types = ['binclass', 'multiclass', 'regression']
    feature_types = ['numerical', 'categorical', 'balanced']
    
    # 先按 size, task, feature 排序輸出
    for size in dataset_sizes:
        for task in task_types:
            for feature in feature_types:
                key = f"{size}+{task}+{feature}"
                if key in results:
                    datasets = results[key]
                    print(f"{key} (總數有 {len(datasets)} 個):")
                    for dataset in datasets:
                        print(f"  {dataset}")
                    print()
    
    # 輸出總統計
    total_count = sum(len(datasets) for datasets in results.values())
    print("=" * 80)
    print(f"總計: {len(results)} 個分類，共 {total_count} 個資料集")
    print("=" * 80)


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='列出資料集分類統計',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用預設路徑（當前目錄下的 datasets）
  python list_datasets.py
  
  # 指定資料集路徑
  python list_datasets.py -p /path/to/datasets
  
  # 只顯示特定分類
  python list_datasets.py --filter small_datasets+binclass
        """
    )
    
    parser.add_argument(
        '-p', '--path',
        help='資料集根目錄路徑（預設: ./datasets）'
    )
    parser.add_argument(
        '--filter',
        help='過濾特定分類（例如: small_datasets+binclass）'
    )
    parser.add_argument(
        '--count-only',
        action='store_true',
        help='只顯示數量統計，不列出資料集名稱'
    )
    
    args = parser.parse_args()
    
    # 獲取資料集分類
    results = list_datasets_by_classification(args.path)
    
    # 如果有過濾條件
    if args.filter:
        results = {k: v for k, v in results.items() if args.filter in k}
        if not results:
            print(f"❌ 找不到符合條件的分類: {args.filter}")
            return
    
    # 如果只顯示數量
    if args.count_only:
        print("資料集數量統計:")
        for key, datasets in results.items():
            print(f"  {key}: {len(datasets)} 個")
        total = sum(len(datasets) for datasets in results.values())
        print(f"\n總計: {total} 個資料集")
    else:
        print_statistics(results)


if __name__ == '__main__':
    main()
