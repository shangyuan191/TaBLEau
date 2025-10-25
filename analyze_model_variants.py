#!/usr/bin/env python3
"""
分析每個模型的6種變體（base + 5種GNN插入階段）的表現
按照 split ratio 和資料集分類進行排名統計
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import json


def get_dataset_classification():
    """獲取資料集分類"""
    base_path = Path(__file__).parent / 'datasets'
    
    dataset_sizes = ['small_datasets', 'large_datasets']
    task_types = ['binclass', 'multiclass', 'regression']
    feature_types = ['numerical', 'categorical', 'balanced']
    
    dataset_to_class = {}
    
    for size in dataset_sizes:
        for task in task_types:
            for feature in feature_types:
                path = base_path / size / task / feature
                if path.exists() and path.is_dir():
                    datasets = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    classification = f"{size}+{task}+{feature}"
                    for dataset in datasets:
                        dataset_to_class[dataset] = {
                            'classification': classification,
                            'size': size,
                            'task': task,
                            'feature': feature
                        }
    
    return dataset_to_class


def parse_result_file(file_path):
    """解析結果檔案，提取所有模型變體的結果"""
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 先按 dataset 切分
    dataset_blocks = re.split(r'(?=^dataset: )', content, flags=re.MULTILINE)
    
    for dataset_block in dataset_blocks:
        if not dataset_block.strip():
            continue
        
        # 提取 dataset 名稱
        dataset_match = re.search(r'^dataset: (.+)$', dataset_block, re.MULTILINE)
        if not dataset_match:
            continue
        dataset_name = dataset_match.group(1).strip()
        
        # 提取模型名稱
        model_match = re.search(r'^\s*模型: (.+)$', dataset_block, re.MULTILINE)
        if not model_match:
            continue
        model_name = model_match.group(1).strip()
        
        # 按 GNN階段 切分（每個階段一個結果）
        gnn_blocks = re.split(r'(?=^\s*GNN階段: )', dataset_block, flags=re.MULTILINE)
        
        for gnn_block in gnn_blocks:
            if 'GNN階段:' not in gnn_block:
                continue
            
            # 提取 GNN 階段
            stage_match = re.search(r'GNN階段: (.+)$', gnn_block, re.MULTILINE)
            if not stage_match:
                continue
            gnn_stage = stage_match.group(1).strip()
            
            # 提取指標
            val_metric_match = re.search(r'Best val metric: ([\d.]+)', gnn_block)
            test_metric_match = re.search(r'Best test metric: ([\d.]+)', gnn_block)
            
            if val_metric_match and test_metric_match:
                # 組合變體名稱
                if gnn_stage == 'none':
                    variant_name = f"{model_name} (base)"
                else:
                    variant_name = f"{model_name} (GNN@{gnn_stage})"
                
                results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'gnn_stage': gnn_stage,
                    'variant': variant_name,
                    'val_metric': float(val_metric_match.group(1)),
                    'test_metric': float(test_metric_match.group(1))
                })
    
    return results


def load_all_results(summary_dir):
    """載入所有實驗結果"""
    summary_path = Path(summary_dir)
    # ratio -> dataset -> [results]
    all_results = defaultdict(lambda: defaultdict(list))
    
    for file in summary_path.glob('*.txt'):
        filename = file.name
        
        # 解析檔案名稱
        match = re.search(r'models_([^_]+)_gnn_stages_([^_]+)(?:_epochs_\d+)?_(.+)\.txt', filename)
        if not match:
            continue
        
        model_name = match.group(1)
        gnn_stages = match.group(2)
        ratio_part = match.group(3)
        
        # 解析 ratio
        ratio_match = re.findall(r'([\d.]+)', ratio_part)
        if len(ratio_match) >= 3:
            ratio = f"{ratio_match[0]}/{ratio_match[1]}/{ratio_match[2]}"
        else:
            continue
        
        print(f"處理檔案: {filename}")
        results = parse_result_file(file)
        print(f"  找到 {len(results)} 個變體結果")
        
        for result in results:
            all_results[ratio][result['dataset']].append(result)
    
    return all_results


def rank_variants_by_dataset(results_by_dataset, task_type):
    """
    對每個資料集的所有變體進行排名
    
    Args:
        results_by_dataset: {dataset: [results]}
        task_type: 'binclass', 'multiclass', 'regression'
    
    Returns:
        {dataset: [(variant, model, gnn_stage, test_metric, rank), ...]}
    """
    ranked = {}
    
    for dataset, results in results_by_dataset.items():
        if not results:
            continue
        
        # 按照 test metric 排序
        if task_type == 'regression':
            # RMSE 越低越好
            sorted_results = sorted(results, key=lambda x: x['test_metric'])
        else:
            # AUC/ACC 越高越好
            sorted_results = sorted(results, key=lambda x: x['test_metric'], reverse=True)
        
        # 賦予排名
        ranked_results = []
        for rank, result in enumerate(sorted_results, 1):
            ranked_results.append({
                'variant': result['variant'],
                'model': result['model'],
                'gnn_stage': result['gnn_stage'],
                'test_metric': result['test_metric'],
                'rank': rank
            })
        
        ranked[dataset] = ranked_results
    
    return ranked


def calculate_model_variant_average_ranks(ranked_by_classification, dataset_classifications):
    """
    計算每個模型的每個變體在每種分類下的平均排名
    
    Returns:
        {
            classification: {
                model: {
                    variant: {
                        'avg_rank': float,
                        'count': int,
                        'datasets': [dataset_names]
                    }
                }
            }
        }
    """
    # classification -> model -> variant -> ranks list
    variant_ranks = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for classification, datasets in ranked_by_classification.items():
        for dataset, rankings in datasets.items():
            for item in rankings:
                model = item['model']
                variant = item['variant']
                rank = item['rank']
                
                variant_ranks[classification][model][variant].append({
                    'dataset': dataset,
                    'rank': rank,
                    'test_metric': item['test_metric']
                })
    
    # 計算平均排名
    avg_ranks = {}
    for classification, models in variant_ranks.items():
        avg_ranks[classification] = {}
        for model, variants in models.items():
            avg_ranks[classification][model] = {}
            for variant, ranks in variants.items():
                avg_rank = sum(r['rank'] for r in ranks) / len(ranks)
                avg_ranks[classification][model][variant] = {
                    'avg_rank': avg_rank,
                    'count': len(ranks),
                    'datasets': [r['dataset'] for r in ranks],
                    'ranks': [r['rank'] for r in ranks]
                }
    
    return avg_ranks


def output_variant_analysis(ratio, avg_ranks_by_class, output_dir):
    """輸出每個模型變體的分析結果"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 安全的檔案名稱
    safe_ratio = ratio.replace('/', '-')
    
    # 1. 詳細報告：每個分類下，每個模型的6種變體的排名
    detail_file = output_path / f'model_variants_by_classification_{safe_ratio}.txt'
    with open(detail_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"模型變體分析 - Split Ratio: {ratio}\n")
        f.write("="*100 + "\n\n")
        
        for classification in sorted(avg_ranks_by_class.keys()):
            f.write("\n" + "="*100 + "\n")
            f.write(f"資料集分類: {classification}\n")
            f.write("="*100 + "\n\n")
            
            models = avg_ranks_by_class[classification]
            
            for model in sorted(models.keys()):
                variants = models[model]
                
                f.write(f"\n{'='*50}\n")
                f.write(f"模型: {model}\n")
                f.write(f"{'='*50}\n\n")
                
                # 按平均排名排序變體
                sorted_variants = sorted(variants.items(), key=lambda x: x[1]['avg_rank'])
                
                f.write(f"{'變體名稱':<40} {'平均排名':>10} {'資料集數':>10}\n")
                f.write("-" * 65 + "\n")
                
                for variant, stats in sorted_variants:
                    f.write(f"{variant:<40} {stats['avg_rank']:>10.2f} {stats['count']:>10}\n")
                
                f.write("\n")
    
    # 2. 匯總報告：每個模型在所有分類的表現
    summary_file = output_path / f'model_variants_summary_{safe_ratio}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"模型變體匯總分析 - Split Ratio: {ratio}\n")
        f.write("="*100 + "\n\n")
        
        # 收集所有模型
        all_models = set()
        for classification in avg_ranks_by_class.values():
            all_models.update(classification.keys())
        
        for model in sorted(all_models):
            f.write("\n" + "="*100 + "\n")
            f.write(f"模型: {model}\n")
            f.write("="*100 + "\n\n")
            
            # 收集該模型在所有分類下的變體平均排名
            # variant -> classification -> avg_rank
            variant_stats = defaultdict(lambda: defaultdict(float))
            variant_counts = defaultdict(lambda: defaultdict(int))
            
            for classification, models in avg_ranks_by_class.items():
                if model in models:
                    for variant, stats in models[model].items():
                        variant_stats[variant][classification] = stats['avg_rank']
                        variant_counts[variant][classification] = stats['count']
            
            # 計算跨所有分類的加權平均
            variant_overall = {}
            for variant in variant_stats.keys():
                total_rank = 0
                total_count = 0
                for classification in variant_stats[variant].keys():
                    total_rank += variant_stats[variant][classification] * variant_counts[variant][classification]
                    total_count += variant_counts[variant][classification]
                
                if total_count > 0:
                    variant_overall[variant] = {
                        'overall_avg_rank': total_rank / total_count,
                        'total_datasets': total_count,
                        'by_classification': dict(variant_stats[variant])
                    }
            
            # 按整體平均排名排序
            sorted_variants = sorted(variant_overall.items(), key=lambda x: x[1]['overall_avg_rank'])
            
            f.write(f"{'排名':<5} {'變體名稱':<40} {'整體平均排名':>12} {'資料集總數':>12}\n")
            f.write("-" * 75 + "\n")
            
            for rank, (variant, stats) in enumerate(sorted_variants, 1):
                f.write(f"{rank:<5} {variant:<40} {stats['overall_avg_rank']:>12.2f} {stats['total_datasets']:>12}\n")
            
            f.write("\n各分類詳細排名:\n")
            line = f"{'變體名稱':<40}"
            classifications = sorted(set(c for v in variant_stats.values() for c in v.keys()))
            for c in classifications:
                line += f" {c.split('+')[-1]:>12}"
            f.write(line + "\n")
            f.write("-" * (40 + 13 * len(classifications)) + "\n")
            
            for variant, stats in sorted_variants:
                line = f"{variant:<40}"
                for c in classifications:
                    if c in stats['by_classification']:
                        line += f" {stats['by_classification'][c]:>12.2f}"
                    else:
                        line += f" {'-':>12}"
                f.write(line + "\n")
            
            f.write("\n")
    
    # 3. JSON 輸出
    json_file = output_path / f'model_variants_{safe_ratio}.json'
    json_data = {
        'ratio': ratio,
        'by_classification': avg_ranks_by_class
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n已生成變體分析報告:")
    print(f"  - 詳細報告: {detail_file}")
    print(f"  - 匯總報告: {summary_file}")
    print(f"  - JSON 資料: {json_file}")


def main():
    # 載入資料集分類
    print("載入資料集分類...")
    dataset_classifications = get_dataset_classification()
    print(f"找到 {len(dataset_classifications)} 個資料集")
    
    # 載入所有實驗結果
    summary_dir = Path(__file__).parent / 'summary_results'
    print(f"\n從 {summary_dir} 載入實驗結果...")
    all_results = load_all_results(summary_dir)
    print(f"找到 {len(all_results)} 種 split ratio 的實驗結果")
    
    # 對每個 ratio 進行分析
    for ratio in sorted(all_results.keys()):
        print(f"\n{'='*80}")
        print(f"分析 Split Ratio: {ratio}")
        print(f"{'='*80}")
        
        results_by_dataset = all_results[ratio]
        
        # 按分類組織資料
        # classification -> dataset -> results
        by_classification = defaultdict(dict)
        
        for dataset, results in results_by_dataset.items():
            if dataset not in dataset_classifications:
                print(f"  警告: 資料集 {dataset} 找不到分類資訊")
                continue
            
            classification = dataset_classifications[dataset]['classification']
            task_type = dataset_classifications[dataset]['task']
            
            # 對該資料集的所有變體進行排名
            ranked = rank_variants_by_dataset({dataset: results}, task_type)
            by_classification[classification][dataset] = ranked[dataset]
        
        print(f"  共處理 {len(by_classification)} 個分類")
        
        # 計算每個模型變體的平均排名
        print("  計算各模型變體的平均排名...")
        avg_ranks_by_class = calculate_model_variant_average_ranks(by_classification, dataset_classifications)
        
        # 輸出結果
        output_dir = Path(__file__).parent / 'variant_analysis'
        output_variant_analysis(ratio, avg_ranks_by_class, output_dir)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
