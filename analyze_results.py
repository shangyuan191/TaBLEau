#!/usr/bin/env python3
"""
分析實驗結果並按照資料集分類進行排名統計
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
    
    # 資料集 -> 分類的映射
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
    """解析結果檔案"""
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析每個 dataset 區塊
    dataset_blocks = re.split(r'(?=^dataset: )', content, flags=re.MULTILINE)
    
    for block in dataset_blocks:
        if not block.strip():
            continue
        
        # 提取 dataset 名稱
        dataset_match = re.search(r'^dataset: (.+)$', block, re.MULTILINE)
        if not dataset_match:
            continue
        dataset_name = dataset_match.group(1).strip()
        
        # 提取模型名稱
        model_match = re.search(r'模型: (.+)$', block, re.MULTILINE)
        if not model_match:
            continue
        model_name = model_match.group(1).strip()
        
        # 提取 GNN 階段
        stage_match = re.search(r'GNN階段: (.+)$', block, re.MULTILINE)
        gnn_stage = stage_match.group(1).strip() if stage_match else 'none'
        
        # 提取指標
        val_metric_match = re.search(r'Best val metric: ([\d.]+)', block)
        test_metric_match = re.search(r'Best test metric: ([\d.]+)', block)
        
        if val_metric_match and test_metric_match:
            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'gnn_stage': gnn_stage,
                'val_metric': float(val_metric_match.group(1)),
                'test_metric': float(test_metric_match.group(1))
            })
    
    return results


def load_all_results(summary_dir):
    """載入所有實驗結果"""
    summary_path = Path(summary_dir)
    all_results = defaultdict(lambda: defaultdict(list))
    
    for file in summary_path.glob('*.txt'):
        filename = file.name
        
        # 解析檔案名稱
        # dataset_size_all_task_type_all_feature_type_all_models_{model}_gnn_stages_{stages}_{ratio}.txt
        match = re.search(r'models_([^_]+)_gnn_stages_([^_]+)(?:_epochs_\d+)?_(.+)\.txt', filename)
        if not match:
            continue
        
        model_name = match.group(1)
        gnn_stages = match.group(2)
        ratio_part = match.group(3)
        
        # 解析 ratio
        ratio_match = re.findall(r'([\d.]+)', ratio_part)
        if len(ratio_match) >= 3:
            ratio = f"{ratio_match[0]}_{ratio_match[1]}_{ratio_match[2]}"
        else:
            continue
        
        print(f"處理檔案: {filename}")
        results = parse_result_file(file)
        
        for result in results:
            # 組合模型變體名稱
            if result['gnn_stage'] == 'none':
                variant_name = model_name
            else:
                variant_name = f"{model_name}+GNN@{result['gnn_stage']}"
            
            result['variant'] = variant_name
            all_results[ratio][result['dataset']].append(result)
    
    return all_results


def rank_models(results_by_dataset, task_type):
    """
    對模型進行排名
    
    Args:
        results_by_dataset: {dataset: [results]}
        task_type: 'binclass', 'multiclass', 'regression'
    
    Returns:
        {dataset: [(variant, test_metric, rank), ...]}
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
        
        # 分配排名
        ranked_results = []
        for rank, result in enumerate(sorted_results, 1):
            ranked_results.append({
                'variant': result['variant'],
                'test_metric': result['test_metric'],
                'rank': rank
            })
        
        ranked[dataset] = ranked_results
    
    return ranked


def calculate_average_ranks(ranked_by_classification):
    """計算每個模型變體在每個分類下的平均排名"""
    avg_ranks = {}
    
    for classification, datasets_ranked in ranked_by_classification.items():
        variant_ranks = defaultdict(list)
        
        for dataset, ranked_results in datasets_ranked.items():
            for result in ranked_results:
                variant_ranks[result['variant']].append(result['rank'])
        
        # 計算平均排名
        avg_ranks[classification] = {}
        for variant, ranks in variant_ranks.items():
            avg_ranks[classification][variant] = {
                'avg_rank': sum(ranks) / len(ranks),
                'count': len(ranks),
                'ranks': ranks
            }
    
    return avg_ranks


def analyze_by_ratio(ratio, all_results, dataset_to_class):
    """分析特定 ratio 的結果"""
    print(f"\n{'='*80}")
    print(f"分析 Train/Val/Test Split Ratio: {ratio.replace('_', '/')}")
    print(f"{'='*80}\n")
    
    # 按照分類組織資料
    results_by_classification = defaultdict(lambda: defaultdict(list))
    
    for dataset, results in all_results[ratio].items():
        if dataset not in dataset_to_class:
            print(f"警告: 資料集 {dataset} 未找到分類")
            continue
        
        classification = dataset_to_class[dataset]['classification']
        results_by_classification[classification][dataset] = results
    
    # 對每個分類進行排名
    ranked_by_classification = {}
    
    for classification, datasets_results in results_by_classification.items():
        task_type = classification.split('+')[1]
        ranked = rank_models(datasets_results, task_type)
        ranked_by_classification[classification] = ranked
    
    # 計算平均排名
    avg_ranks = calculate_average_ranks(ranked_by_classification)
    
    # 輸出結果
    output_results(ratio, ranked_by_classification, avg_ranks)
    
    return ranked_by_classification, avg_ranks


def output_results(ratio, ranked_by_classification, avg_ranks):
    """輸出分析結果"""
    output_dir = Path(__file__).parent / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    ratio_str = ratio.replace('_', '-')
    
    # 1. 輸出詳細排名
    detail_file = output_dir / f'detailed_rankings_{ratio_str}.txt'
    with open(detail_file, 'w', encoding='utf-8') as f:
        f.write(f"實驗結果詳細排名 - Split Ratio: {ratio.replace('_', '/')}\n")
        f.write("=" * 100 + "\n\n")
        
        for classification in sorted(ranked_by_classification.keys()):
            datasets_ranked = ranked_by_classification[classification]
            f.write(f"\n{'='*100}\n")
            f.write(f"分類: {classification}\n")
            f.write(f"資料集數量: {len(datasets_ranked)}\n")
            f.write(f"{'='*100}\n\n")
            
            for dataset in sorted(datasets_ranked.keys()):
                ranked_results = datasets_ranked[dataset]
                f.write(f"\n資料集: {dataset}\n")
                f.write(f"{'-'*100}\n")
                f.write(f"{'排名':<6} {'模型變體':<40} {'Test Metric':<15}\n")
                f.write(f"{'-'*100}\n")
                
                for result in ranked_results:
                    f.write(f"{result['rank']:<6} {result['variant']:<40} {result['test_metric']:<15.6f}\n")
                f.write("\n")
    
    print(f"✅ 詳細排名已保存至: {detail_file}")
    
    # 2. 輸出平均排名統計
    summary_file = output_dir / f'average_rankings_{ratio_str}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"模型平均排名統計 - Split Ratio: {ratio.replace('_', '/')}\n")
        f.write("=" * 100 + "\n\n")
        
        for classification in sorted(avg_ranks.keys()):
            variant_stats = avg_ranks[classification]
            f.write(f"\n{'='*100}\n")
            f.write(f"分類: {classification}\n")
            f.write(f"{'='*100}\n\n")
            f.write(f"{'排名':<6} {'模型變體':<40} {'平均排名':<12} {'出現次數':<10}\n")
            f.write(f"{'-'*100}\n")
            
            # 按平均排名排序
            sorted_variants = sorted(variant_stats.items(), key=lambda x: x[1]['avg_rank'])
            
            for overall_rank, (variant, stats) in enumerate(sorted_variants, 1):
                f.write(f"{overall_rank:<6} {variant:<40} {stats['avg_rank']:<12.2f} {stats['count']:<10}\n")
            f.write("\n")
    
    print(f"✅ 平均排名已保存至: {summary_file}")
    
    # 3. 輸出 JSON 格式供進一步分析
    json_file = output_dir / f'rankings_{ratio_str}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'ratio': ratio,
            'ranked_by_classification': {
                k: {dk: [dict(r) for r in dv] for dk, dv in v.items()}
                for k, v in ranked_by_classification.items()
            },
            'average_ranks': {
                k: {vk: dict(vv) for vk, vv in v.items()}
                for k, v in avg_ranks.items()
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 數據已保存至: {json_file}")


def generate_summary_report(all_avg_ranks):
    """生成總體摘要報告"""
    output_dir = Path(__file__).parent / 'analysis_results'
    summary_file = output_dir / 'overall_summary.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("實驗結果總體摘要報告\n")
        f.write("=" * 100 + "\n\n")
        
        for ratio, avg_ranks in all_avg_ranks.items():
            f.write(f"\n{'='*100}\n")
            f.write(f"Split Ratio: {ratio.replace('_', '/')}\n")
            f.write(f"{'='*100}\n\n")
            
            # 收集所有模型變體的總體平均排名
            overall_avg = defaultdict(lambda: {'ranks': [], 'counts': []})
            
            for classification, variant_stats in avg_ranks.items():
                for variant, stats in variant_stats.items():
                    overall_avg[variant]['ranks'].append(stats['avg_rank'])
                    overall_avg[variant]['counts'].append(stats['count'])
            
            # 計算總體平均
            variant_overall = {}
            for variant, data in overall_avg.items():
                # 加權平均（按出現次數）
                total_count = sum(data['counts'])
                weighted_avg = sum(r * c for r, c in zip(data['ranks'], data['counts'])) / total_count
                variant_overall[variant] = {
                    'weighted_avg_rank': weighted_avg,
                    'total_count': total_count,
                    'classifications': len(data['ranks'])
                }
            
            # 排序並輸出
            sorted_variants = sorted(variant_overall.items(), key=lambda x: x[1]['weighted_avg_rank'])
            
            f.write(f"{'排名':<6} {'模型變體':<40} {'加權平均排名':<15} {'總數據集數':<12} {'分類數':<10}\n")
            f.write(f"{'-'*100}\n")
            
            for rank, (variant, stats) in enumerate(sorted_variants, 1):
                f.write(f"{rank:<6} {variant:<40} {stats['weighted_avg_rank']:<15.2f} "
                       f"{stats['total_count']:<12} {stats['classifications']:<10}\n")
            f.write("\n")
    
    print(f"\n✅ 總體摘要報告已保存至: {summary_file}")


def main():
    """主函數"""
    print("開始分析實驗結果...")
    print("=" * 80)
    
    # 獲取資料集分類
    print("\n1. 載入資料集分類...")
    dataset_to_class = get_dataset_classification()
    print(f"   找到 {len(dataset_to_class)} 個資料集")
    
    # 載入所有結果
    print("\n2. 載入實驗結果檔案...")
    summary_dir = Path(__file__).parent / 'summary_results'
    all_results = load_all_results(summary_dir)
    print(f"   找到 {len(all_results)} 種 split ratio 的實驗結果")
    
    # 分析每個 ratio
    print("\n3. 開始分析各個 split ratio...")
    all_avg_ranks = {}
    
    for ratio in sorted(all_results.keys()):
        ranked, avg_ranks = analyze_by_ratio(ratio, all_results, dataset_to_class)
        all_avg_ranks[ratio] = avg_ranks
    
    # 生成總體報告
    print("\n4. 生成總體摘要報告...")
    generate_summary_report(all_avg_ranks)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
