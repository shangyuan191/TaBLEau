#!/usr/bin/env python3
"""
分析GNN增強效果：比較每個可插入GNN的模型的原始表現與GNN增強變體

比較模式：
1. 'single_model' (預設): 19個競爭者
   - 目標模型的7個配置（大訓練集baseline + 小訓練集6變體）
   - 6個參考模型的兩種ratio（12個配置）

2. 'all_models': 132個競爭者
   - 10個可插入GNN的模型 × 6個變體 × 2種ratio = 120個
   - 6個參考模型 × 2種ratio = 12個

使用方法：
    python analyze_gnn_enhancement_flexible.py [mode]
    
    mode: 'single_model' 或 'all_models' (預設: 'single_model')
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# 可以插入GNN的模型
GNN_INSERTABLE_MODELS = [
    'excelformer', 'fttransformer', 'resnet', 'tabnet', 
    'tabtransformer', 'trompt', 'vime', 'scarf', 'subtab', 'tabm'
]

# 參考模型（無法插入GNN）
REFERENCE_MODELS = ['t2g-former', 'tabpfn', 'xgboost', 'catboost', 'lightgbm', 'tabgnn']

# GNN插入階段
GNN_STAGES = ['none', 'start', 'materialize', 'encoding', 'columnwise', 'decoding']

def parse_result_file(file_path):
    """解析實驗結果檔案"""
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 split ratio
    ratio_match = re.search(r'(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)\.txt$', str(file_path))
    if not ratio_match:
        return results
    
    train_ratio = ratio_match.group(1)
    val_ratio = ratio_match.group(2)
    test_ratio = ratio_match.group(3)
    ratio_str = f"{train_ratio}/{val_ratio}/{test_ratio}"
    
    # 按資料集分割
    dataset_blocks = re.split(r'dataset:\s*(\S+)', content)[1:]
    
    for i in range(0, len(dataset_blocks), 2):
        if i + 1 >= len(dataset_blocks):
            break
            
        dataset_name = dataset_blocks[i].strip()
        block_content = dataset_blocks[i + 1]
        
        # 提取任務類型（從第一個模型結果中）
        task_match = re.search(r'任務類型:\s*(\w+)', block_content)
        task_type = task_match.group(1) if task_match else None
        
        # 提取每個模型的結果
        model_blocks = re.split(r'模型:\s*(\S+)', block_content)[1:]
        
        for j in range(0, len(model_blocks), 2):
            if j + 1 >= len(model_blocks):
                break
                
            model_name = model_blocks[j].strip()
            model_content = model_blocks[j + 1]
            
            # 提取GNN階段結果
            stage_blocks = re.split(r'GNN階段:\s*(\S+)', model_content)[1:]
            
            for k in range(0, len(stage_blocks), 2):
                if k + 1 >= len(stage_blocks):
                    break
                    
                gnn_stage = stage_blocks[k].strip()
                stage_content = stage_blocks[k + 1]
                
                # 提取指標
                test_metric_match = re.search(r'Best test metric:\s*([\d.]+)', stage_content)
                
                if test_metric_match:
                    test_metric = float(test_metric_match.group(1))
                    
                    results.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'gnn_stage': gnn_stage,
                        'ratio': ratio_str,
                        'task_type': task_type,
                        'test_metric': test_metric
                    })
    
    return results

def load_dataset_classifications(datasets_folder):
    """載入資料集分類"""
    classifications = {}
    datasets_path = Path(datasets_folder)
    
    for size_dir in datasets_path.iterdir():
        if not size_dir.is_dir():
            continue
        size = size_dir.name
        
        for task_dir in size_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            
            for feature_dir in task_dir.iterdir():
                if not feature_dir.is_dir():
                    continue
                feature = feature_dir.name
                
                for dataset_dir in feature_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    dataset_name = dataset_dir.name
                    classification = f"{size}+{task}+{feature}"
                    classifications[dataset_name] = classification
    
    return classifications

def rank_competitors(results_list, task_type):
    """
    對競爭者進行排名
    results_list: 包含所有競爭者結果的列表
    task_type: 任務類型（binclass, multiclass, regression）
    """
    if not results_list:
        return {}
    
    # 根據任務類型決定排序方式
    if task_type == 'regression':
        # RMSE: 越小越好
        sorted_results = sorted(results_list, key=lambda x: x['test_metric'])
    else:
        # AUC/ACC: 越大越好
        sorted_results = sorted(results_list, key=lambda x: x['test_metric'], reverse=True)
    
    # 分配排名
    rankings = {}
    for rank, result in enumerate(sorted_results, 1):
        key = f"{result['model']}(ratio={result['ratio']}, gnn_stage={result['gnn_stage']})"
        rankings[key] = {
            'rank': rank,
            'test_metric': result['test_metric']
        }
    
    return rankings

def analyze_gnn_enhancement(results_folder, datasets_folder, output_folder, comparison_mode='single_model'):
    """
    分析GNN增強效果
    
    Args:
        results_folder: 實驗結果資料夾
        datasets_folder: 資料集資料夾
        output_folder: 輸出資料夾
        comparison_mode: 比較模式
            - 'single_model': 17個競爭者（預設）
            - 'all_models': 118個競爭者
    """
    
    # 載入資料集分類
    print("載入資料集分類...")
    classifications = load_dataset_classifications(datasets_folder)
    print(f"找到 {len(classifications)} 個資料集")
    
    # 載入所有實驗結果
    print("\n載入實驗結果...")
    all_results = []
    results_path = Path(results_folder)
    
    for result_file in results_path.glob('*.txt'):
        results = parse_result_file(result_file)
        all_results.extend(results)
        print(f"  處理: {result_file.name} - 找到 {len(results)} 筆結果")
    
    print(f"\n總共載入 {len(all_results)} 筆實驗結果")
    print(f"比較模式: {comparison_mode}")
    
    if comparison_mode == 'single_model':
        # now: 7 configs for the target model + 6 reference models * 2 ratios = 7 + 12 = 19
        expected_competitors = 19
        print("  - 目標模型: 7個配置（大訓練集baseline + 小訓練集6變體）")
        print("  - 參考模型: 12個配置（6個模型 × 2種ratio）")
    else:  # all_models
        # now: 10 insertable models × 6 variants × 2 ratios = 120
        # plus 6 reference models × 2 ratios = 12 -> total 132
        expected_competitors = 132
        print("  - 可插入GNN的模型: 120個配置（10個模型 × 6變體 × 2種ratio）")
        print("  - 參考模型: 12個配置（6個模型 × 2種ratio）")
    
    # 組織資料：按模型、分類、資料集分組
    organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 同時收集參考模型的資料
    reference_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for result in all_results:
        dataset = result['dataset']
        model = result['model']
        classification = classifications.get(dataset, 'unknown')
        
        if classification == 'unknown':
            continue
        
        if model in REFERENCE_MODELS:
            reference_data[model][classification][dataset].append(result)
        else:
            organized_data[model][classification][dataset].append(result)
    
    # 為每個可插入GNN的模型生成報告
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    all_model_rankings = {}  # 儲存所有模型的排名資料
    
    for model in GNN_INSERTABLE_MODELS:
        print(f"\n分析模型: {model}")
        
        if model not in organized_data:
            print(f"  警告: 找不到模型 {model} 的資料")
            continue
        
        model_rankings = {}  # 該模型在各分類下的排名
        
        # 對每個分類進行分析
        for classification in sorted(organized_data[model].keys()):
            datasets_in_class = organized_data[model][classification]
            
            # 收集該分類下所有資料集的排名
            classification_rankings = defaultdict(list)
            
            for dataset, dataset_results in datasets_in_class.items():
                # 準備競爭者的結果
                competitors = []
                
                if comparison_mode == 'single_model':
                    # 模式1: 17個競爭者
                    # 1. 該模型在大訓練集的baseline (0.8/0.15/0.05, none)
                    # 2. 該模型在小訓練集的6種變體 (0.05/0.15/0.8, all stages)
                    for result in dataset_results:
                        if result['model'] == model:
                            # 只保留: 大訓練集的none + 小訓練集的所有變體
                            if (result['ratio'] == '0.8/0.15/0.05' and result['gnn_stage'] == 'none') or \
                               (result['ratio'] == '0.05/0.15/0.8'):
                                competitors.append(result)
                    
                    # 3. 5個參考模型在兩種ratio下的表現
                    for ref_model in REFERENCE_MODELS:
                        if ref_model in reference_data:
                            if classification in reference_data[ref_model]:
                                if dataset in reference_data[ref_model][classification]:
                                    for result in reference_data[ref_model][classification][dataset]:
                                        competitors.append(result)
                
                else:  # all_models
                    # 模式2: 118個競爭者
                    # 1. 所有可插入GNN的模型的所有配置
                    for gnn_model in GNN_INSERTABLE_MODELS:
                        if gnn_model in organized_data:
                            if classification in organized_data[gnn_model]:
                                if dataset in organized_data[gnn_model][classification]:
                                    for result in organized_data[gnn_model][classification][dataset]:
                                        competitors.append(result)
                    
                    # 2. 5個參考模型在兩種ratio下的表現
                    for ref_model in REFERENCE_MODELS:
                        if ref_model in reference_data:
                            if classification in reference_data[ref_model]:
                                if dataset in reference_data[ref_model][classification]:
                                    for result in reference_data[ref_model][classification][dataset]:
                                        competitors.append(result)
                
                # 進行排名
                if competitors:
                    task_type = competitors[0]['task_type']
                    rankings = rank_competitors(competitors, task_type)
                    
                    # 記錄排名
                    for competitor_key, rank_info in rankings.items():
                        classification_rankings[competitor_key].append(rank_info['rank'])
            
            # 計算平均排名
            avg_rankings = {}
            for competitor_key, ranks in classification_rankings.items():
                avg_rankings[competitor_key] = {
                    'avg_rank': sum(ranks) / len(ranks),
                    'count': len(ranks),
                    'ranks': ranks
                }
            
            model_rankings[classification] = {
                'dataset_count': len(datasets_in_class),
                'rankings': avg_rankings
            }
        
        all_model_rankings[model] = model_rankings
        
        # 只在 single_model 模式下輸出每個模型的文字報告
        if comparison_mode == 'single_model':
            output_file = output_path / f"{model}_gnn_enhancement.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"{'='*100}\n")
                f.write(f"GNN增強效果分析 - 模型: {model.upper()}\n")
                f.write(f"{'='*100}\n\n")
                f.write("說明：\n")
                f.write(f"- 比較 {model} 模型的原始表現與GNN增強變體\n")
                f.write("- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）\n")
                f.write("- 共17個競爭者：\n")
                f.write(f"  * {model}的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)\n")
                f.write("  * 5個參考模型的兩種ratio：每個模型2個配置(10)\n")
                f.write("- 排名越小表示表現越好\n\n")
                
                for classification in sorted(model_rankings.keys()):
                    class_data = model_rankings[classification]
                    f.write(f"\n{'='*100}\n")
                    f.write(f"分類: {classification} (包含 {class_data['dataset_count']} 個資料集)\n")
                    f.write(f"{'='*100}\n\n")
                    
                    # 按平均排名排序
                    sorted_competitors = sorted(
                        class_data['rankings'].items(),
                        key=lambda x: x[1]['avg_rank']
                    )
                    
                    f.write(f"{'排名':<8}{'競爭者':<70}{'平均排名':<12}{'資料集數':<10}\n")
                    f.write(f"{'-'*100}\n")
                    
                    for rank, (competitor, data) in enumerate(sorted_competitors, 1):
                        f.write(f"{rank:<8}{competitor:<70}{data['avg_rank']:<12.2f}{data['count']:<10}\n")
            
            print(f"  生成報告: {output_file}")
    
    # 在 all_models 模式下，生成按分類的總排名報告
    if comparison_mode == 'all_models':
        print("\n生成跨模型總排名報告...")
        cross_model_file = output_path / "all_models_ranking_by_classification.txt"
        
        # 收集所有分類的排名資料
        classification_rankings = defaultdict(lambda: defaultdict(list))
        classification_dataset_counts = {}  # 記錄每個分類的實際資料集數量
        
        for model in GNN_INSERTABLE_MODELS:
            if model not in all_model_rankings:
                continue
            for classification, class_data in all_model_rankings[model].items():
                # 記錄該分類的資料集數量（只需記錄一次）
                if classification not in classification_dataset_counts:
                    classification_dataset_counts[classification] = class_data['dataset_count']
                
                for competitor, rank_data in class_data['rankings'].items():
                    classification_rankings[classification][competitor].extend(rank_data['ranks'])
        
        # 計算每個競爭者在各分類下的平均排名
        classification_avg_rankings = {}
        for classification, competitors in classification_rankings.items():
            classification_avg_rankings[classification] = {
                competitor: {
                    'avg_rank': sum(ranks) / len(ranks),
                    'count': len(ranks)
                }
                for competitor, ranks in competitors.items()
            }
        
        # 寫入報告
        with open(cross_model_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*100}\n")
            f.write("跨模型總排名 - 按資料集分類\n")
            f.write(f"{'='*100}\n\n")
            f.write("說明：\n")
            f.write("- 包含所有9個可插入GNN的模型和5個參考模型\n")
            f.write("- 共118個競爭者：\n")
            f.write("  * 9個可插入GNN模型：每個6變體 × 2種ratio = 108個配置\n")
            f.write("  * 5個參考模型：每個2種ratio = 10個配置\n")
            f.write("- 排名越小表示表現越好\n\n")
            
            for classification in sorted(classification_avg_rankings.keys()):
                avg_rankings = classification_avg_rankings[classification]
                
                # 取得實際的資料集數量
                dataset_count = classification_dataset_counts.get(classification, 0)
                
                f.write(f"\n{'='*100}\n")
                f.write(f"分類: {classification} (包含 {dataset_count} 個資料集)\n")
                f.write(f"{'='*100}\n\n")
                
                # 按平均排名排序
                sorted_competitors = sorted(
                    avg_rankings.items(),
                    key=lambda x: x[1]['avg_rank']
                )
                
                f.write(f"{'排名':<8}{'競爭者':<70}{'平均排名':<12}{'出現次數':<10}\n")
                f.write(f"{'-'*100}\n")
                
                for rank, (competitor, data) in enumerate(sorted_competitors, 1):
                    f.write(f"{rank:<8}{competitor:<70}{data['avg_rank']:<12.2f}{data['count']:<10}\n")
        
        print(f"  生成跨模型排名報告: {cross_model_file}")
    
    # 輸出JSON格式
    mode_suffix = f"_{comparison_mode}" if comparison_mode == 'all_models' else ""
    json_file = output_path / f"gnn_enhancement_all_models{mode_suffix}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_model_rankings, f, indent=2, ensure_ascii=False)
    print(f"\n生成JSON檔案: {json_file}")
    
    # 生成總體摘要
    summary_file = output_path / f"gnn_enhancement_summary{mode_suffix}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write("GNN增強效果總體摘要\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"比較模式: {comparison_mode}\n")
        f.write(f"競爭者數量: {expected_competitors}\n\n")
        
        for model in GNN_INSERTABLE_MODELS:
            if model not in all_model_rankings:
                continue
                
            f.write(f"\n{'='*100}\n")
            f.write(f"模型: {model.upper()}\n")
            f.write(f"{'='*100}\n\n")
            
            model_data = all_model_rankings[model]
            
            # 找出該模型最佳的GNN配置
            best_configs = defaultdict(list)
            
            for classification, class_data in model_data.items():
                rankings = class_data['rankings']
                
                # 找出該模型自己的配置中最好的
                model_configs = {k: v for k, v in rankings.items() if k.startswith(f"{model}(")}
                
                if model_configs:
                    best_config = min(model_configs.items(), key=lambda x: x[1]['avg_rank'])
                    best_configs[classification] = best_config
                    
                    # 計算相對於大訓練集baseline的改善
                    baseline_key = f"{model}(ratio=0.8/0.15/0.05, gnn_stage=none)"
                    if baseline_key in rankings:
                        baseline_rank = rankings[baseline_key]['avg_rank']
                        improvement = baseline_rank - best_config[1]['avg_rank']
                        
                        f.write(f"分類: {classification}\n")
                        f.write(f"  資料集數: {class_data['dataset_count']}\n")
                        f.write(f"  Baseline (0.8/0.15/0.05, none): 平均排名 {baseline_rank:.2f}\n")
                        f.write(f"  最佳配置: {best_config[0]}\n")
                        f.write(f"  最佳平均排名: {best_config[1]['avg_rank']:.2f}\n")
                        f.write(f"  改善: {improvement:+.2f} {'✓' if improvement > 0 else '✗'}\n\n")
    
    print(f"生成總體摘要: {summary_file}")
    print("\n分析完成！")

if __name__ == "__main__":
    # 從命令列參數獲取比較模式
    comparison_mode = 'all_models'
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg in ['single_model', 'all_models']:
            comparison_mode = mode_arg
        else:
            print(f"警告: 無效的模式 '{sys.argv[1]}'，使用預設模式 'single_model'")
            print("可用模式: 'single_model' 或 'all_models'")
    
    results_folder = "/home/shangyuan/ModelComparison/TaBLEau/summary_results"
    datasets_folder = "/home/shangyuan/ModelComparison/TaBLEau/datasets"
    output_folder = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis"
    
    print(f"\n{'='*100}")
    print(f"GNN增強效果分析")
    print(f"{'='*100}")
    print(f"比較模式: {comparison_mode}\n")
    
    analyze_gnn_enhancement(results_folder, datasets_folder, output_folder, comparison_mode)
