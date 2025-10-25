#!/usr/bin/env python3
"""
分析GNN增強效果：比較每個可插入GNN的模型的原始表現與GNN增強變體
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# 可以插入GNN的模型
GNN_INSERTABLE_MODELS = [
    'excelformer', 'fttransformer', 'resnet', 'tabnet', 
    'tabtransformer', 'trompt', 'vime', 'scarf', 'subtab'
]

# 參考模型（無法插入GNN）
REFERENCE_MODELS = ['t2gformer', 'tabpfn', 'xgboost', 'catboost', 'lightgbm']

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

def analyze_gnn_enhancement(results_folder, datasets_folder, output_folder):
    """分析GNN增強效果"""
    
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
    
    # 組織資料：按模型、分類、資料集分組
    organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for result in all_results:
        dataset = result['dataset']
        model = result['model']
        classification = classifications.get(dataset, 'unknown')
        
        if classification == 'unknown':
            continue
        
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
                # 準備17個競爭者的結果
                competitors = []
                
                # 1. 該模型在大訓練集的原始表現 (0.8/0.15/0.05, none)
                # 2. 該模型在小訓練集的6種變體 (0.05/0.15/0.8, 6 stages)
                for result in dataset_results:
                    if result['model'] == model:
                        competitors.append(result)
                
                # 3-7. 5個參考模型在兩種ratio下的表現
                for ref_model in REFERENCE_MODELS:
                    if ref_model in organized_data:
                        if classification in organized_data[ref_model]:
                            if dataset in organized_data[ref_model][classification]:
                                for result in organized_data[ref_model][classification][dataset]:
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
        
        # 輸出該模型的文字報告
        output_file = output_path / f"{model}_gnn_enhancement.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*100}\n")
            f.write(f"GNN增強效果分析 - 模型: {model.upper()}\n")
            f.write(f"{'='*100}\n\n")
            f.write("說明：\n")
            f.write(f"- 比較 {model} 模型的原始表現與GNN增強變體\n")
            f.write("- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）\n")
            f.write("- 共17個競爭者：大訓練集原始(1) + 小訓練集6變體(6) + 參考模型兩種ratio(10)\n")
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
    
    # 輸出JSON格式
    json_file = output_path / "gnn_enhancement_all_models.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_model_rankings, f, indent=2, ensure_ascii=False)
    print(f"\n生成JSON檔案: {json_file}")
    
    # 生成總體摘要
    summary_file = output_path / "gnn_enhancement_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write("GNN增強效果總體摘要\n")
        f.write(f"{'='*100}\n\n")
        
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
    results_folder = "/home/shangyuan/ModelComparison/TaBLEau/summary_results"
    datasets_folder = "/home/shangyuan/ModelComparison/TaBLEau/datasets"
    output_folder = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis"
    
    analyze_gnn_enhancement(results_folder, datasets_folder, output_folder)
