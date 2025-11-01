#!/usr/bin/env python3
"""
生成GNN增強模型相對排名 - 重新計算在50個GNN變體之間的排名
從原始實驗結果重新讀取每個數據集的性能指標，只在50個GNN變體中重新排名
"""

import json
from pathlib import Path
from collections import defaultdict

# 可以插入GNN的模型
GNN_INSERTABLE_MODELS = [
    'excelformer', 'fttransformer', 'resnet', 'tabnet', 
    'tabtransformer', 'trompt', 'vime', 'scarf', 'subtab', 'tabm'
]

# GNN插入階段（排除none）
GNN_STAGES_ONLY = ['start', 'materialize', 'encoding', 'columnwise', 'decoding']

def load_experiment_results(results_folder):
    """載入所有實驗結果"""
    # The summary files are plain text reports placed in summary_results.
    # We'll parse all .txt files and extract dataset / model / ratio / gnn_stage / Best test metric.
    results = defaultdict(lambda: defaultdict(dict))
    results_path = Path(results_folder)

    import re

    def parse_txt_file(fp):
        """Parse a summary .txt file and return list of records.

        Each record: {'dataset', 'model', 'gnn_stage', 'ratio', 'test_metric', 'task_type'}
        """
        txt = fp.read_text(encoding='utf-8', errors='ignore')
        records = []

        # infer ratio and model from filename
        name = fp.stem
        # filename pattern contains model and ratio at end like ..._0.05_0.15_0.8.txt
        ratio_match = re.search(r'(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)$', name)
        ratio = None
        if ratio_match:
            ratio = f"{ratio_match.group(1)}/{ratio_match.group(2)}/{ratio_match.group(3)}"

        # try to extract model name (one of insertable models or others)
        model_name = None
        for m in GNN_INSERTABLE_MODELS:
            if m in name:
                model_name = m
                break
        # also consider tabgnn or reference models may be in filename
        if not model_name:
            for m in ['tabgnn','t2gformer','tabpfn','xgboost','catboost','lightgbm']:
                if m in name:
                    model_name = m
                    break

        # split into dataset blocks by 'dataset:' markers
        parts = re.split(r'dataset:\s*(\S+)', txt)[1:]
        for i in range(0, len(parts), 2):
            ds = parts[i].strip()
            block = parts[i+1]
            # detect task type if present
            task_match = re.search(r'任務類型:\s*(\w+)', block)
            task = task_match.group(1) if task_match else 'unknown'

            # find each GNN stage block
            stage_blocks = re.split(r'GNN階段:\s*(\S+)', block)[1:]
            for j in range(0, len(stage_blocks), 2):
                stage = stage_blocks[j].strip()
                content = stage_blocks[j+1]
                # find Best test metric
                m = re.search(r'Best test metric:\s*([0-9\.eE+-]+)', content)
                if m:
                    try:
                        metric = float(m.group(1))
                    except:
                        continue
                    rec = {
                        'dataset': ds,
                        'model': model_name,
                        'gnn_stage': stage,
                        'ratio': ratio,
                        'test_metric': metric,
                        'task_type': task
                    }
                    records.append(rec)
        return records

    for txt_file in results_path.glob('*.txt'):
        try:
            recs = parse_txt_file(txt_file)
            for r in recs:
                # only consider insertable models and their GNN stages (exclude none)
                if r['model'] not in GNN_INSERTABLE_MODELS:
                    continue
                if r['gnn_stage'] not in GNN_STAGES_ONLY:
                    continue
                # store
                competitor_key = f"{r['model']}(ratio={r['ratio']}, gnn_stage={r['gnn_stage']})"
                results[r['dataset']][competitor_key] = {
                    'metric': r['test_metric'],
                    'task': r['task_type'],
                    'model': r['model'],
                    'ratio': r['ratio'],
                    'stage': r['gnn_stage']
                }
        except Exception as e:
            print(f"警告: 無法處理文件 {txt_file}: {e}")
            continue

    return results

def classify_dataset(dataset_name, results):
    """根據數據集名稱和結果分類數據集"""
    # 這裡需要根據實際的數據集分類邏輯來實現
    # 暫時返回一個默認分類
    return "unknown_classification"

def generate_relative_ranking(results_folder, target_ratio, output_folder, dataset_classifications):
    """
    為特定 ratio 生成GNN變體之間的相對排名
    
    Args:
        results_folder: 實驗結果資料夾路徑
        target_ratio: '0.8/0.15/0.05' 或 '0.05/0.15/0.8'
        output_folder: 輸出資料夾路徑
        dataset_classifications: 數據集分類字典
    """
    output_path = Path(output_folder)
    ratio_suffix = target_ratio.replace('/', '-')
    output_file = output_path / f"gnn_relative_ranking_{ratio_suffix}.txt"
    
    print(f"\n處理 Train/Val/Test Ratio: {target_ratio}")
    
    # 載入實驗結果
    print("  載入實驗結果...")
    all_results = load_experiment_results(results_folder)
    
    # 按分類組織數據
    classification_rankings = defaultdict(lambda: defaultdict(list))
    classification_dataset_counts = defaultdict(int)
    
    for dataset_name, competitors in all_results.items():
        # 過濾出目標ratio的GNN競爭者
        target_competitors = {
            k: v for k, v in competitors.items()
            if v['ratio'] == target_ratio and v['stage'] in GNN_STAGES_ONLY
        }
        
        if len(target_competitors) < 2:
            continue
        
        # 獲取數據集分類
        classification = dataset_classifications.get(dataset_name, 'unknown')
        classification_dataset_counts[classification] += 1
        
        # 根據task確定排序方向
        task = list(target_competitors.values())[0]['task']
        reverse = (task == 'regression')  # regression任務metric越小越好
        
        # 對這45個GNN變體進行排名
        sorted_competitors = sorted(
            target_competitors.items(),
            key=lambda x: x[1]['metric'],
            reverse=not reverse  # 如果reverse=True(regression)，則不reverse排序
        )
        
        # 分配排名（1-45）
        for rank, (competitor_key, _) in enumerate(sorted_competitors, 1):
            classification_rankings[classification][competitor_key].append(rank)
    
    # 計算平均排名
    classification_avg_rankings = {}
    for classification, competitors in classification_rankings.items():
        classification_avg_rankings[classification] = {
            competitor: {
                'avg_rank': sum(ranks) / len(ranks),
                'count': len(ranks)
            }
            for competitor, ranks in competitors.items()
        }
    
    # 統計競爭者數量
    all_competitors = set()
    for competitors in classification_rankings.values():
        all_competitors.update(competitors.keys())
    
    print(f"  找到 {len(all_competitors)} 個競爭者")
    print(f"  涵蓋 {len(classification_avg_rankings)} 個資料集分類")
    
    # 寫入報告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write(f"GNN增強模型相對排名 - Train/Val/Test Ratio: {target_ratio}\n")
        f.write(f"{'='*100}\n\n")
        f.write("說明：\n")
        f.write("- 只在45個GNN變體之間進行相對排名（排名範圍: 1-45）\n")
        f.write(f"- Train/Val/Test 比例: {target_ratio}\n")
        f.write(f"- 共 {len(all_competitors)} 個競爭者：\n")
        f.write("  * 9個模型：excelformer, fttransformer, resnet, tabnet, tabtransformer, trompt, vime, scarf, subtab\n")
        f.write("  * 5種GNN插入階段：start, materialize, encoding, columnwise, decoding\n")
        f.write("  * 不包含原始模型（none階段）和參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）\n")
        f.write("- 排名越小表示表現越好\n\n")
        
        for classification in sorted(classification_avg_rankings.keys()):
            avg_rankings = classification_avg_rankings[classification]
            dataset_count = classification_dataset_counts[classification]
            
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
            
            for rank, (competitor, comp_data) in enumerate(sorted_competitors, 1):
                f.write(f"{rank:<8}{competitor:<70}{comp_data['avg_rank']:<12.2f}{comp_data['count']:<10}\n")
    
    print(f"  生成報告: {output_file}")
    return output_file

def load_dataset_classifications():
    """載入數據集分類信息

    掃描 datasets 目錄結構：
    datasets/<size>/<task>/<feature>/<dataset_name>
    並返回一個 mapping: dataset_name -> "<size>+<task>+<feature>"
    """
    classifications = {}
    base = Path('/home/shangyuan/ModelComparison/TaBLEau/datasets')
    if not base.exists():
        return classifications

    for size_dir in base.iterdir():
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
                    classifications[dataset_name] = f"{size}+{task}+{feature}"

    return classifications

def main():
    # 路徑設置
    # summary_results contains the per-model text reports generated earlier
    results_folder = "/home/shangyuan/ModelComparison/TaBLEau/summary_results"
    output_folder = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis"
    
    print("="*100)
    print("GNN增強模型相對排名分析（重新計算45個GNN變體之間的排名）")
    print("="*100)
    
    # 載入數據集分類
    print("\n載入數據集分類...")
    dataset_classifications = load_dataset_classifications()
    
    # 為兩種 ratio 生成報告
    print("\n生成GNN相對排名報告...")
    
    # 1. 大訓練集 (0.8/0.15/0.05)
    generate_relative_ranking(results_folder, '0.8/0.15/0.05', output_folder, dataset_classifications)
    
    # 2. 小訓練集 (0.05/0.15/0.8)
    generate_relative_ranking(results_folder, '0.05/0.15/0.8', output_folder, dataset_classifications)
    
    print("\n注意: 此腳本需要訪問原始實驗結果文件才能重新計算排名")
    print("請確保 results 資料夾中包含所有實驗結果的JSON文件")

if __name__ == "__main__":
    main()
