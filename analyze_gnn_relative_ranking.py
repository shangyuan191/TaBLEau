#!/usr/bin/env python3
"""
生成GNN增強模型相對排名 - 重新計算在45個GNN變體之間的排名
從原始實驗結果重新讀取每個數據集的性能指標，只在45個GNN變體中重新排名
"""

import json
from pathlib import Path
from collections import defaultdict

# 可以插入GNN的模型
GNN_INSERTABLE_MODELS = [
    'excelformer', 'fttransformer', 'resnet', 'tabnet', 
    'tabtransformer', 'trompt', 'vime', 'scarf', 'subtab'
]

# GNN插入階段（排除none）
GNN_STAGES_ONLY = ['start', 'materialize', 'encoding', 'columnwise', 'decoding']

def load_experiment_results(results_folder):
    """載入所有實驗結果"""
    results = defaultdict(lambda: defaultdict(dict))
    results_path = Path(results_folder)
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 解析文件名獲取模型、ratio和stage信息
            filename = json_file.stem
            parts = filename.split('_')
            
            # 找到模型名
            model_name = None
            for model in GNN_INSERTABLE_MODELS:
                if model in filename:
                    model_name = model
                    break
            
            if not model_name:
                continue
                
            # 解析ratio和stage
            ratio = None
            stage = None
            
            for i, part in enumerate(parts):
                if part == 'ratio':
                    ratio = f"{parts[i+1]}/{parts[i+2]}/{parts[i+3]}"
                elif part == 'stage':
                    stage = parts[i+1]
            
            if not ratio or not stage or stage == 'none':
                continue
            
            # 只保留GNN階段
            if stage not in GNN_STAGES_ONLY:
                continue
            
            # 存儲每個數據集的結果
            for dataset_name, dataset_results in data.items():
                if isinstance(dataset_results, dict) and 'test_metric' in dataset_results:
                    competitor_key = f"{model_name}(ratio={ratio}, gnn_stage={stage})"
                    results[dataset_name][competitor_key] = {
                        'metric': dataset_results['test_metric'],
                        'task': dataset_results.get('task', 'unknown'),
                        'model': model_name,
                        'ratio': ratio,
                        'stage': stage
                    }
        except Exception as e:
            print(f"警告: 無法處理文件 {json_file}: {e}")
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
    """載入數據集分類信息"""
    # 從現有JSON文件中提取數據集分類
    json_file = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/gnn_enhancement_all_models_all_models.json"
    
    classifications = {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 從任一模型的數據中提取分類信息
        if data:
            first_model = list(data.keys())[0]
            model_data = data[first_model]
            
            for classification, class_data in model_data.items():
                # 從rankings中提取數據集名稱
                # 注意：這裡需要有實際的數據集名稱，暫時先跳過
                pass
    except Exception as e:
        print(f"警告: 無法載入數據集分類: {e}")
    
    return classifications

def main():
    # 路徑設置
    results_folder = "/home/shangyuan/ModelComparison/TaBLEau/results"
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
    # generate_relative_ranking(results_folder, '0.8/0.15/0.05', output_folder, dataset_classifications)
    
    # 2. 小訓練集 (0.05/0.15/0.8)
    # generate_relative_ranking(results_folder, '0.05/0.15/0.8', output_folder, dataset_classifications)
    
    print("\n注意: 此腳本需要訪問原始實驗結果文件才能重新計算排名")
    print("請確保 results 資料夾中包含所有實驗結果的JSON文件")

if __name__ == "__main__":
    main()
