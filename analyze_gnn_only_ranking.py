#!/usr/bin/env python3
"""
生成GNN增強模型專屬排名 - 按train/val/test ratio分開
只包含9個可插入GNN的模型的5種GNN階段（排除none和參考模型）
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

def load_all_models_json(json_file):
    """載入 all_models 模式的 JSON 檔案"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_gnn_only_ranking(data, target_ratio, output_folder, recalculate_ranks=False):
    """
    為特定 ratio 生成只包含GNN階段的排名
    
    Args:
        data: 從 JSON 載入的所有模型資料
        target_ratio: '0.8/0.15/0.05' 或 '0.05/0.15/0.8'
        output_folder: 輸出資料夾路徑
        recalculate_ranks: 是否重新計算在45個GNN變體之間的相對排名
    """
    output_path = Path(output_folder)
    ratio_suffix = target_ratio.replace('/', '-')
    
    if recalculate_ranks:
        output_file = output_path / f"gnn_relative_ranking_{ratio_suffix}.txt"
    else:
        output_file = output_path / f"gnn_only_ranking_{ratio_suffix}.txt"
    
    print(f"\n處理 Train/Val/Test Ratio: {target_ratio}")
    if recalculate_ranks:
        print("  模式: 重新計算45個GNN變體之間的相對排名")
    
    # 收集所有分類的排名資料
    classification_raw_ranks = defaultdict(lambda: defaultdict(dict))  # [classification][dataset_idx][competitor] = original_rank
    classification_dataset_counts = {}
    
    # 第一步：收集所有GNN競爭者的原始排名
    for model in GNN_INSERTABLE_MODELS:
        if model not in data:
            continue
        
        model_data = data[model]
        
        for classification, class_data in model_data.items():
            # 記錄該分類的資料集數量
            if classification not in classification_dataset_counts:
                classification_dataset_counts[classification] = class_data['dataset_count']
            
            rankings = class_data['rankings']
            
            for competitor, rank_data in rankings.items():
                # 只保留：特定ratio + GNN階段（排除none）
                if f'ratio={target_ratio}' in competitor and 'gnn_stage=none' not in competitor:
                    # 確認競爭者屬於9個可插入GNN的模型
                    is_gnn_model = any(f'{m}(' in competitor for m in GNN_INSERTABLE_MODELS)
                    if is_gnn_model:
                        # 存儲每個數據集上的原始排名
                        for idx, rank in enumerate(rank_data['ranks']):
                            classification_raw_ranks[classification][idx][competitor] = rank
    
    # 第二步：計算排名
    classification_rankings = defaultdict(lambda: defaultdict(list))
    
    if recalculate_ranks:
        # 重新計算：在45個GNN變體中的相對排名
        print("  重新計算45個GNN變體之間的相對排名...")
        for classification, datasets in classification_raw_ranks.items():
            for dataset_idx, competitors in datasets.items():
                if len(competitors) < 2:
                    continue
                
                # 按原始排名排序（原始排名越小越好）
                sorted_competitors = sorted(competitors.items(), key=lambda x: x[1])
                
                # 分配新的相對排名（1到實際競爭者數量）
                for new_rank, (competitor, _) in enumerate(sorted_competitors, 1):
                    classification_rankings[classification][competitor].append(new_rank)
    else:
        # 使用原始排名（在118個競爭者中）
        for classification, datasets in classification_raw_ranks.items():
            for dataset_idx, competitors in datasets.items():
                for competitor, rank in competitors.items():
                    classification_rankings[classification][competitor].append(rank)
    
    # 計算每個競爭者在各分類下的平均排名
    classification_avg_rankings = {}
    for classification, competitors in classification_rankings.items():
        if competitors:  # 只處理有資料的分類
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
        if recalculate_ranks:
            f.write(f"GNN增強模型相對排名 - Train/Val/Test Ratio: {target_ratio}\n")
        else:
            f.write(f"GNN增強模型專屬排名 - Train/Val/Test Ratio: {target_ratio}\n")
        f.write(f"{'='*100}\n\n")
        f.write("說明：\n")
        f.write("- 只包含9個可插入GNN的模型的GNN增強變體\n")
        f.write(f"- Train/Val/Test 比例: {target_ratio}\n")
        f.write(f"- 共 {len(all_competitors)} 個競爭者：\n")
        f.write("  * 9個模型：excelformer, fttransformer, resnet, tabnet, tabtransformer, trompt, vime, scarf, subtab\n")
        f.write("  * 5種GNN插入階段：start, materialize, encoding, columnwise, decoding\n")
        f.write("  * 不包含原始模型（none階段）和參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）\n")
        if recalculate_ranks:
            f.write("- 排名計算：只在這45個GNN變體之間重新計算相對排名（排名範圍: 1-45）\n")
        else:
            f.write("- 排名計算：在所有118個競爭者（包含none和參考模型）中的排名\n")
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
            
            for rank, (competitor, comp_data) in enumerate(sorted_competitors, 1):
                f.write(f"{rank:<8}{competitor:<70}{comp_data['avg_rank']:<12.2f}{comp_data['count']:<10}\n")
    
    print(f"  生成報告: {output_file}")
    return output_file

def generate_summary(output_folder):
    """生成兩份報告的總結比較"""
    output_path = Path(output_folder)
    summary_file = output_path / "gnn_only_ranking_summary.txt"
    
    # 讀取兩份報告的前5名
    ratios = ['0.8-0.15-0.05', '0.05-0.15-0.8']
    ratio_names = ['大訓練集 (0.8/0.15/0.05)', '小訓練集 (0.05/0.15/0.8)']
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write("GNN增強模型排名總結\n")
        f.write(f"{'='*100}\n\n")
        f.write("說明：\n")
        f.write("- 比較兩種不同 Train/Val/Test 比例下的GNN增強效果\n")
        f.write("- 只包含5種GNN插入階段（start, materialize, encoding, columnwise, decoding）\n")
        f.write("- 排除原始模型（none）和參考模型\n\n")
        
        f.write(f"生成的報告檔案：\n")
        f.write(f"- gnn_only_ranking_0.8-0.15-0.05.txt (大訓練集，45個競爭者)\n")
        f.write(f"- gnn_only_ranking_0.05-0.15-0.8.txt (小訓練集，45個競爭者)\n\n")
        
        f.write(f"{'='*100}\n")
        f.write("分析要點\n")
        f.write(f"{'='*100}\n\n")
        f.write("1. 哪些GNN插入階段在不同訓練集大小下表現最好？\n")
        f.write("2. 哪些模型從GNN增強中獲益最多？\n")
        f.write("3. 大訓練集 vs 小訓練集：GNN增強的效果差異\n")
        f.write("4. 不同資料集分類下的最佳GNN配置\n\n")
    
    print(f"\n生成總結檔案: {summary_file}")

def main():
    # 輸入和輸出路徑
    json_file = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/gnn_enhancement_all_models_all_models.json"
    output_folder = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis"
    
    print("="*100)
    print("GNN增強模型專屬排名分析")
    print("="*100)
    
    # 載入資料
    print("\n載入資料...")
    data = load_all_models_json(json_file)
    print(f"載入 {len(data)} 個模型的資料")
    
    # 為兩種 ratio 生成報告
    print("\n生成GNN相對排名報告（重新計算45個GNN變體之間的排名）...")
    
    # 1. 大訓練集 (0.8/0.15/0.05) - 相對排名
    generate_gnn_only_ranking(data, '0.8/0.15/0.05', output_folder, recalculate_ranks=True)
    
    # 2. 小訓練集 (0.05/0.15/0.8) - 相對排名
    generate_gnn_only_ranking(data, '0.05/0.15/0.8', output_folder, recalculate_ranks=True)
    
    # 生成總結
    print("\n生成總結檔案...")
    generate_summary(output_folder)
    
    print("\n" + "="*100)
    print("分析完成！")
    print("="*100)
    print("\n生成的檔案：")
    print("- gnn_relative_ranking_0.8-0.15-0.05.txt (大訓練集，45個GNN變體，相對排名1-45)")
    print("- gnn_relative_ranking_0.05-0.15-0.8.txt (小訓練集，45個GNN變體，相對排名1-45)")
    print("- gnn_only_ranking_summary.txt (總結)")
    print("\n說明：")
    print("- 相對排名：在45個GNN變體之間重新計算的排名（1-45）")
    print("- 之前的檔案（gnn_only_ranking_*.txt）保留了在118個競爭者中的原始排名")

if __name__ == "__main__":
    main()
