#!/usr/bin/env python3
"""
生成GNN模型整體排名 - 跨所有數據集的總排名
不區分數據集分類，計算每個模型在所有數據集上的平均表現
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

def load_all_models_json(json_file):
    """載入 all_models 模式的 JSON 檔案"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_overall_ranking(data, target_ratio, output_folder):
    """
    為特定 ratio 生成跨所有數據集的整體排名
    
    Args:
        data: 從 JSON 載入的所有模型資料
        target_ratio: '0.8/0.15/0.05' 或 '0.05/0.15/0.8'
        output_folder: 輸出資料夾路徑
    """
    output_path = Path(output_folder)
    ratio_suffix = target_ratio.replace('/', '-')
    output_file = output_path / f"gnn_overall_ranking_{ratio_suffix}.txt"
    
    print(f"\n處理 Train/Val/Test Ratio: {target_ratio}")
    
    # 收集所有數據集的排名資料（不區分分類）
    competitor_all_ranks = defaultdict(list)
    total_datasets = 0
    classification_dataset_counts = {}
    
    # 第一步：收集所有GNN競爭者在所有數據集上的原始排名
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
                    # 確認競爭者屬於10個可插入GNN的模型
                    is_gnn_model = any(f'{m}(' in competitor for m in GNN_INSERTABLE_MODELS)
                    if is_gnn_model:
                        # 收集所有排名（跨所有分類）
                        competitor_all_ranks[competitor].extend(rank_data['ranks'])
    
    # 計算總數據集數量
    total_datasets = sum(classification_dataset_counts.values())
    
    # 第二步：為每個數據集重新計算50個GNN變體之間的相對排名
    print("  重新計算50個GNN變體之間的相對排名...")
    
    # 按數據集索引組織排名
    dataset_raw_ranks = defaultdict(dict)
    
    for model in GNN_INSERTABLE_MODELS:
        if model not in data:
            continue
        
        model_data = data[model]
        
        for classification, class_data in model_data.items():
            rankings = class_data['rankings']
            
            for competitor, rank_data in rankings.items():
                if f'ratio={target_ratio}' in competitor and 'gnn_stage=none' not in competitor:
                    is_gnn_model = any(f'{m}(' in competitor for m in GNN_INSERTABLE_MODELS)
                    if is_gnn_model:
                        # 為每個數據集存儲原始排名
                        # 使用classification和索引的組合作為唯一標識
                        for idx, rank in enumerate(rank_data['ranks']):
                            dataset_key = f"{classification}_{idx}"
                            dataset_raw_ranks[dataset_key][competitor] = rank
    
    # 計算相對排名
    competitor_relative_ranks = defaultdict(list)
    
    for dataset_key, competitors in dataset_raw_ranks.items():
        if len(competitors) < 2:
            continue
        
        # 按原始排名排序（原始排名越小越好）
        sorted_competitors = sorted(competitors.items(), key=lambda x: x[1])
        
        # 分配新的相對排名（1到實際競爭者數量）
        for new_rank, (competitor, _) in enumerate(sorted_competitors, 1):
            competitor_relative_ranks[competitor].append(new_rank)
    
    # 計算整體平均排名
    overall_rankings = {}
    for competitor, ranks in competitor_relative_ranks.items():
        overall_rankings[competitor] = {
            'avg_rank': sum(ranks) / len(ranks),
            'count': len(ranks),
            'min_rank': min(ranks),
            'max_rank': max(ranks)
        }
    
    # 統計信息
    all_competitors = list(overall_rankings.keys())
    print(f"  找到 {len(all_competitors)} 個競爭者")
    print(f"  涵蓋 {total_datasets} 個資料集（{len(classification_dataset_counts)} 個分類）")
    
    # 寫入報告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*120}\n")
        f.write(f"GNN增強模型整體排名 - Train/Val/Test Ratio: {target_ratio}\n")
        f.write(f"{'='*120}\n\n")
        f.write("說明：\n")
        f.write("- 整體排名：跨所有數據集的平均表現（不區分數據集分類）\n")
        f.write(f"- Train/Val/Test 比例: {target_ratio}\n")
        f.write(f"- 共 {len(all_competitors)} 個競爭者：\n")
        f.write("  * 10個模型：excelformer, fttransformer, resnet, tabnet, tabtransformer, trompt, vime, scarf, subtab, tabm\n")
        f.write("  * 5種GNN插入階段：start, materialize, encoding, columnwise, decoding\n")
        f.write("  * 不包含原始模型（none階段）與參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm, tabgnn）\n")
        f.write("- 排名計算：只在這50個GNN變體之間重新計算相對排名（排名範圍: 1-50）\n")
        f.write(f"- 數據集總數: {total_datasets} 個（涵蓋 {len(classification_dataset_counts)} 種分類）\n")
        f.write("- 排名越小表示表現越好\n\n")
        
        # 按平均排名排序
        sorted_competitors = sorted(
            overall_rankings.items(),
            key=lambda x: x[1]['avg_rank']
        )
        
        f.write(f"{'='*120}\n")
        f.write(f"整體排名（跨所有 {total_datasets} 個資料集）\n")
        f.write(f"{'='*120}\n\n")
        f.write(f"{'排名':<8}{'競爭者':<75}{'平均排名':<12}{'最佳':<8}{'最差':<8}{'數據集數':<10}\n")
        f.write(f"{'-'*120}\n")
        
        for rank, (competitor, comp_data) in enumerate(sorted_competitors, 1):
            f.write(f"{rank:<8}{competitor:<75}{comp_data['avg_rank']:<12.2f}")
            f.write(f"{comp_data['min_rank']:<8}{comp_data['max_rank']:<8}{comp_data['count']:<10}\n")
        
        # 添加統計分析
        f.write(f"\n{'='*120}\n")
        f.write("統計分析\n")
        f.write(f"{'='*120}\n\n")
        
        # 按模型分組統計
        model_stats = defaultdict(list)
        for competitor, comp_data in overall_rankings.items():
            for model in GNN_INSERTABLE_MODELS:
                if f"{model}(" in competitor:
                    model_stats[model].append(comp_data['avg_rank'])
                    break
        
        f.write("各模型的平均表現（所有GNN階段的平均）：\n\n")
        model_avg = []
        for model in GNN_INSERTABLE_MODELS:
            if model in model_stats:
                avg = sum(model_stats[model]) / len(model_stats[model])
                model_avg.append((model, avg, len(model_stats[model])))
        
        model_avg.sort(key=lambda x: x[1])
        f.write(f"{'排名':<8}{'模型':<20}{'平均排名':<15}{'變體數量':<10}\n")
        f.write(f"{'-'*60}\n")
        for rank, (model, avg, count) in enumerate(model_avg, 1):
            f.write(f"{rank:<8}{model:<20}{avg:<15.2f}{count:<10}\n")
        
        # 按階段分組統計
        stage_stats = defaultdict(list)
        for competitor, comp_data in overall_rankings.items():
            for stage in GNN_STAGES_ONLY:
                if f"gnn_stage={stage}" in competitor:
                    stage_stats[stage].append(comp_data['avg_rank'])
                    break
        
        f.write("\n各GNN階段的平均表現（所有模型的平均）：\n\n")
        stage_avg = []
        for stage in GNN_STAGES_ONLY:
            if stage in stage_stats:
                avg = sum(stage_stats[stage]) / len(stage_stats[stage])
                stage_avg.append((stage, avg, len(stage_stats[stage])))
        
        stage_avg.sort(key=lambda x: x[1])
        f.write(f"{'排名':<8}{'GNN階段':<20}{'平均排名':<15}{'變體數量':<10}\n")
        f.write(f"{'-'*60}\n")
        for rank, (stage, avg, count) in enumerate(stage_avg, 1):
            f.write(f"{rank:<8}{stage:<20}{avg:<15.2f}{count:<10}\n")
        
        # Top 10 和 Bottom 10
        f.write(f"\n{'='*120}\n")
        f.write("Top 10 最佳配置\n")
        f.write(f"{'='*120}\n\n")
        f.write(f"{'排名':<8}{'競爭者':<75}{'平均排名':<15}\n")
        f.write(f"{'-'*100}\n")
        for rank, (competitor, comp_data) in enumerate(sorted_competitors[:10], 1):
            f.write(f"{rank:<8}{competitor:<75}{comp_data['avg_rank']:<15.2f}\n")
        
        f.write(f"\n{'='*120}\n")
        f.write("Bottom 10 最差配置\n")
        f.write(f"{'='*120}\n\n")
        f.write(f"{'排名':<8}{'競爭者':<75}{'平均排名':<15}\n")
        f.write(f"{'-'*100}\n")
        for rank, (competitor, comp_data) in enumerate(sorted_competitors[-10:], len(sorted_competitors)-9):
            f.write(f"{rank:<8}{competitor:<75}{comp_data['avg_rank']:<15.2f}\n")
    
    print(f"  生成報告: {output_file}")
    return output_file

def generate_comparison_summary(output_folder):
    """生成兩種ratio的比較總結"""
    output_path = Path(output_folder)
    summary_file = output_path / "gnn_overall_ranking_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write("GNN增強模型整體排名總結\n")
        f.write(f"{'='*100}\n\n")
        f.write("說明：\n")
        f.write("- 比較兩種不同 Train/Val/Test 比例下的整體表現\n")
        f.write("- 整體排名：跨所有數據集計算，不區分數據集分類\n")
        f.write("- 只在45個GNN變體之間重新計算相對排名（1-45）\n\n")
        
        f.write(f"生成的報告檔案：\n")
        f.write(f"- gnn_overall_ranking_0.8-0.15-0.05.txt (大訓練集)\n")
        f.write(f"- gnn_overall_ranking_0.05-0.15-0.8.txt (小訓練集)\n\n")
        
        f.write(f"{'='*100}\n")
        f.write("分析要點\n")
        f.write(f"{'='*100}\n\n")
        f.write("1. 哪些GNN配置在所有數據集上表現最穩定？\n")
        f.write("2. 訓練集大小對GNN增強效果的影響\n")
        f.write("3. 最佳模型和GNN階段的組合\n")
        f.write("4. 不同模型對GNN插入階段的敏感度\n\n")
    
    print(f"\n生成總結檔案: {summary_file}")

def main():
    # 輸入和輸出路徑
    json_file = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/gnn_enhancement_all_models_all_models.json"
    output_folder = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis"
    
    print("="*100)
    print("GNN增強模型整體排名分析（跨所有數據集）")
    print("="*100)
    
    # 載入資料
    print("\n載入資料...")
    data = load_all_models_json(json_file)
    print(f"載入 {len(data)} 個模型的資料")
    
    # 為兩種 ratio 生成整體排名報告
    print("\n生成整體排名報告...")
    
    # 1. 大訓練集 (0.8/0.15/0.05)
    generate_overall_ranking(data, '0.8/0.15/0.05', output_folder)
    
    # 2. 小訓練集 (0.05/0.15/0.8)
    generate_overall_ranking(data, '0.05/0.15/0.8', output_folder)
    
    # 生成比較總結
    print("\n生成比較總結...")
    generate_comparison_summary(output_folder)
    
    print("\n" + "="*100)
    print("分析完成！")
    print("="*100)
    print("\n生成的檔案：")
    print("- gnn_overall_ranking_0.8-0.15-0.05.txt (大訓練集，整體排名)")
    print("- gnn_overall_ranking_0.05-0.15-0.8.txt (小訓練集，整體排名)")
    print("- gnn_overall_ranking_summary.txt (比較總結)")
    print("\n特點：")
    print("- 跨所有數據集的整體排名，不區分數據集分類")
    print("- 包含各模型和各GNN階段的統計分析")
    print("- 顯示Top 10和Bottom 10配置")

if __name__ == "__main__":
    main()
