#!/usr/bin/env python3
"""
視覺化GNN增強效果分析結果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 可以插入GNN的模型
GNN_INSERTABLE_MODELS = [
    'excelformer', 'fttransformer', 'resnet', 'tabnet', 
    'tabtransformer', 'trompt', 'vime', 'scarf', 'subtab'
]

# GNN插入階段
GNN_STAGES = ['none', 'start', 'materialize', 'encoding', 'columnwise', 'decoding']

def load_analysis_data(json_file):
    """載入分析結果JSON"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_gnn_enhancement_heatmap(data, output_folder):
    """
    為每個模型繪製熱力圖：顯示不同GNN配置在各分類下的平均排名
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    for model in GNN_INSERTABLE_MODELS:
        if model not in data:
            continue
        
        model_data = data[model]
        
        # 準備資料
        classifications = sorted(model_data.keys())
        
        # 定義配置順序
        configs = [
            f"{model}(ratio=0.8/0.15/0.05, gnn_stage=none)",  # baseline
            f"{model}(ratio=0.05/0.15/0.8, gnn_stage=none)",
            f"{model}(ratio=0.05/0.15/0.8, gnn_stage=start)",
            f"{model}(ratio=0.05/0.15/0.8, gnn_stage=materialize)",
            f"{model}(ratio=0.05/0.15/0.8, gnn_stage=encoding)",
            f"{model}(ratio=0.05/0.15/0.8, gnn_stage=columnwise)",
            f"{model}(ratio=0.05/0.15/0.8, gnn_stage=decoding)",
        ]
        
        # 簡化標籤
        config_labels = [
            "Baseline\n(0.8 train)",
            "Small train\n(none)",
            "Small train\n(start)",
            "Small train\n(materialize)",
            "Small train\n(encoding)",
            "Small train\n(columnwise)",
            "Small train\n(decoding)"
        ]
        
        # 建立熱力圖矩陣
        heatmap_data = []
        valid_classifications = []
        
        for classification in classifications:
            class_data = model_data[classification]
            rankings = class_data['rankings']
            
            row = []
            has_data = False
            for config in configs:
                if config in rankings:
                    row.append(rankings[config]['avg_rank'])
                    has_data = True
                else:
                    row.append(np.nan)
            
            if has_data:
                heatmap_data.append(row)
                valid_classifications.append(classification)
        
        if not heatmap_data:
            continue
        
        # 繪製熱力圖
        fig, ax = plt.subplots(figsize=(14, max(8, len(valid_classifications) * 0.5)))
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=17)
        
        # 設定軸標籤
        ax.set_xticks(np.arange(len(config_labels)))
        ax.set_yticks(np.arange(len(valid_classifications)))
        ax.set_xticklabels(config_labels, fontsize=9)
        ax.set_yticklabels(valid_classifications, fontsize=8)
        
        # 旋轉x軸標籤
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加數值標註
        for i in range(len(valid_classifications)):
            for j in range(len(config_labels)):
                if not np.isnan(heatmap_array[i, j]):
                    text = ax.text(j, i, f'{heatmap_array[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=7)
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Rank (lower is better)', rotation=270, labelpad=20)
        
        ax.set_title(f'GNN Enhancement Effect - {model.upper()}\nAverage Rank across Dataset Classifications', 
                    fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{model}_gnn_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"生成熱力圖: {model}_gnn_heatmap.png")

def plot_improvement_comparison(data, output_folder):
    """
    比較各模型在不同分類下，GNN增強相對於baseline的改善程度
    """
    output_path = Path(output_folder)
    
    # 收集改善資料
    improvement_data = {}
    
    for model in GNN_INSERTABLE_MODELS:
        if model not in data:
            continue
        
        model_data = data[model]
        improvements = []
        classifications = []
        
        for classification, class_data in model_data.items():
            rankings = class_data['rankings']
            
            baseline_key = f"{model}(ratio=0.8/0.15/0.05, gnn_stage=none)"
            if baseline_key not in rankings:
                continue
            
            baseline_rank = rankings[baseline_key]['avg_rank']
            
            # 找出小訓練集配置中最好的
            best_small_rank = baseline_rank
            for config, rank_info in rankings.items():
                if 'ratio=0.05/0.15/0.8' in config:
                    if rank_info['avg_rank'] < best_small_rank:
                        best_small_rank = rank_info['avg_rank']
            
            improvement = baseline_rank - best_small_rank
            improvements.append(improvement)
            classifications.append(classification)
        
        if improvements:
            improvement_data[model] = {
                'improvements': improvements,
                'classifications': classifications,
                'avg_improvement': np.mean(improvements)
            }
    
    # 繪製改善比較圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 圖1：各模型的平均改善
    models = sorted(improvement_data.keys(), key=lambda x: improvement_data[x]['avg_improvement'], reverse=True)
    avg_improvements = [improvement_data[m]['avg_improvement'] for m in models]
    
    colors = ['green' if x > 0 else 'red' for x in avg_improvements]
    bars = ax1.barh(models, avg_improvements, color=colors, alpha=0.7)
    
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Average Rank Improvement (positive = better with GNN)', fontsize=11)
    ax1.set_title('Average GNN Enhancement Effect Across All Classifications', fontsize=13, pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加數值標註
    for i, (model, improvement) in enumerate(zip(models, avg_improvements)):
        ax1.text(improvement, i, f'{improvement:+.2f}', 
                va='center', ha='left' if improvement >= 0 else 'right', fontsize=9)
    
    # 圖2：分類別的改善分布（箱型圖）
    improvement_by_classification = {}
    for model, model_info in improvement_data.items():
        for classification, improvement in zip(model_info['classifications'], model_info['improvements']):
            if classification not in improvement_by_classification:
                improvement_by_classification[classification] = []
            improvement_by_classification[classification].append(improvement)
    
    sorted_classifications = sorted(improvement_by_classification.keys(), 
                                   key=lambda x: np.mean(improvement_by_classification[x]), 
                                   reverse=True)
    
    improvement_lists = [improvement_by_classification[c] for c in sorted_classifications]
    
    bp = ax2.boxplot(improvement_lists, labels=sorted_classifications, vert=False, patch_artist=True)
    
    # 為箱型圖上色
    for patch, values in zip(bp['boxes'], improvement_lists):
        if np.mean(values) > 0:
            patch.set_facecolor('lightgreen')
        else:
            patch.set_facecolor('lightcoral')
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Rank Improvement (positive = better with GNN)', fontsize=11)
    ax2.set_title('GNN Enhancement Effect by Dataset Classification', fontsize=13, pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'gnn_improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("生成改善比較圖: gnn_improvement_comparison.png")

def plot_stage_effectiveness(data, output_folder):
    """
    分析哪個GNN插入階段最有效
    """
    output_path = Path(output_folder)
    
    # 收集每個階段的改善資料
    stage_improvements = {stage: [] for stage in GNN_STAGES[1:]}  # 排除'none'
    
    for model in GNN_INSERTABLE_MODELS:
        if model not in data:
            continue
        
        model_data = data[model]
        
        for classification, class_data in model_data.items():
            rankings = class_data['rankings']
            
            baseline_key = f"{model}(ratio=0.8/0.15/0.05, gnn_stage=none)"
            if baseline_key not in rankings:
                continue
            
            baseline_rank = rankings[baseline_key]['avg_rank']
            
            # 比較各階段
            for stage in GNN_STAGES[1:]:
                stage_key = f"{model}(ratio=0.05/0.15/0.8, gnn_stage={stage})"
                if stage_key in rankings:
                    improvement = baseline_rank - rankings[stage_key]['avg_rank']
                    stage_improvements[stage].append(improvement)
    
    # 繪製
    fig, ax = plt.subplots(figsize=(12, 7))
    
    stages = list(stage_improvements.keys())
    bp = ax.boxplot([stage_improvements[s] for s in stages], 
                    labels=stages, patch_artist=True)
    
    # 上色
    for patch, stage in zip(bp['boxes'], stages):
        values = stage_improvements[stage]
        if np.mean(values) > 0:
            patch.set_facecolor('lightgreen')
        else:
            patch.set_facecolor('lightcoral')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Rank Improvement (positive = better with GNN)', fontsize=11)
    ax.set_xlabel('GNN Insertion Stage', fontsize=11)
    ax.set_title('Effectiveness of Different GNN Insertion Stages\nAcross All Models and Classifications', 
                fontsize=13, pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加平均值標註
    for i, stage in enumerate(stages, 1):
        mean_val = np.mean(stage_improvements[stage])
        ax.text(i, mean_val, f'{mean_val:+.2f}', 
               ha='center', va='bottom' if mean_val > 0 else 'top', 
               fontsize=10, fontweight='bold', color='darkblue')
    
    plt.tight_layout()
    plt.savefig(output_path / 'gnn_stage_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("生成階段有效性圖: gnn_stage_effectiveness.png")

def plot_task_type_analysis(data, output_folder):
    """
    分析GNN增強在不同任務類型（binclass, multiclass, regression）下的效果
    """
    output_path = Path(output_folder)
    
    # 按任務類型分組
    task_improvements = {
        'binclass': [],
        'multiclass': [],
        'regression': []
    }
    
    for model in GNN_INSERTABLE_MODELS:
        if model not in data:
            continue
        
        model_data = data[model]
        
        for classification, class_data in model_data.items():
            # 從分類名稱中提取任務類型
            if 'binclass' in classification:
                task_type = 'binclass'
            elif 'multiclass' in classification:
                task_type = 'multiclass'
            elif 'regression' in classification:
                task_type = 'regression'
            else:
                continue
            
            rankings = class_data['rankings']
            baseline_key = f"{model}(ratio=0.8/0.15/0.05, gnn_stage=none)"
            
            if baseline_key not in rankings:
                continue
            
            baseline_rank = rankings[baseline_key]['avg_rank']
            
            # 找出最佳小訓練集配置
            best_small_rank = baseline_rank
            for config, rank_info in rankings.items():
                if 'ratio=0.05/0.15/0.8' in config:
                    if rank_info['avg_rank'] < best_small_rank:
                        best_small_rank = rank_info['avg_rank']
            
            improvement = baseline_rank - best_small_rank
            task_improvements[task_type].append(improvement)
    
    # 繪製
    fig, ax = plt.subplots(figsize=(10, 7))
    
    task_types = ['binclass', 'multiclass', 'regression']
    task_labels = ['Binary Classification', 'Multi-class Classification', 'Regression']
    
    bp = ax.boxplot([task_improvements[t] for t in task_types], 
                    labels=task_labels, patch_artist=True)
    
    # 上色
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Rank Improvement (positive = better with GNN)', fontsize=11)
    ax.set_xlabel('Task Type', fontsize=11)
    ax.set_title('GNN Enhancement Effect by Task Type', fontsize=13, pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加統計資訊
    for i, task_type in enumerate(task_types, 1):
        values = task_improvements[task_type]
        mean_val = np.mean(values)
        median_val = np.median(values)
        
        ax.text(i, mean_val, f'μ={mean_val:+.2f}', 
               ha='center', va='bottom' if mean_val > 0 else 'top', 
               fontsize=9, fontweight='bold', color='red')
        
        ax.text(i, median_val, f'M={median_val:+.2f}', 
               ha='center', va='top' if median_val > 0 else 'bottom', 
               fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig(output_path / 'gnn_task_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("生成任務類型分析圖: gnn_task_type_analysis.png")

def main():
    # 載入資料
    json_file = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/gnn_enhancement_all_models.json"
    output_folder = "/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_visualizations"
    
    print("載入分析資料...")
    data = load_analysis_data(json_file)
    
    print(f"\n找到 {len(data)} 個模型的資料")
    
    # 生成各種視覺化
    print("\n生成視覺化...")
    
    print("\n1. 為每個模型生成熱力圖...")
    plot_gnn_enhancement_heatmap(data, output_folder)
    
    print("\n2. 生成改善比較圖...")
    plot_improvement_comparison(data, output_folder)
    
    print("\n3. 生成階段有效性分析...")
    plot_stage_effectiveness(data, output_folder)
    
    print("\n4. 生成任務類型分析...")
    plot_task_type_analysis(data, output_folder)
    
    print(f"\n所有視覺化已儲存至: {output_folder}")
    print("\n完成！")

if __name__ == "__main__":
    main()
