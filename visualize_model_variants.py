#!/usr/bin/env python3
"""
基於模型變體分析結果創建視覺化圖表
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict

# 設定中文字體
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 設定圖表樣式
plt.style.use('seaborn-v0_8-darkgrid')


def load_variant_data(json_file):
    """載入變體分析的 JSON 資料"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_1_model_variants_heatmap(data, ratio, output_dir):
    """
    圖表1: 每個模型的變體在不同分類下的平均排名熱圖
    """
    by_class = data['by_classification']
    
    # 收集所有模型和變體
    all_models = set()
    all_variants = {}  # model -> [variants]
    
    for classification, models in by_class.items():
        for model, variants in models.items():
            all_models.add(model)
            if model not in all_variants:
                all_variants[model] = set()
            all_variants[model].update(variants.keys())
    
    # 對每個有GNN變體的模型繪製熱圖
    models_with_gnn = {m: v for m, v in all_variants.items() if len(v) > 1}
    
    if not models_with_gnn:
        print("沒有找到有GNN變體的模型")
        return
    
    # 計算子圖網格
    n_models = len(models_with_gnn)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    # 分類簡稱
    class_short_names = {
        'small_datasets+binclass+numerical': 'S-Bin-Num',
        'small_datasets+binclass+categorical': 'S-Bin-Cat',
        'small_datasets+binclass+balanced': 'S-Bin-Bal',
        'small_datasets+multiclass+numerical': 'S-Mul-Num',
        'small_datasets+multiclass+categorical': 'S-Mul-Cat',
        'small_datasets+multiclass+balanced': 'S-Mul-Bal',
        'small_datasets+regression+numerical': 'S-Reg-Num',
        'small_datasets+regression+balanced': 'S-Reg-Bal',
        'large_datasets+binclass+numerical': 'L-Bin-Num',
        'large_datasets+regression+numerical': 'L-Reg-Num',
    }
    
    classifications = sorted(by_class.keys())
    class_labels = [class_short_names.get(c, c) for c in classifications]
    
    for idx, (model, variants) in enumerate(sorted(models_with_gnn.items())):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 組織變體順序：base, start, materialize, encoding, columnwise, decoding
        variant_order = []
        variant_labels = []
        
        for v in variants:
            if '(base)' in v:
                variant_order.insert(0, v)
                variant_labels.insert(0, 'Base')
            elif '(GNN@start)' in v:
                variant_order.append(v)
                variant_labels.append('GNN@Start')
            elif '(GNN@materialize)' in v:
                variant_order.append(v)
                variant_labels.append('GNN@Materialize')
            elif '(GNN@encoding)' in v:
                variant_order.append(v)
                variant_labels.append('GNN@Encoding')
            elif '(GNN@columnwise)' in v:
                variant_order.append(v)
                variant_labels.append('GNN@Columnwise')
            elif '(GNN@decoding)' in v:
                variant_order.append(v)
                variant_labels.append('GNN@Decoding')
        
        # 構建熱圖矩陣
        matrix = []
        for variant in variant_order:
            row_data = []
            for classification in classifications:
                if model in by_class[classification] and variant in by_class[classification][model]:
                    avg_rank = by_class[classification][model][variant]['avg_rank']
                    row_data.append(avg_rank)
                else:
                    row_data.append(np.nan)
            matrix.append(row_data)
        
        matrix = np.array(matrix)
        
        # 繪製熱圖
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=60)
        
        # 設定標籤
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(variant_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(variant_labels, fontsize=9)
        
        # 添加數值標註
        for i in range(len(variant_labels)):
            for j in range(len(class_labels)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=7)
        
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
        
        # 添加顏色條
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隱藏多餘的子圖
    for idx in range(len(models_with_gnn), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Model Variants Average Rank by Dataset Classification\nSplit Ratio: {ratio}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'variants_heatmap_{ratio.replace("/", "-")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已生成: {output_file}")
    plt.close()


def plot_2_best_variants_comparison(data_05, data_08, output_dir):
    """
    圖表2: 比較兩種split ratio下，每個模型最佳變體的表現
    """
    ratios = ['0.05/0.15/0.8', '0.8/0.15/0.05']
    data_list = [data_05, data_08]
    
    # 收集每個模型的最佳變體
    model_best_variants = defaultdict(dict)
    
    for ratio, data in zip(ratios, data_list):
        by_class = data['by_classification']
        
        # 對每個模型，計算整體平均排名
        model_variants_avg = defaultdict(lambda: defaultdict(list))
        
        for classification, models in by_class.items():
            for model, variants in models.items():
                for variant, stats in variants.items():
                    # 加權平均
                    for _ in range(stats['count']):
                        model_variants_avg[model][variant].append(stats['avg_rank'])
        
        # 找出每個模型的最佳變體
        for model, variants in model_variants_avg.items():
            best_variant = min(variants.items(), key=lambda x: np.mean(x[1]))
            model_best_variants[model][ratio] = {
                'variant': best_variant[0],
                'avg_rank': np.mean(best_variant[1])
            }
    
    # 繪圖
    models = sorted(model_best_variants.keys())
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ranks_05 = [model_best_variants[m].get('0.05/0.15/0.8', {}).get('avg_rank', 0) for m in models]
    ranks_08 = [model_best_variants[m].get('0.8/0.15/0.05', {}).get('avg_rank', 0) for m in models]
    
    bars1 = ax.bar(x - width/2, ranks_05, width, label='Train:0.05 (Small)', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, ranks_08, width, label='Train:0.8 (Large)', color='salmon', edgecolor='black')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rank (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Best Variant Performance Comparison Across Split Ratios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加數值標註
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'best_variants_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已生成: {output_file}")
    plt.close()


def plot_3_gnn_stage_effectiveness(data, ratio, output_dir):
    """
    圖表3: GNN插入階段的有效性分析
    分析哪個GNN插入階段平均表現最好
    """
    by_class = data['by_classification']
    
    # 收集各GNN階段的排名
    stage_ranks = defaultdict(list)
    
    for classification, models in by_class.items():
        for model, variants in models.items():
            for variant, stats in variants.items():
                if 'GNN@' in variant:
                    stage = variant.split('GNN@')[1].rstrip(')')
                    # 重複添加以反映數量
                    for _ in range(stats['count']):
                        stage_ranks[stage].append(stats['avg_rank'])
    
    if not stage_ranks:
        print(f"沒有找到GNN變體資料 (ratio: {ratio})")
        return
    
    # 計算統計資料
    stages = sorted(stage_ranks.keys())
    means = [np.mean(stage_ranks[s]) for s in stages]
    stds = [np.std(stage_ranks[s]) for s in stages]
    
    # 繪製箱形圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子圖1: 箱形圖
    bp = ax1.boxplot([stage_ranks[s] for s in stages], labels=stages, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
    
    ax1.set_xlabel('GNN Insertion Stage', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
    ax1.set_title(f'GNN Stage Effectiveness Distribution\nSplit Ratio: {ratio}', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    
    # 子圖2: 平均值柱狀圖
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(stages)))
    bars = ax2.bar(stages, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    
    ax2.set_xlabel('GNN Insertion Stage', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Rank (± Std)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Mean Performance by GNN Stage\nSplit Ratio: {ratio}', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=15)
    
    # 添加數值標註
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, mean,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / f'gnn_stage_effectiveness_{ratio.replace("/", "-")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已生成: {output_file}")
    plt.close()


def plot_4_model_gnn_improvement(data, ratio, output_dir):
    """
    圖表4: 每個模型從基礎版本到最佳GNN變體的改善幅度
    """
    by_class = data['by_classification']
    
    # 收集每個模型的base和GNN變體排名
    model_comparison = defaultdict(lambda: {'base': [], 'best_gnn': [], 'best_gnn_name': ''})
    
    for classification, models in by_class.items():
        for model, variants in models.items():
            base_variant = None
            gnn_variants = {}
            
            for variant, stats in variants.items():
                if '(base)' in variant:
                    base_variant = stats
                elif 'GNN@' in variant:
                    gnn_variants[variant] = stats
            
            if base_variant and gnn_variants:
                # 加權添加base排名
                for _ in range(base_variant['count']):
                    model_comparison[model]['base'].append(base_variant['avg_rank'])
                
                # 找最佳GNN變體
                best_gnn = min(gnn_variants.items(), key=lambda x: x[1]['avg_rank'])
                for _ in range(best_gnn[1]['count']):
                    model_comparison[model]['best_gnn'].append(best_gnn[1]['avg_rank'])
                
                if not model_comparison[model]['best_gnn_name']:
                    model_comparison[model]['best_gnn_name'] = best_gnn[0].split('GNN@')[1].rstrip(')')
    
    # 計算改善幅度
    models = []
    base_ranks = []
    best_gnn_ranks = []
    improvements = []
    gnn_names = []
    
    for model, data_dict in sorted(model_comparison.items()):
        if data_dict['base'] and data_dict['best_gnn']:
            base_avg = np.mean(data_dict['base'])
            gnn_avg = np.mean(data_dict['best_gnn'])
            improvement = base_avg - gnn_avg  # 正值表示改善
            
            models.append(model)
            base_ranks.append(base_avg)
            best_gnn_ranks.append(gnn_avg)
            improvements.append(improvement)
            gnn_names.append(data_dict['best_gnn_name'])
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 子圖1: Base vs Best GNN
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, base_ranks, width, label='Base Model', color='lightcoral', edgecolor='black')
    bars2 = ax1.bar(x + width/2, best_gnn_ranks, width, label='Best GNN Variant', color='lightgreen', edgecolor='black')
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
    ax1.set_title(f'Base Model vs Best GNN Variant\nSplit Ratio: {ratio}', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加GNN階段標註
    for i, (bar, gnn_name) in enumerate(zip(bars2, gnn_names)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                gnn_name[:3], ha='center', va='bottom', fontsize=7, rotation=0)
    
    # 子圖2: 改善幅度
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(models, improvements, color=colors, edgecolor='black', alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (Base Rank - GNN Rank)', fontsize=12, fontweight='bold')
    ax2.set_title(f'GNN Improvement over Base Model\nPositive = GNN Better, Split Ratio: {ratio}',
                 fontsize=13, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加數值標註
    for bar, imp, gnn_name in zip(bars, improvements, gnn_names):
        y_pos = imp + (0.5 if imp > 0 else -0.5)
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{imp:.1f}\n({gnn_name[:3]})', ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / f'gnn_improvement_{ratio.replace("/", "-")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已生成: {output_file}")
    plt.close()


def plot_5_classification_performance_matrix(data, ratio, output_dir):
    """
    圖表5: 各模型在不同資料集分類下的最佳變體熱圖
    """
    by_class = data['by_classification']
    
    # 收集資料
    classifications = sorted(by_class.keys())
    all_models = set()
    for models in by_class.values():
        all_models.update(models.keys())
    models = sorted(all_models)
    
    # 構建矩陣：每個模型在每個分類下的最佳變體排名
    matrix = np.zeros((len(models), len(classifications)))
    best_variants = [['' for _ in classifications] for _ in models]
    
    for j, classification in enumerate(classifications):
        for i, model in enumerate(models):
            if model in by_class[classification]:
                variants = by_class[classification][model]
                if variants:
                    best = min(variants.items(), key=lambda x: x[1]['avg_rank'])
                    matrix[i, j] = best[1]['avg_rank']
                    # 提取變體簡稱
                    if '(base)' in best[0]:
                        best_variants[i][j] = 'B'
                    elif 'GNN@' in best[0]:
                        stage = best[0].split('GNN@')[1].rstrip(')')
                        best_variants[i][j] = stage[0].upper()  # 首字母
                else:
                    matrix[i, j] = np.nan
            else:
                matrix[i, j] = np.nan
    
    # 繪製熱圖
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 分類簡稱
    class_short = []
    for c in classifications:
        parts = c.split('+')
        short = f"{parts[0][0].upper()}-{parts[1][:3].title()}-{parts[2][:3].title()}"
        class_short.append(short)
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=50)
    
    ax.set_xticks(np.arange(len(class_short)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(class_short, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(models, fontsize=10)
    
    # 添加數值和變體標註
    for i in range(len(models)):
        for j in range(len(classifications)):
            if not np.isnan(matrix[i, j]):
                text = f'{matrix[i, j]:.1f}\n[{best_variants[i][j]}]'
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title(f'Best Variant Performance by Model and Classification\nSplit Ratio: {ratio}\n[B=Base, S=Start, M=Materialize, E=Encoding, C=Columnwise, D=Decoding]',
                fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Average Rank')
    plt.tight_layout()
    
    output_file = output_dir / f'classification_matrix_{ratio.replace("/", "-")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已生成: {output_file}")
    plt.close()


def main():
    # 設定路徑
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'variant_analysis'
    output_dir = base_dir / 'variant_visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("開始生成模型變體視覺化圖表")
    print("="*80)
    
    # 載入資料
    print("\n載入資料...")
    json_05 = data_dir / 'model_variants_0.05-0.15-0.8.json'
    json_08 = data_dir / 'model_variants_0.8-0.15-0.05.json'
    
    data_05 = load_variant_data(json_05)
    data_08 = load_variant_data(json_08)
    
    print(f"已載入 Split Ratio 0.05/0.15/0.8 的資料")
    print(f"已載入 Split Ratio 0.8/0.15/0.05 的資料")
    
    # 生成圖表
    print("\n" + "="*80)
    print("生成圖表...")
    print("="*80)
    
    print("\n圖表 1: 模型變體排名熱圖 (Split Ratio: 0.05/0.15/0.8)")
    plot_1_model_variants_heatmap(data_05, '0.05/0.15/0.8', output_dir)
    
    print("\n圖表 1: 模型變體排名熱圖 (Split Ratio: 0.8/0.15/0.05)")
    plot_1_model_variants_heatmap(data_08, '0.8/0.15/0.05', output_dir)
    
    print("\n圖表 2: 最佳變體表現比較")
    plot_2_best_variants_comparison(data_05, data_08, output_dir)
    
    print("\n圖表 3: GNN插入階段有效性分析 (Split Ratio: 0.05/0.15/0.8)")
    plot_3_gnn_stage_effectiveness(data_05, '0.05/0.15/0.8', output_dir)
    
    print("\n圖表 3: GNN插入階段有效性分析 (Split Ratio: 0.8/0.15/0.05)")
    plot_3_gnn_stage_effectiveness(data_08, '0.8/0.15/0.05', output_dir)
    
    print("\n圖表 4: GNN改善幅度分析 (Split Ratio: 0.05/0.15/0.8)")
    plot_4_model_gnn_improvement(data_05, '0.05/0.15/0.8', output_dir)
    
    print("\n圖表 4: GNN改善幅度分析 (Split Ratio: 0.8/0.15/0.05)")
    plot_4_model_gnn_improvement(data_08, '0.8/0.15/0.05', output_dir)
    
    print("\n圖表 5: 分類表現矩陣 (Split Ratio: 0.05/0.15/0.8)")
    plot_5_classification_performance_matrix(data_05, '0.05/0.15/0.8', output_dir)
    
    print("\n圖表 5: 分類表現矩陣 (Split Ratio: 0.8/0.15/0.05)")
    plot_5_classification_performance_matrix(data_08, '0.8/0.15/0.05', output_dir)
    
    print("\n" + "="*80)
    print("所有視覺化圖表已生成完成！")
    print(f"圖表保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
