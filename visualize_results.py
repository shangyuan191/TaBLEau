#!/usr/bin/env python3
"""
視覺化實驗結果腳本
基於 analyze_results.py 生成的 JSON 資料創建多種圖表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 設定輸出目錄
OUTPUT_DIR = Path(__file__).parent / "visualization_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 設定資料目錄
DATA_DIR = Path(__file__).parent / "analysis_results"

def load_json_data(ratio):
    """載入指定 split ratio 的 JSON 資料"""
    ratio_str = ratio.replace('/', '-')
    json_file = DATA_DIR / f"rankings_{ratio_str}.json"
    
    if not json_file.exists():
        print(f"找不到檔案: {json_file}")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_overall_rankings(data):
    """從 JSON 資料中提取總體排名"""
    # 從 average_ranks 中提取整體排名
    avg_ranks = data.get('average_ranks', {})
    
    # 轉換為列表並排序
    rankings = []
    for model, info in avg_ranks.items():
        if isinstance(info, dict):
            rankings.append({
                'model': model,
                'avg_rank': info.get('weighted_avg_rank', info.get('avg_rank', 0)),
                'dataset_count': info.get('total_datasets', info.get('count', 0))
            })
    
    # 按平均排名排序
    rankings.sort(key=lambda x: x['avg_rank'])
    return rankings

def get_classification_rankings(data):
    """從 JSON 資料中提取各分類的平均排名"""
    # 使用 ranked_by_classification 或 average_ranks
    if 'ranked_by_classification' in data:
        return data['ranked_by_classification']
    return data.get('average_ranks', {})

def plot_overall_comparison(data_small, data_large):
    """繪製兩種 split ratio 的總體排名比較圖"""
    rankings_small = get_overall_rankings(data_small)
    rankings_large = get_overall_rankings(data_large)
    
    # 準備資料
    models = [r['model'] for r in rankings_small]
    ranks_small = [r['avg_rank'] for r in rankings_small]
    
    # 確保 large 的順序與 small 一致
    ranks_large_dict = {r['model']: r['avg_rank'] for r in rankings_large}
    ranks_large = [ranks_large_dict[m] for m in models]
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ranks_small, width, label='0.05/0.15/0.8 (Small Train)', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, ranks_large, width, label='0.8/0.15/0.05 (Large Train)', 
                   color='coral', alpha=0.8)
    
    # 設定標籤
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Small vs Large Training Set\n(Lower rank = Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在條形上標註數值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "overall_comparison.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ 已生成: {output_file}")
    plt.close()

def plot_rank_change(data_small, data_large):
    """繪製模型排名變化圖（小訓練集 vs 大訓練集）"""
    rankings_small = get_overall_rankings(data_small)
    rankings_large = get_overall_rankings(data_large)
    
    # 建立排名字典
    rank_small_dict = {r['model']: i+1 for i, r in enumerate(rankings_small)}
    rank_large_dict = {r['model']: i+1 for i, r in enumerate(rankings_large)}
    
    # 計算排名變化
    models = list(rank_small_dict.keys())
    changes = []
    for model in models:
        change = rank_small_dict[model] - rank_large_dict[model]
        changes.append({
            'model': model,
            'change': change,
            'rank_small': rank_small_dict[model],
            'rank_large': rank_large_dict[model]
        })
    
    # 按變化排序
    changes.sort(key=lambda x: x['change'], reverse=True)
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models_sorted = [c['model'] for c in changes]
    change_values = [c['change'] for c in changes]
    
    colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in change_values]
    
    bars = ax.barh(models_sorted, change_values, color=colors, alpha=0.7)
    
    ax.set_xlabel('Rank Change (Positive = Improved with Large Train Set)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Rank Change: Small Train → Large Train Set\n(Green = Improved, Red = Degraded)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 標註具體排名
    for i, (bar, c) in enumerate(zip(bars, changes)):
        width = bar.get_width()
        label = f"{c['rank_small']}→{c['rank_large']}"
        x_pos = width + (0.3 if width > 0 else -0.3)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
               ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "rank_change.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ 已生成: {output_file}")
    plt.close()

def plot_classification_heatmap(data, ratio_name):
    """繪製各分類的模型排名熱力圖"""
    classifications = get_classification_rankings(data)
    
    if not classifications:
        print(f"警告: 找不到分類排名資料 ({ratio_name})")
        return
    
    # 提取所有模型
    all_models = set()
    for class_data in classifications.values():
        all_models.update(class_data.keys())
    all_models = sorted(all_models)
    
    # 提取分類名稱
    class_names = sorted(classifications.keys())
    
    # 建立矩陣
    matrix = []
    for class_name in class_names:
        row = []
        for model in all_models:
            if model in classifications[class_name]:
                rank = classifications[class_name][model]['avg_rank']
                row.append(rank)
            else:
                row.append(np.nan)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=14)
    
    # 設定刻度
    ax.set_xticks(np.arange(len(all_models)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    
    # 在每個格子中標註數值
    for i in range(len(class_names)):
        for j in range(len(all_models)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title(f'Average Model Ranking by Dataset Classification\nSplit Ratio: {ratio_name}\n(Lower rank = Better performance)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Rank', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"classification_heatmap_{ratio_name.replace('/', '-')}.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ 已生成: {output_file}")
    plt.close()

def plot_top_models_by_classification(data, ratio_name, top_n=5):
    """繪製各分類的 Top N 模型"""
    classifications = get_classification_rankings(data)
    
    if not classifications:
        print(f"警告: 找不到分類排名資料 ({ratio_name})")
        return
    
    class_names = sorted(classifications.keys())
    n_classes = len(class_names)
    
    # 設定子圖佈局
    n_cols = 2
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        
        # 提取並排序該分類的模型
        models_data = classifications[class_name]
        sorted_models = sorted(models_data.items(), key=lambda x: x[1]['avg_rank'])[:top_n]
        
        models = [m[0] for m in sorted_models]
        ranks = [m[1]['avg_rank'] for m in sorted_models]
        
        # 繪製橫向條形圖
        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, ranks, color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Average Rank', fontsize=10)
        ax.set_title(f'{class_name}\n(Top {top_n} Models)', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 標註數值
        for bar, rank in zip(bars, ranks):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{rank:.2f}', ha='left', va='center', fontsize=9)
    
    # 隱藏多餘的子圖
    for idx in range(n_classes, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Top {top_n} Models by Dataset Classification\nSplit Ratio: {ratio_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"top_models_by_classification_{ratio_name.replace('/', '-')}.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ 已生成: {output_file}")
    plt.close()

def plot_model_stability(data_small, data_large):
    """繪製模型穩定性分析圖（標準差分析）"""
    # 從兩個 split ratio 計算每個模型的排名標準差
    classifications_small = get_classification_rankings(data_small)
    classifications_large = get_classification_rankings(data_large)
    
    # 收集每個模型在各分類中的排名
    model_ranks = defaultdict(lambda: {'small': [], 'large': []})
    
    for class_name, models in classifications_small.items():
        for model, info in models.items():
            model_ranks[model]['small'].append(info['avg_rank'])
    
    for class_name, models in classifications_large.items():
        for model, info in models.items():
            model_ranks[model]['large'].append(info['avg_rank'])
    
    # 計算統計量
    stability_data = []
    for model, ranks in model_ranks.items():
        all_ranks = ranks['small'] + ranks['large']
        if all_ranks:
            stability_data.append({
                'model': model,
                'std': np.std(all_ranks),
                'mean': np.mean(all_ranks),
                'min': np.min(all_ranks),
                'max': np.max(all_ranks)
            })
    
    stability_data.sort(key=lambda x: x['std'])
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 左圖：標準差
    models = [d['model'] for d in stability_data]
    stds = [d['std'] for d in stability_data]
    
    bars = ax1.barh(models, stds, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Standard Deviation of Ranks', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax1.set_title('Model Stability Across Classifications\n(Lower = More Consistent)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    for bar, std in zip(bars, stds):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{std:.2f}', ha='left', va='center', fontsize=9)
    
    # 右圖：範圍圖（min-max）
    means = [d['mean'] for d in stability_data]
    mins = [d['min'] for d in stability_data]
    maxs = [d['max'] for d in stability_data]
    
    y_pos = np.arange(len(models))
    
    # 繪製範圍線
    for i, (model, mean, min_val, max_val) in enumerate(zip(models, means, mins, maxs)):
        ax2.plot([min_val, max_val], [i, i], 'o-', linewidth=2, markersize=6, 
                color='steelblue', alpha=0.6)
        ax2.plot(mean, i, 'D', markersize=8, color='coral', alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel('Average Rank', fontsize=12, fontweight='bold')
    ax2.set_title('Rank Range Across Classifications\n(Diamond = Mean, Line = Min-Max)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "model_stability.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ 已生成: {output_file}")
    plt.close()

def plot_radar_chart(data_small, data_large, top_n=6):
    """繪製 Top N 模型的雷達圖（跨不同分類的表現）"""
    # 獲取整體排名 Top N
    rankings_small = get_overall_rankings(data_small)
    top_models = [r['model'] for r in rankings_small[:top_n]]
    
    classifications_small = get_classification_rankings(data_small)
    classifications_large = get_classification_rankings(data_large)
    
    class_names = sorted(classifications_small.keys())
    
    # 設定雷達圖
    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]  # 閉合
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for idx, model in enumerate(top_models):
        # Small train data
        values_small = []
        for class_name in class_names:
            if model in classifications_small[class_name]:
                values_small.append(classifications_small[class_name][model]['avg_rank'])
            else:
                values_small.append(14)  # 最差排名
        values_small += values_small[:1]
        
        ax1.plot(angles, values_small, 'o-', linewidth=2, label=model, color=colors[idx])
        ax1.fill(angles, values_small, alpha=0.15, color=colors[idx])
        
        # Large train data
        values_large = []
        for class_name in class_names:
            if model in classifications_large[class_name]:
                values_large.append(classifications_large[class_name][model]['avg_rank'])
            else:
                values_large.append(14)
        values_large += values_large[:1]
        
        ax2.plot(angles, values_large, 'o-', linewidth=2, label=model, color=colors[idx])
        ax2.fill(angles, values_large, alpha=0.15, color=colors[idx])
    
    # 設定標籤
    class_labels = [name.replace('_', '\n') for name in class_names]
    class_labels += class_labels[:1]
    
    for ax, title in zip([ax1, ax2], ['0.05/0.15/0.8 (Small Train)', '0.8/0.15/0.05 (Large Train)']):
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(class_labels[:-1], fontsize=8)
        ax.set_ylim(0, 14)
        ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
        ax.set_yticklabels(['2', '4', '6', '8', '10', '12', '14'], fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.invert_yaxis()  # 較小的值在外圍（更好）
    
    fig.suptitle(f'Top {top_n} Models Performance Across Classifications\n(Closer to center = Better)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "radar_chart_top_models.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ 已生成: {output_file}")
    plt.close()

def main():
    print("=" * 80)
    print("開始生成視覺化圖表...")
    print("=" * 80)
    print()
    
    # 載入資料
    print("載入 JSON 資料...")
    data_small = load_json_data('0.05/0.15/0.8')
    data_large = load_json_data('0.8/0.15/0.05')
    
    if not data_small or not data_large:
        print("錯誤: 無法載入資料檔案")
        return
    
    print("✓ 資料載入完成\n")
    
    # 生成各種圖表
    print("生成圖表:")
    print("-" * 80)
    
    print("\n1. 總體排名比較圖...")
    plot_overall_comparison(data_small, data_large)
    
    print("\n2. 排名變化圖...")
    plot_rank_change(data_small, data_large)
    
    print("\n3. 分類熱力圖 (Small Train)...")
    plot_classification_heatmap(data_small, '0.05/0.15/0.8')
    
    print("\n4. 分類熱力圖 (Large Train)...")
    plot_classification_heatmap(data_large, '0.8/0.15/0.05')
    
    print("\n5. Top 模型分類圖 (Small Train)...")
    plot_top_models_by_classification(data_small, '0.05/0.15/0.8', top_n=5)
    
    print("\n6. Top 模型分類圖 (Large Train)...")
    plot_top_models_by_classification(data_large, '0.8/0.15/0.05', top_n=5)
    
    print("\n7. 模型穩定性分析圖...")
    plot_model_stability(data_small, data_large)
    
    print("\n8. 雷達圖 (Top 6 Models)...")
    plot_radar_chart(data_small, data_large, top_n=6)
    
    print("\n" + "=" * 80)
    print(f"✓ 所有圖表已生成完成！")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
