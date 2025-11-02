#!/usr/bin/env python3
import csv
from pathlib import Path

base = Path(__file__).resolve().parents[1] / 'analysis' / 'regression_target_distributions_analysis'
summary = base / 'target_distribution_summary.csv'
md = base / 'primary_label_table.md'

# Read CSV
rows = []
with open(summary, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# helper
import math

def primary_from_labels(labels):
    return labels.split(';')[0]


def reason_for_row(r):
    # use simple logic matching our priority
    z = float(r['zero_prop'])
    unique_ratio = float(r['unique_ratio'])
    peaks = int(r['peaks']) if r['peaks'] else 0
    try:
        skew = float(r['skewness'])
    except:
        skew = 0.0
    try:
        kurt = float(r['kurtosis'])
    except:
        kurt = 0.0
    if unique_ratio <= 0.01:
        return 'constant (unique_ratio ≤ 0.01)'
    if z >= 0.20:
        return f'zero_inflated (zero_prop = {z})'
    # discrete
    if unique_ratio <= 0.05:
        return f'discrete (unique_ratio = {unique_ratio})'
    if peaks >= 2:
        return f'multimodal (peaks = {peaks})'
    if kurt > 5:
        return f'heavy_tailed (kurtosis = {kurt:.2f})'
    if abs(skew) >= 1.5:
        return f'highly_skewed (skewness = {skew:.2f})'
    if abs(skew) >= 0.5:
        return f'moderately_skewed (skewness = {skew:.2f})'
    return 'approx_normal (no strong skew/multi/tail)'

# Build appended content
lines = []
lines.append('\n## 類別定義與判定依據（自動化說明）\n')
lines.append('以下為各 primary_label 的中文定義、啟發式判定條件以及建議的前處理或注意事項。')
lines.append('\n- 檢查優先順序（first-match）：constant → zero_inflated → discrete → bounded_0_1 → multimodal → heavy_tailed → highly_skewed → moderately_skewed → approx_normal。')
lines.append('\n- 使用的關鍵統計：unique_ratio, zero_prop, peaks (KDE 峰數), skewness, kurtosis。\n')

# short definitions (Chinese) - reuse earlier text but concise
defs = {
    'approx_normal':'近似常態：單峰且偏度小（|skew|<0.5），kurtosis 非極端。',
    'moderately_skewed':'中等偏態：單峰但偏度中等（0.5≤|skew|<1.5）。',
    'highly_skewed':'高度偏態：偏度很大（|skew|≥1.5）或尾部極長。',
    'multimodal':'多峰：KDE 峰數 ≥ 2，表示可能有混合分布或潛在子群。',
    'heavy_tailed':'重尾：excess kurtosis 高（例如 >5），出現頻繁極端值。',
    'zero_inflated':'零膨脹：大量觀測為 0（zero_prop ≥ 0.20）。',
    'discrete':'離散：unique_ratio 很小或樣本值為整數（視為分類/計數）。',
    'constant':'常數：#unique == 1 或 unique_ratio ≤ 0.01。'
}
for k,v in defs.items():
    lines.append(f'- **{k}**：{v}')
lines.append('\n---\n')

# Add per-dataset table header
lines.append('\n## 每個資料集的關鍵統計與標籤依據\n')
lines.append('\n| dataset | primary_label | skewness | kurtosis | zero_prop | peaks | unique_ratio | 判定理由 |')
lines.append('|---|---|---:|---:|---:|---:|---:|---|')

for r in rows:
    csvpath = r['csv']
    fname = Path(csvpath).name
    labels = r['labels']
    primary = primary_from_labels(labels)
    skew = r.get('skewness','')
    kurt = r.get('kurtosis','')
    zero = r.get('zero_prop','')
    peaks = r.get('peaks','')
    uniq = r.get('unique_ratio','')
    reason = reason_for_row(r)
    # escape pipe in fname
    fname = fname.replace('|','\\|')
    lines.append(f'| {fname} | {primary} | {skew} | {kurt} | {zero} | {peaks} | {uniq} | {reason} |')

# Write back: append to existing md file
text = '\n'.join(lines) + '\n'
with open(md, 'a', encoding='utf-8') as f:
    f.write(text)
print('Appended details to', md)
