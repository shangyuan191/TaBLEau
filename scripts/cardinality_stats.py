#!/usr/bin/env python3
"""Compute non-parametric tests for label-cardinality effects per variant.

Reads:
 - gnn_enhancement_analysis/per_dataset_variant_ranks_mapped.csv

Writes:
 - analysis/variant_cardinality_stats.csv

For each (variant_id, ratio_cat) we run Kruskal-Wallis across cardinality bins (2,3-10,>10).
If KW p<0.05 we run pairwise Mann-Whitney U tests with Bonferroni correction.
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / 'gnn_enhancement_analysis' / 'per_dataset_variant_ranks_mapped.csv'
OUT_CSV = ROOT / 'analysis' / 'variant_cardinality_stats.csv'


def parse_variant(variant_str):
    m = re.search(r"^([^\(]+)\(ratio=([^,\)]+),\s*gnn_stage=([^\)]+)\)", variant_str)
    if not m:
        return variant_str, '', ''
    model = m.group(1).strip()
    ratio = m.group(2).strip()
    gnn_stage = m.group(3).strip()
    return model, ratio, gnn_stage


def ratio_cat_from_ratio(ratio):
    if isinstance(ratio, str) and ratio.startswith('0.05/'):
        return 'few'
    if isinstance(ratio, str) and ratio.startswith('0.8/'):
        return 'full'
    return 'other'


def cardinality_bin(n):
    try:
        n = int(n)
    except Exception:
        return 'unknown'
    if n == 2:
        return '2'
    if 3 <= n <= 10:
        return '3-10'
    if n > 10:
        return '>10'
    return 'unknown'


def pairwise_mwu(groups):
    # groups: dict label -> list of values
    keys = sorted(groups.keys(), key=lambda x: ['2','3-10','>10'].index(x) if x in ['2','3-10','>10'] else 99)
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = groups[keys[i]]
            b = groups[keys[j]]
            if len(a) < 3 or len(b) < 3:
                p = np.nan
                stat = np.nan
            else:
                try:
                    stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                except Exception:
                    stat, p = np.nan, np.nan
            pairs.append(((keys[i], keys[j]), stat, p))
    return pairs


def main():
    print('Reading', IN_CSV)
    df = pd.read_csv(IN_CSV)
    parsed = df['variant'].astype(str).apply(parse_variant)
    df[['model_parsed','ratio','gnn_stage']] = pd.DataFrame(parsed.tolist(), index=df.index)
    df['ratio_cat'] = df['ratio'].apply(ratio_cat_from_ratio)
    df['card_bin'] = df['n_unique'].apply(cardinality_bin)

    df = df[df['group'].str.contains('binclass|multiclass')]
    stages = {'encoding','decoding','columnwise','start','materialize'}
    df = df[df['gnn_stage'].isin(stages)]
    df['variant_id'] = df['model_parsed'] + '::' + df['gnn_stage']

    rows = []
    for (variant_id, ratio_cat), sub in df.groupby(['variant_id','ratio_cat']):
        groups = {}
        counts = {}
        for k, g in sub.groupby('card_bin'):
            groups[k] = g['rank'].dropna().astype(float).tolist()
            counts[k] = len(groups[k])

        # need at least two groups with >=3 items to run KW
        valid_groups = [k for k,v in counts.items() if v >= 3]
        H = np.nan; p_kw = np.nan; sig = False
        if len(valid_groups) >= 2:
            samples = [groups[k] for k in sorted(valid_groups)]
            try:
                H, p_kw = stats.kruskal(*samples)
                sig = p_kw < 0.05
            except Exception:
                H, p_kw = np.nan, np.nan

        # pairwise
        pair_res = { '2_vs_3-10_p': np.nan, '2_vs_>10_p': np.nan, '3-10_vs_>10_p': np.nan }
        if sig:
            pairs = pairwise_mwu(groups)
            # Bonferroni correction for up to 3 comparisons
            for (a,b), stat, p in pairs:
                key = f"{a}_vs_{b}_p"
                if key in pair_res:
                    pair_res[key] = min(1.0, p * 3) if not np.isnan(p) else np.nan

        rows.append({
            'variant_id': variant_id,
            'ratio_cat': ratio_cat,
            'n_2': counts.get('2',0),'n_3-10': counts.get('3-10',0),'n_>10': counts.get('>10',0),
            'H': H, 'p_kruskal': p_kw, 'sig_kruskal': sig,
            '2_vs_3-10_p': pair_res['2_vs_3-10_p'], '2_vs_>10_p': pair_res['2_vs_>10_p'], '3-10_vs_>10_p': pair_res['3-10_vs_>10_p']
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print('Wrote', OUT_CSV)


if __name__ == '__main__':
    main()
