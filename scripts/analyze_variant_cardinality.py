#!/usr/bin/env python3
"""Analyze how label cardinality affects GNN variant ranks.

Outputs:
 - analysis/variant_cardinality_summary.csv
 - analysis/figs/variant_cardinality_boxplots/*.png

This script reads:
 - gnn_enhancement_analysis/per_dataset_variant_ranks_mapped.csv

It produces summaries for models x gnn_stage (only the 5 stages requested)
and splits by ratio_cat (few/full) and cardinality bins (2 / 3-10 / >10).
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / 'gnn_enhancement_analysis' / 'per_dataset_variant_ranks_mapped.csv'
OUT_DIR = ROOT / 'analysis'
FIG_DIR = OUT_DIR / 'figs' / 'variant_cardinality_boxplots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_variant(variant_str):
    # examples: resnet(ratio=0.8/0.15/0.05, gnn_stage=decoding)
    m = re.search(r"^([^\(]+)\(ratio=([^,\)]+),\s*gnn_stage=([^\)]+)\)", variant_str)
    if not m:
        # try looser parse
        m2 = re.search(r"^([^\(]+)", variant_str)
        model = m2.group(1).strip() if m2 else variant_str
        return model, '', ''
    model = m.group(1).strip()
    ratio = m.group(2).strip()
    gnn_stage = m.group(3).strip()
    return model, ratio, gnn_stage


def ratio_cat_from_ratio(ratio):
    if ratio.startswith('0.05/'):
        return 'few'
    if ratio.startswith('0.8/'):
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


def main():
    print('Reading mapped per-dataset ranks (if available):', IN_CSV)
    mapped_path = IN_CSV
    df_mapped = None
    if mapped_path.exists():
        df_mapped = pd.read_csv(mapped_path)
        df_mapped.columns = [c.strip() for c in df_mapped.columns]
    else:
        print('Warning: per-dataset mapped CSV not found, will build from parsed CSV fallback.')

    # read parsed aggregated CSV and per-dataset cardinality mapping
    parsed_csv = ROOT / 'analysis' / 'gnn_enhancement_parsed.csv'
    mapping_json = ROOT / 'gnn_enhancement_analysis' / 'per_dataset_target_value_counts_full.json'
    print('Reading parsed CSV:', parsed_csv)
    df_parsed = pd.read_csv(parsed_csv)
    import json
    mapping = json.load(mapping_json.open())

    # target stages
    stages = {'encoding','decoding','columnwise','start','materialize'}

    rows = []

    # iterate parsed rows where competitor==model
    sel = df_parsed[(df_parsed['competitor'] == df_parsed['model']) & (df_parsed['gnn_stage'].isin(stages)) & (df_parsed['group'].str.contains('binclass|multiclass'))]
    print('Found', len(sel), 'parsed rows for competitor==model and target stages')

    for _, r in sel.iterrows():
        variant = f"{r['model']}(ratio={r['ratio']}, gnn_stage={r['gnn_stage']})"
        # try mapped per-dataset
        used = False
        if df_mapped is not None:
            sub = df_mapped[df_mapped['variant'] == variant]
            if not sub.empty:
                for _, s in sub.iterrows():
                    rows.append({'variant': variant, 'chosen_csv': s['chosen_csv'], 'n_unique': int(s['n_unique']), 'rank': float(s['rank']), 'group': s.get('group', r['group'])})
                used = True

        if not used:
            # fallback: expand to all dataset paths in mapping whose path contains the group path
            group_path = '/datasets/' + r['group'].replace('+', '/') + '/'
            matched = [p for p in mapping.keys() if group_path in p]
            if not matched:
                # try looser match without leading '/datasets'
                group_path2 = r['group'].replace('+', '/')
                matched = [p for p in mapping.keys() if group_path2 in p]
            for p in matched:
                n_unique = int(mapping[p].get('n_unique', 0))
                # use avg_rank as fallback per-dataset rank
                rank_val = float(r.get('avg_rank', np.nan))
                rows.append({'variant': variant, 'chosen_csv': p, 'n_unique': n_unique, 'rank': rank_val, 'group': r['group']})

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print('No per-dataset rows could be produced. Exiting.')
        return

    # parse variant into components
    parsed = df_all['variant'].astype(str).apply(parse_variant)
    df_all[['model_parsed','ratio','gnn_stage']] = pd.DataFrame(parsed.tolist(), index=df_all.index)
    df_all['ratio_cat'] = df_all['ratio'].astype(str).apply(ratio_cat_from_ratio)
    df_all['card_bin'] = df_all['n_unique'].apply(cardinality_bin)

    # filter again for safety
    df = df_all[df_all['group'].str.contains('binclass|multiclass')]
    df = df[df['gnn_stage'].isin(stages)]
    df['variant_id'] = df['model_parsed'] + '::' + df['gnn_stage']

    # aggregate
    agg = df.groupby(['variant_id','model_parsed','gnn_stage','ratio_cat','card_bin'])['rank'] \
        .agg(['count','mean','median','std']) \
        .reset_index()
    agg = agg.rename(columns={'count':'n','mean':'mean_rank','median':'median_rank','std':'std_rank'})

    def iqr(series):
        return float(np.percentile(series,75) - np.percentile(series,25))

    iqr_df = df.groupby(['variant_id','ratio_cat','card_bin'])['rank'].apply(iqr).reset_index().rename(columns={'rank':'iqr_rank'})
    agg = agg.merge(iqr_df, on=['variant_id','ratio_cat','card_bin'], how='left')

    out_csv = OUT_DIR / 'variant_cardinality_summary.csv'
    agg.to_csv(out_csv, index=False)
    print('Wrote summary to', out_csv)

    variants = sorted(df['variant_id'].unique())
    print('Found', len(variants), 'variants to plot')

    for v in variants:
        sub = df[df['variant_id']==v].copy()
        if sub.empty:
            continue
        cat_order = ['2','3-10','>10']
        sub['card_bin'] = pd.Categorical(sub['card_bin'], categories=cat_order, ordered=True)

        plt.figure(figsize=(8,5))
        sns.boxplot(x='card_bin', y='rank', hue='ratio_cat', data=sub, showfliers=False)
        plt.title(f'{v} (n={len(sub)})')
        plt.ylabel('rank (lower is better)')
        plt.xlabel('label cardinality bin')
        plt.legend(title='ratio_cat')
        fname = FIG_DIR / f"{v.replace('/','_').replace(':','-')}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    print('Saved plots to', FIG_DIR)


if __name__ == '__main__':
    main()
