#!/usr/bin/env python3
"""
Compute paired per-dataset differences (baseline vs GNN variants) using mapped per-dataset ranks
with parsed avg_rank as fallback. Group datasets by primary_label and run paired Wilcoxon tests.

Outputs (under analysis/paired_tests/):
- per_dataset_paired_table.csv  : per-dataset rows with baseline/variant ranks, diff, pct_change, is_fallback
- group_test_results.csv       : per (model,ratio,gnn_stage,primary_label) Wilcoxon results and effect sizes
- figs/                       : PNG boxplots per primary_label summarizing diffs across variants

This script expects these files to exist in the repo:
- gnn_enhancement_analysis/per_dataset_variant_ranks_mapped.csv
- analysis/gnn_enhancement_parsed.csv
- gnn_enhancement_analysis/per_dataset_target_value_counts_full.json
- analysis/regression_target_distributions_analysis/target_distribution_classified.csv

Usage: run in the project's conda env where pandas/scipy/matplotlib are installed.
"""
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(__file__))
MAP_CSV = os.path.join(ROOT, 'gnn_enhancement_analysis', 'per_dataset_variant_ranks_mapped.csv')
PARSED_CSV = os.path.join(ROOT, 'analysis', 'gnn_enhancement_parsed.csv')
TARGET_JSON = os.path.join(ROOT, 'gnn_enhancement_analysis', 'per_dataset_target_value_counts_full.json')
CLASSIFIED = os.path.join(ROOT, 'analysis', 'regression_target_distributions_analysis', 'target_distribution_classified.csv')
OUT_DIR = os.path.join(ROOT, 'analysis', 'paired_tests')
FIG_DIR = os.path.join(OUT_DIR, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)


def make_variant_string(row):
    return f"{row['model']}(ratio={row['ratio']}, gnn_stage={row['gnn_stage']})"


def load_inputs():
    df_map = pd.read_csv(MAP_CSV)
    df_parsed = pd.read_csv(PARSED_CSV)
    with open(TARGET_JSON, 'r', encoding='utf-8') as f:
        ds_target = json.load(f)
    df_class = pd.read_csv(CLASSIFIED)
    return df_map, df_parsed, ds_target, df_class


def build_parsed_variants(df_parsed):
    df = df_parsed.copy()
    df['variant'] = df.apply(lambda r: f"{r.model}(ratio={r.ratio}, gnn_stage={r.gnn_stage})", axis=1)
    # keep representative avg_rank per variant (group by variant and take mean)
    parsed_avg = df.groupby('variant', as_index=False).agg({'avg_rank':'mean', 'model':'first', 'ratio':'first', 'gnn_stage':'first'})
    return parsed_avg


def expand_per_dataset(parsed_avg, ds_target):
    # ds_target: mapping dataset_path -> n_unique
    datasets = sorted(ds_target.keys())
    rows = []
    for _, pv in parsed_avg.iterrows():
        for d in datasets:
            rows.append({'variant': pv['variant'], 'chosen_csv': d, 'avg_rank': pv['avg_rank'], 'model': pv['model'], 'ratio': pv['ratio'], 'gnn_stage': pv['gnn_stage']})
    return pd.DataFrame(rows)


def build_per_dataset_table(df_map, parsed_expanded, parsed_avg):
    # Start from expanded parsed (one row per dataset x variant)
    df = parsed_expanded.copy()
    # merge true mapped ranks where available
    df_map2 = df_map.rename(columns={'rank':'mapped_rank'})[['variant','chosen_csv','mapped_rank']]
    df = df.merge(df_map2, how='left', left_on=['variant','chosen_csv'], right_on=['variant','chosen_csv'])
    # if mapped_rank exists use it, else fallback to avg_rank
    df['rank'] = df['mapped_rank'].where(~df['mapped_rank'].isna(), df['avg_rank'])
    df['is_fallback'] = df['mapped_rank'].isna()
    # attach primary_label mapping
    return df


def attach_primary_label(df, df_class):
    # df_class columns include 'chosen_csv' and 'primary_label'
    if 'chosen_csv' not in df_class.columns:
        # try dataset column name
        df_class = df_class.rename(columns={df_class.columns[0]:'chosen_csv'})
    if 'primary_label' not in df_class.columns:
        raise RuntimeError('target_distribution_classified.csv missing primary_label column')
    df = df.merge(df_class[['chosen_csv','primary_label']], how='left', left_on='chosen_csv', right_on='chosen_csv')
    return df


def find_baseline_variant(parsed_avg, model, ratio):
    # baseline is gnn_stage == 'none' for same model+ratio
    candidates = parsed_avg[(parsed_avg['model']==model)&(parsed_avg['ratio']==ratio)&(parsed_avg['gnn_stage']=='none')]
    if not candidates.empty:
        return candidates['variant'].iloc[0]
    # fallback: try gnn_stage='None' case-insensitive
    candidates = parsed_avg[(parsed_avg['model']==model)&(parsed_avg['ratio']==ratio)&(parsed_avg['gnn_stage'].str.lower()=='none')]
    if not candidates.empty:
        return candidates['variant'].iloc[0]
    return None


def run_tests_and_save(df, parsed_avg, min_group_size=5, out_dir=OUT_DIR, fig_dir=FIG_DIR):
    per_dataset_out = []
    test_rows = []
    print('DEBUG: parsed_avg groups:', parsed_avg.groupby(['model','ratio']).ngroups)
    print('DEBUG: total rows in df (per-dataset merged):', len(df))
    # iterate over unique model+ratio groups
    groups = parsed_avg.groupby(['model','ratio'])
    for (model, ratio), group in groups:
        baseline_variant = find_baseline_variant(parsed_avg, model, ratio)
        if baseline_variant is None:
            continue
        # variants for this model+ratio excluding baseline
        variants = group['variant'].unique().tolist()
        variants = [v for v in variants if v != baseline_variant]
        if len(variants)==0:
            continue
        # for each primary_label
        for plabel, df_pl in df.groupby('primary_label'):
            ds_in_group = df_pl['chosen_csv'].unique().tolist()
            if len(ds_in_group) < min_group_size:
                # skip small groups but still record counts
                for v in variants:
                    test_rows.append({'model':model,'ratio':ratio,'gnn_variant':v,'primary_label':plabel,'n_datasets':len(ds_in_group),'wilcoxon_stat':None,'wilcoxon_p':None,'median_diff':None,'mean_diff':None,'sd_diff':None,'cohens_d':None,'n_fallbacks':None})
                continue
            # build arrays of baseline and each variant ranks across datasets in group
            base_df = df[(df['variant']==baseline_variant)&(df['chosen_csv'].isin(ds_in_group))][['chosen_csv','rank','is_fallback']].rename(columns={'rank':'baseline_rank','is_fallback':'baseline_is_fallback'})
            for v in variants:
                var_df = df[(df['variant']==v)&(df['chosen_csv'].isin(ds_in_group))][['chosen_csv','rank','is_fallback']].rename(columns={'rank':'variant_rank','is_fallback':'variant_is_fallback'})
                merged = base_df.merge(var_df, on='chosen_csv', how='inner')
                n = len(merged)
                n_fallbacks = int((merged['baseline_is_fallback'] | merged['variant_is_fallback']).sum())
                if n==0:
                    test_rows.append({'model':model,'ratio':ratio,'gnn_variant':v,'primary_label':plabel,'n_datasets':0,'wilcoxon_stat':None,'wilcoxon_p':None,'median_diff':None,'mean_diff':None,'sd_diff':None,'cohens_d':None,'n_fallbacks':n_fallbacks})
                    continue
                # compute baseline - variant (positive => baseline better/higher rank)
                merged['diff'] = merged['baseline_rank'] - merged['variant_rank']
                merged['pct_change'] = 100.0 * (merged['variant_rank'] - merged['baseline_rank']) / merged['baseline_rank'].replace({0:np.nan})
                per_dataset_out.append({'model':model,'ratio':ratio,'gnn_variant':v,'primary_label':plabel,'n_datasets':n,'n_fallbacks':n_fallbacks,'dataset_diffs':merged[['chosen_csv','baseline_rank','variant_rank','diff','pct_change','baseline_is_fallback','variant_is_fallback']].to_json(orient='records')})
                # test
                if n < min_group_size:
                    stat = None; p = None
                else:
                    try:
                        stat, p = stats.wilcoxon(merged['baseline_rank'], merged['variant_rank'])
                    except Exception:
                        # fallback: use paired t-test
                        stat, p = stats.ttest_rel(merged['baseline_rank'], merged['variant_rank'])
                mean_diff = merged['diff'].mean()
                median_diff = merged['diff'].median()
                sd_diff = merged['diff'].std(ddof=1)
                cohens_d = mean_diff / sd_diff if (sd_diff and not np.isnan(sd_diff)) else None
                test_rows.append({'model':model,'ratio':ratio,'gnn_variant':v,'primary_label':plabel,'n_datasets':n,'wilcoxon_stat':stat,'wilcoxon_p':p,'median_diff':median_diff,'mean_diff':mean_diff,'sd_diff':sd_diff,'cohens_d':cohens_d,'n_fallbacks':n_fallbacks})

    print('DEBUG: per_dataset_out rows:', len(per_dataset_out))
    print('DEBUG: test_rows rows:', len(test_rows))
    per_dataset_df = pd.DataFrame(per_dataset_out)
    test_df = pd.DataFrame(test_rows)
    # multiple-testing correction: Bonferroni within each (model,ratio,primary_label) across variants
    if not test_df.empty and {'model','ratio','primary_label','wilcoxon_p'}.issubset(set(test_df.columns)):
        for (model,ratio,pl), sub in test_df.groupby(['model','ratio','primary_label']):
            indices = sub.index.tolist()
            ps = sub['wilcoxon_p'].to_numpy()
            mask = ~pd.isna(ps)
            corrected_p = [None] * len(ps)
            if mask.any():
                m = int(mask.sum())
                corrected_vals = np.minimum(1.0, ps[mask] * m)
                j = 0
                for i in range(len(ps)):
                    if mask[i]:
                        corrected_p[i] = float(corrected_vals[j])
                        j += 1
            # assign back to dataframe
            test_df.loc[indices, 'wilcoxon_p_bonf'] = corrected_p

    # save outputs
    per_dataset_csv = os.path.join(out_dir, 'per_dataset_paired_table.csv')
    test_csv = os.path.join(out_dir, 'group_test_results.csv')
    per_dataset_df.to_csv(per_dataset_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # simple boxplot per primary_label: collect diffs across all (model,ratio,gnn_variant)
    if (not per_dataset_df.empty) and ('primary_label' in per_dataset_df.columns):
        for pl, sub in per_dataset_df.groupby('primary_label'):
            if sub.empty:
                continue
            # each row has dataset_diffs json: extract diffs
            items = []
            labels = []
            for _, r in sub.iterrows():
                arr = pd.read_json(r['dataset_diffs'])
                items.append(arr['diff'].values)
                labels.append(f"{r['model']}|{r['ratio']}|{r['gnn_variant'].split(')',1)[0]+')'}")
            plt.figure(figsize=(max(8, len(items)*0.4),4))
            plt.boxplot(items, labels=labels, vert=False)
            plt.title(f'Diff (baseline - variant) per variant — primary_label={pl}')
            plt.tight_layout()
            fn = os.path.join(fig_dir, f'boxplot_diffs_primarylabel_{pl}.png')
            plt.savefig(fn, dpi=150)
            plt.close()

    print('Saved', per_dataset_csv, test_csv, 'and figures in', fig_dir)


def main():
    df_map, df_parsed, ds_target, df_class = load_inputs()
    parsed_avg = build_parsed_variants(df_parsed)
    parsed_expanded = expand_per_dataset(parsed_avg, ds_target)
    df = build_per_dataset_table(df_map, parsed_expanded, parsed_avg)
    df = attach_primary_label(df, df_class)
    # use min_group_size=1 so we still collect results even for small groups (we'll flag small groups)
    run_tests_and_save(df, parsed_avg, min_group_size=1)


if __name__ == '__main__':
    main()
