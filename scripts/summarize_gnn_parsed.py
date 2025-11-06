#!/usr/bin/env python3
"""
Summarize GNN enhancement parsed CSV into human-readable summaries.

Outputs (written to project `analysis/`):
- gnn_summary.json  : machine-readable summary per-model
- gnn_summary.md    : concise markdown summary
- gnn_stage_dataset_summary.csv : stage x dataset-type breakdown

This script is conservative (no external deps) and robust to missing values.
"""
import csv
import os
import json
from collections import defaultdict


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_CSV = os.path.join(REPO_ROOT, 'analysis', 'gnn_enhancement_parsed.csv')
OUT_JSON = os.path.join(REPO_ROOT, 'analysis', 'gnn_summary.json')
OUT_MD = os.path.join(REPO_ROOT, 'analysis', 'gnn_summary.md')
OUT_STAGE_CSV = os.path.join(REPO_ROOT, 'analysis', 'gnn_stage_dataset_summary.csv')

REFERENCE_MODELS = set(['tabpfn', 't2g-former', 'tabgnn', 'xgboost', 'catboost', 'lightgbm'])


def float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_group_tokens(group):
    g = group.lower() if group else ''
    size = 'small' if 'small' in g else ('large' if 'large' in g else 'unknown')
    task = 'regression' if 'regression' in g else ('multiclass' if 'multiclass' in g else ('classification' if 'class' in g or 'binclass' in g else 'unknown'))
    dtype = 'numerical' if 'numerical' in g else ('categorical' if 'categorical' in g else ('balanced' if 'balanced' in g else 'unknown'))
    return size, task, dtype


def read_rows(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize keys
            row = {k.strip(): (v.strip() if v is not None else '') for k, v in r.items()}
            # normalize avg_rank and ratio
            row['avg_rank_f'] = float_or_none(row.get('avg_rank', ''))
            # preserve raw ratio string and classify into categories
            ratio_raw = row.get('ratio', '')
            row['ratio_raw'] = ratio_raw
            # many rows encode the split as a triple like '0.05/0.15/0.8' or '0.8/0.15/0.05'
            # we treat strings starting with '0.05' as few-shot, starting with '0.8' as full-sample
            if ratio_raw.startswith('0.05'):
                row['ratio_cat'] = 'few'
            elif ratio_raw.startswith('0.8'):
                row['ratio_cat'] = 'full'
            else:
                row['ratio_cat'] = 'other'
            rows.append(row)
    return rows


def summarize(rows):
    # Index rows by model, group, ratio, competitor
    by_model_group = defaultdict(list)
    by_group_ratio = defaultdict(list)
    for r in rows:
        model = r.get('model','')
        group = r.get('group','')
        ratio = r.get('ratio_f')
        by_model_group[(model, group)].append(r)
        by_group_ratio[(group, ratio)].append(r)

    models = sorted(set(r.get('model','') for r in rows))

    # treat ratio categories detected in parsing: 'few' and 'full'
    FEW_CAT = 'few'
    FULL_CAT = 'full'

    summary = {}

    for model in models:
        summary[model] = {
            'groups_total': 0,
            'fewshot_gnn_better_vs_fewshot_baseline': 0,
            'fewshot_gnn_better_vs_full_baseline': 0,
            'examples_vs_fewshot_baseline': [],
            'examples_vs_full_baseline': [],
            'vs_references_fewshot_better_count': 0,
            'vs_references_full_better_count': 0,
        }

    # iterate groups per model
    for (model, group), group_rows in by_model_group.items():
        summary[model]['groups_total'] += 1
        # collect baseline few and full (gnn_stage == none)
        baseline_few = min((r['avg_rank_f'] for r in group_rows if (r.get('gnn_stage','').lower()=='none' and r.get('ratio_cat')==FEW_CAT) and r['avg_rank_f'] is not None), default=None)
        baseline_full = min((r['avg_rank_f'] for r in group_rows if (r.get('gnn_stage','').lower()=='none' and r.get('ratio_cat')==FULL_CAT) and r['avg_rank_f'] is not None), default=None)

        # best GNN in few-shot
        best_gnn_few = None
        best_gnn_few_row = None
        for r in group_rows:
            if r.get('gnn_stage','').lower()!='none' and r.get('ratio_cat')==FEW_CAT and r['avg_rank_f'] is not None:
                if best_gnn_few is None or r['avg_rank_f'] < best_gnn_few:
                    best_gnn_few = r['avg_rank_f']
                    best_gnn_few_row = r

        if baseline_few is not None and best_gnn_few is not None and best_gnn_few < baseline_few:
            summary[model]['fewshot_gnn_better_vs_fewshot_baseline'] += 1
            if len(summary[model]['examples_vs_fewshot_baseline']) < 10:
                summary[model]['examples_vs_fewshot_baseline'].append({'group': group, 'baseline_few': baseline_few, 'best_gnn_few': best_gnn_few, 'row': best_gnn_few_row})

        # best GNN compared to full baseline
        if baseline_full is not None and best_gnn_few is not None and best_gnn_few < baseline_full:
            summary[model]['fewshot_gnn_better_vs_full_baseline'] += 1
            if len(summary[model]['examples_vs_full_baseline']) < 10:
                summary[model]['examples_vs_full_baseline'].append({'group': group, 'baseline_full': baseline_full, 'best_gnn_few': best_gnn_few, 'row': best_gnn_few_row})

        # compare to references (per ratio category)
        # few-shot reference best
        ref_best_few = min((r['avg_rank_f'] for r in group_rows if r.get('competitor','').lower() in REFERENCE_MODELS and r.get('ratio_cat')==FEW_CAT and r['avg_rank_f'] is not None), default=None)
        if ref_best_few is not None and best_gnn_few is not None and best_gnn_few < ref_best_few:
            summary[model]['vs_references_fewshot_better_count'] += 1

        # full-sample compare: consider gnn variants at full if present
        best_gnn_full = None
        for r in group_rows:
            if r.get('gnn_stage','').lower()!='none' and r.get('ratio_cat')==FULL_CAT and r['avg_rank_f'] is not None:
                if best_gnn_full is None or r['avg_rank_f'] < best_gnn_full:
                    best_gnn_full = r['avg_rank_f']

        ref_best_full = min((r['avg_rank_f'] for r in group_rows if r.get('competitor','').lower() in REFERENCE_MODELS and r.get('ratio_cat')==FULL_CAT and r['avg_rank_f'] is not None), default=None)
        if ref_best_full is not None and best_gnn_full is not None and best_gnn_full < ref_best_full:
            summary[model]['vs_references_full_better_count'] += 1

    # Stage x dataset-type summary
    stage_dtype_counts = defaultdict(lambda: {'count':0, 'better_than_fewshot_baseline':0})

    for (group, ratio), grouprows in by_group_ratio.items():
        # build baseline few for the group across all models (per-model baseline is used above);
        # here we examine each GNN row and test if it's better than that row's model baseline_few
        for r in grouprows:
            if r.get('gnn_stage','').lower()=='none':
                continue
            # only consider few-shot rows for the stage-dtype summary
            if r.get('ratio_cat') != FEW_CAT:
                continue
            model = r.get('model','')
            # find corresponding model baseline few in same group
            candidates = [x for x in by_model_group.get((model, r.get('group','')), []) if x.get('gnn_stage','').lower()=='none' and x.get('ratio_cat')==FEW_CAT and x.get('avg_rank_f') is not None]
            if not candidates:
                continue
            baseline_few_val = min(x['avg_rank_f'] for x in candidates)
            size, task, dtype = parse_group_tokens(r.get('group',''))
            key = (r.get('gnn_stage','').lower(), size, task, dtype)
            stage_dtype_counts[key]['count'] += 1
            if r.get('avg_rank_f') is not None and baseline_few_val is not None and r['avg_rank_f'] < baseline_few_val:
                stage_dtype_counts[key]['better_than_fewshot_baseline'] += 1

    return summary, stage_dtype_counts


def write_outputs(summary, stage_dtype_counts):
    # JSON
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    # Markdown: concise per-model table
    lines = []
    lines.append('# GNN enhancement summary')
    lines.append('')
    for m, s in sorted(summary.items(), key=lambda x: x[0]):
        lines.append(f'## {m}')
        lines.append(f'- groups considered: {s["groups_total"]}')
        lines.append(f'- few-shot GNN better vs few-shot baseline: {s["fewshot_gnn_better_vs_fewshot_baseline"]}')
        lines.append(f'- few-shot GNN better vs full-sample baseline: {s["fewshot_gnn_better_vs_full_baseline"]}')
        lines.append(f'- vs reference models (few-shot) better count: {s.get("vs_references_fewshot_better_count",0)}')
        lines.append(f'- vs reference models (full) better count: {s.get("vs_references_full_better_count",0)}')
        if s['examples_vs_fewshot_baseline']:
            lines.append('- examples (few-shot > few-shot baseline):')
            for ex in s['examples_vs_fewshot_baseline']:
                lines.append(f'  - group: {ex["group"]}, baseline_few={ex["baseline_few"]}, best_gnn_few={ex["best_gnn_few"]}, row_source={ex["row"].get("source_file","?")}')
        lines.append('')
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines))

    # stage x dataset csv
    with open(OUT_STAGE_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gnn_stage','size','task','dtype','count','better_than_fewshot_baseline','prop_better'])
        for k, v in sorted(stage_dtype_counts.items(), key=lambda x: (-x[1]['count'], x[0])):
            stage, size, task, dtype = k
            count = v['count']
            better = v['better_than_fewshot_baseline']
            prop = better / count if count else 0
            w.writerow([stage, size, task, dtype, count, better, '{:.3f}'.format(prop)])


def main():
    if not os.path.exists(INPUT_CSV):
        print('ERROR: parsed CSV not found at', INPUT_CSV)
        return 2
    rows = read_rows(INPUT_CSV)
    summary, stage_dtype_counts = summarize(rows)
    write_outputs(summary, stage_dtype_counts)
    # print short interactive summary
    print('Wrote summaries:')
    print(' -', OUT_JSON)
    print(' -', OUT_MD)
    print(' -', OUT_STAGE_CSV)
    # quick per-model headline
    for m, s in sorted(summary.items(), key=lambda x: (-x[1]['fewshot_gnn_better_vs_fewshot_baseline'], x[0])):
        print(f"{m}: groups={s['groups_total']}, fewshot>fewshot_baseline={s['fewshot_gnn_better_vs_fewshot_baseline']}, fewshot>full_baseline={s['fewshot_gnn_better_vs_full_baseline']}, vs_refs_few={s['vs_references_fewshot_better_count']}, vs_refs_full={s['vs_references_full_better_count']}")


if __name__ == '__main__':
    exit(main())
