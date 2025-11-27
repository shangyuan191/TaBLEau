import os
import re
import argparse
import csv
from pathlib import Path
import pandas as pd

# This script aggregates the per-category avg_rank tables that were
# produced under `gnn_injection_analysis/per_model_result/*.txt` and
# computes a dataset-weighted combined average rank across all categories
# (i.e. "all datasets"). The source .txt files already contain avg_rank
# values that were computed per-category from per-dataset metrics; those
# rankings already account for metric direction (classification higher-is-better
# vs regression lower-is-better) so we simply weight by dataset counts.

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "gnn_injection_analysis", "per_model_result")

def parse_enhancement_txt(path: str):
    """Parse a *_gnn_enhancement.txt file and return mapping:
    competitor -> (sum_rank_times_count, total_count)

    The txt files contain multiple category sections. Each table row has the
    form (columns separated by two or more spaces):
      <rank>  <competitor_str>  <avg_rank>  <dataset_count>
    We will extract competitor_str, avg_rank (float) and dataset_count (int)
    and accumulate avg_rank * dataset_count to compute a weighted average later.
    """
    acc = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # skip separator lines
            if not line.strip():
                continue
            # split by 2+ spaces to isolate columns (rank, competitor, avg_rank, count)
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 4:
                continue
            # first part should be the rank (integer)
            try:
                _rank = int(parts[0])
            except Exception:
                continue
            competitor = parts[1].strip()
            try:
                avg_rank = float(parts[2])
                cnt = int(parts[3])
            except Exception:
                # if parsing fails, skip this line
                continue
            if cnt <= 0:
                continue
            if competitor not in acc:
                acc[competitor] = {"weighted_sum": 0.0, "total_datasets": 0}
            acc[competitor]["weighted_sum"] += avg_rank * cnt
            acc[competitor]["total_datasets"] += cnt
    return acc


def compute_combined_ranks_for_file(txt_path: str):
    acc = parse_enhancement_txt(txt_path)
    # compute combined average rank and total datasets
    results = []
    total_datasets = 0
    for comp, v in acc.items():
        total_datasets += v["total_datasets"]
    for comp, v in acc.items():
        avg = v["weighted_sum"] / v["total_datasets"]
        results.append({"competitor": comp, "avg_rank": avg, "datasets": v["total_datasets"]})
    # sort ascending avg_rank (lower is better)
    results = sorted(results, key=lambda x: x["avg_rank"])  
    return results, total_datasets


def write_all_dataset_md(results, total_datasets, out_path, title="All datasets"):
    header = f"## Category: all dataset (包含 {total_datasets} 個資料集)\n"
    df_lines = [
        "| 排名 | 競爭者 | 平均排名 | 資料集數 |",
        "|---:|---|---:|---:|",
    ]
    for i, r in enumerate(results, start=1):
        df_lines.append(f"| {i} | {r['competitor']} | {r['avg_rank']:.2f} | {r['datasets']} |")
    content = header + "\n" + "\n".join(df_lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


if __name__ == '__main__':
    # We'll compute combined rankings for both 'test' and 'val' metrics
    metrics = ['test', 'val']
    SUMMARY_DIR = os.path.join(BASE_DIR, "summary_results")
    if os.path.isdir(SUMMARY_DIR):
        print(f"Found summary_results at {SUMMARY_DIR}, parsing per-dataset metrics...")
        # build dataset -> task_type and dataset -> category map by scanning datasets folder
        datasets_root = os.path.join(BASE_DIR, "datasets")
        dataset_task_map = {}
        dataset_category_map = {}
        if os.path.isdir(datasets_root):
            for size in os.listdir(datasets_root):
                size_dir = os.path.join(datasets_root, size)
                if not os.path.isdir(size_dir):
                    continue
                for task_type in os.listdir(size_dir):
                    task_dir = os.path.join(size_dir, task_type)
                    if not os.path.isdir(task_dir):
                        continue
                    for feature in os.listdir(task_dir):
                        feature_dir = os.path.join(task_dir, feature)
                        if not os.path.isdir(feature_dir):
                            continue
                        for dataset in os.listdir(feature_dir):
                            dataset_task_map[dataset] = task_type
                            dataset_category_map[dataset] = f"{size}+{task_type}+{feature}"
        # parse all summary files for each requested metric and write outputs
        # CLI args: allow per-dataset wins and eps sensitivity
        parser = argparse.ArgumentParser()
        parser.add_argument('--per-dataset-wins', action='store_true', help='Compute per-dataset wins and write CSVs')
        parser.add_argument('--eps-list', default='0.1', help='Comma-separated eps values for tie sensitivity (e.g. 0.1,0.05,0.01)')
        parser.add_argument('--eps-mode', choices=['absolute','relative'], default='relative', help='Whether eps is absolute difference or relative fraction (relative uses denominator |b|+tiny)')
        parser.add_argument('--eps-behavior', choices=['symmetric','asymmetric'], default='asymmetric', help='Tie/beat decision behavior: symmetric (measure compared to ±eps) or asymmetric (any improvement -> strict beat; small losses within eps -> tie)')
        parser.add_argument('--make-wide-sensitivity', action='store_true', help='Generate wide-format sensitivity CSVs (one per eps)')
        parser.add_argument('--wide-by-category', action='store_true', help='Also write per-primary markdown with per-category wide tables')
        args = parser.parse_args()

        eps_list = [float(x) for x in args.eps_list.split(',') if x.strip()]

        for metric_name in metrics:
            per_dataset_competitors = {}  # dataset -> {competitor: metric}
            for fn in os.listdir(SUMMARY_DIR):
                if not fn.endswith('.txt'):
                    continue
                path = os.path.join(SUMMARY_DIR, fn)
                # try to infer model and ratio from filename
                m = re.search(r"all_models_([^_]+).*_(\d+\.\d+_\d+\.\d+_\d+\.\d+)\.txt$", fn)
                file_model = None
                file_ratio = None
                if m:
                    file_model = m.group(1)
                    file_ratio = m.group(2).replace('_', '/')
                # open and parse
                with open(path, 'r', encoding='utf-8') as f:
                    current_dataset = None
                    current_model = None
                    current_stage = None
                    for line in f:
                        line = line.strip()
                        if line.startswith('dataset:'):
                            current_dataset = line.split(':', 1)[1].strip()
                            continue
                        if line.startswith('模型:'):
                            current_model = line.split(':', 1)[1].strip()
                            continue
                        if line.startswith('GNN階段:'):
                            current_stage = line.split(':', 1)[1].strip()
                            continue
                        if line.startswith(f'Best {metric_name} metric:') and current_dataset and current_model and current_stage:
                            try:
                                metric = float(line.split(':', 1)[1].strip())
                            except Exception:
                                continue
                            # competitor label: model(ratio=..., gnn_stage=...)
                            ratio_label = file_ratio if file_ratio is not None else 'unknown'
                            comp = f"{current_model}(ratio={ratio_label}, gnn_stage={current_stage})"
                            per_dataset_competitors.setdefault(current_dataset, {})[comp] = metric

            # Now compute per-dataset ranks and aggregate
            competitor_rank_sums = {}  # comp -> sum ranks
            competitor_counts = {}
            total_datasets = 0
            for dataset, comp_dict in per_dataset_competitors.items():
                total_datasets += 1
                task = dataset_task_map.get(dataset, 'binclass')
                # sort competitors according to task direction
                items = list(comp_dict.items())  # (comp, metric)
                # remove comps with NaN
                items = [(c, m) for c, m in items if isinstance(m, (int, float))]
                if not items:
                    continue
                reverse = True if task in ('binclass', 'multiclass') else False
                # sort by metric (reverse True for higher-is-better)
                items_sorted = sorted(items, key=lambda x: x[1], reverse=reverse)
                # assign ranks with tie handling (equal metric -> same rank)
                prev_metric = None
                rank = 0
                next_rank = 1
                tol = 1e-6
                for comp, metric in items_sorted:
                    if prev_metric is None or abs(metric - prev_metric) > tol:
                        rank = next_rank
                    # accumulate
                    competitor_rank_sums[comp] = competitor_rank_sums.get(comp, 0.0) + rank
                    competitor_counts[comp] = competitor_counts.get(comp, 0) + 1
                    prev_metric = metric
                    next_rank += 1
            # compute avg ranks
            combined = []
            for comp, s in competitor_rank_sums.items():
                cnt = competitor_counts.get(comp, 1)
                combined.append({'competitor': comp, 'avg_rank': s / cnt, 'datasets': cnt})
            combined_sorted = sorted(combined, key=lambda x: x['avg_rank'])
            out_all = os.path.join(RESULT_DIR, f'all_models_ranking_all_datasets_by_{metric_name}.md')
            # write markdown
            with open(out_all, 'w', encoding='utf-8') as f:
                f.write(f"## All datasets combined (包含 {total_datasets} 個 dataset)\n\n")
                f.write("| 排名 | 競爭者 | 平均排名 | 資料集數 |\n")
                f.write("|---:|---|---:|---:|\n")
                for i, r in enumerate(combined_sorted, start=1):
                    f.write(f"| {i} | {r['competitor']} | {r['avg_rank']:.2f} | {r['datasets']} |\n")
            print(f"Wrote global all-dataset ranking to: {out_all}")

            # --- produce one table per primary (splittable) model
            primaries = [
                'excelformer', 'fttransformer', 'resnet', 'scarf', 'subtab',
                'tabm', 'tabnet', 'tabtransformer', 'trompt', 'vime'
            ]
            reference_models = ['t2g-former', 'tabgnn', 'tabpfn', 'xgboost', 'catboost', 'lightgbm']
            avg_lookup = {c['competitor']: (c['avg_rank'], c['datasets']) for c in combined}

            def parse_comp_label(comp_label):
                base = comp_label.split('(', 1)[0].strip()
                ratio = None
                stage = None
                m_ratio = re.search(r"ratio=([^,\)\s]+)", comp_label)
                if m_ratio:
                    ratio = m_ratio.group(1)
                m_stage = re.search(r"gnn_stage=([^,\)\s]+)", comp_label)
                if m_stage:
                    stage = m_stage.group(1)
                return base, ratio, stage

            combined_out = os.path.join(RESULT_DIR, f'primaries_all_datasets_ranking_by_{metric_name}.md')
            with open(combined_out, 'w', encoding='utf-8') as outf:
                outf.write(f"# All primary models — all datasets (包含 {total_datasets} 個資料集)\n\n")
                for primary in primaries:
                    filtered = []
                    for comp, (avg_r, cnt) in avg_lookup.items():
                        base, ratio, stage = parse_comp_label(comp)
                        if base in reference_models:
                            if ratio in ('0.05/0.15/0.8', '0.8/0.15/0.05'):
                                filtered.append({'competitor': comp, 'avg_rank': avg_r, 'datasets': cnt})
                            continue
                        if base == primary:
                            if ratio == '0.8/0.15/0.05':
                                if stage == 'none':
                                    filtered.append({'competitor': comp, 'avg_rank': avg_r, 'datasets': cnt})
                            else:
                                filtered.append({'competitor': comp, 'avg_rank': avg_r, 'datasets': cnt})
                    filtered = sorted(filtered, key=lambda x: x['avg_rank'])
                    outf.write(f"## Primary: {primary}\n\n")
                    outf.write("| 排名 | 競爭者 | 平均排名 | 資料集數 |\n")
                    outf.write("|---:|---|---:|---:|\n")
                    for i, r in enumerate(filtered, start=1):
                        outf.write(f"| {i} | {r['competitor']} | {r['avg_rank']:.2f} | {r['datasets']} |\n")
                    outf.write("\n")
            print(f"Wrote combined primaries ranking to: {combined_out}")

            # If requested, compute per-dataset wins and eps sensitivity
            if args.per_dataset_wins:
                print('Computing per-dataset wins and eps sensitivity...')
                # per_dataset_competitors: dataset -> {comp_label: metric}
                primaries_list = ['excelformer', 'fttransformer', 'resnet', 'scarf', 'subtab',
                                  'tabm', 'tabnet', 'tabtransformer', 'trompt', 'vime']
                trees = ['xgboost', 'catboost', 'lightgbm']
                gnn_refs = ['t2g-former', 'tabgnn']
                tabpfn = 'tabpfn'
                injection_stages = ['columnwise','none','decoding','encoding','start','materialize']
                small_ratio = '0.05/0.15/0.8'
                large_ratio = '0.8/0.15/0.05'

                # helper: compare metrics with eps given direction
                def is_better(a, b, eps, higher_is_better, eps_mode='absolute', eps_behavior='symmetric'):
                    # returns True (a better), False (a worse), None (tie/unknown)
                    if a is None or b is None:
                        return None
                    tiny = 1e-12
                    if eps_mode == 'absolute':
                        measure = a - b
                    else:
                        denom = abs(b) + tiny
                        measure = (a - b) / denom
                    # For lower-is-better, invert the sign so positive measure means 'a is better'
                    if not higher_is_better:
                        measure = -measure
                    if eps_behavior == 'symmetric':
                        if measure > eps:
                            return True
                        if abs(measure) <= eps:
                            return None
                        return False
                    else:
                        # asymmetric: any positive measure => strict beat
                        if measure > 0:
                            return True
                        # if measure <= 0 but within eps (i.e., a slightly worse or equal), treat as tie
                        if abs(measure) <= eps:
                            return None
                        # otherwise a is worse
                        return False

                # write per-primary CSVs (one file per primary) and sensitivity summary
                combined_md_paths_all_primaries = []
                # store combined_agg per primary so we can build a group-level summary later
                group_agg_by_primary = {}
                for p in primaries_list:
                    rows = []
                    for dataset, comp_metrics in per_dataset_competitors.items():
                        task = dataset_task_map.get(dataset, 'binclass')
                        higher_is_better = True if task in ('binclass', 'multiclass') else False
                        for inj in injection_stages:
                            # determine comp labels for primary
                            if inj == 'none':
                                label = f"{p}(ratio={small_ratio}, gnn_stage=none)"
                            else:
                                label = f"{p}(ratio={small_ratio}, gnn_stage={inj})"
                            full_label = f"{p}(ratio={large_ratio}, gnn_stage=none)"
                            val = comp_metrics.get(label)
                            baseline_val = comp_metrics.get(f"{p}(ratio={small_ratio}, gnn_stage=none)")
                            full_val = comp_metrics.get(full_label)
                            # include reference models (both ratios)
                            ref_vals = {}
                            for ref in trees + gnn_refs + [tabpfn]:
                                ref_vals[f"{ref}__few"] = comp_metrics.get(f"{ref}(ratio={small_ratio}, gnn_stage=none)")
                                ref_vals[f"{ref}__full"] = comp_metrics.get(f"{ref}(ratio={large_ratio}, gnn_stage=none)")
                            # build row
                            row = {
                                'dataset': dataset,
                                'category': dataset_category_map.get(dataset, 'unknown'),
                                'injection': inj,
                                'metric': val if isinstance(val, (int, float)) else None,
                                'baseline_metric': baseline_val if isinstance(baseline_val, (int, float)) else None,
                                'full_metric': full_val if isinstance(full_val, (int, float)) else None,
                            }
                            row.update(ref_vals)
                            rows.append(row)

                    # write CSV of raw per-dataset metrics
                    out_csv = os.path.join(RESULT_DIR, f'{p}_per_dataset_metrics_{metric_name}.csv')
                    # fieldnames: base fields + reference model few/full columns
                    ref_fields = []
                    for ref in trees + gnn_refs + [tabpfn]:
                        ref_fields.append(f"{ref}__few")
                        ref_fields.append(f"{ref}__full")
                    fieldnames = ['dataset','category','injection','metric','baseline_metric','full_metric'] + ref_fields
                    with open(out_csv, 'w', encoding='utf-8', newline='') as csvf:
                        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in rows:
                            # ensure all keys exist in the row (csv.DictWriter requires exact fields)
                            for k in fieldnames:
                                if k not in r:
                                    r[k] = ''
                            writer.writerow({k: r.get(k, '') for k in fieldnames})
                    print('Wrote', out_csv)

                    # sensitivity: for each eps, compute aggregated beats per opponent, per category
                    # We'll produce tall CSV with columns: eps,category,injection,opponent,beats,total
                    sens_rows = []
                    opponents = []
                    # opponents include primary baseline/full plus each ref few/full
                    opponents.append((f"{p}_few", f"{p}(ratio={small_ratio}, gnn_stage=none)"))
                    opponents.append((f"{p}_full", f"{p}(ratio={large_ratio}, gnn_stage=none)"))
                    for ref in trees + gnn_refs + [tabpfn]:
                        opponents.append((f"{ref}_few", f"{ref}(ratio={small_ratio}, gnn_stage=none)"))
                        opponents.append((f"{ref}_full", f"{ref}(ratio={large_ratio}, gnn_stage=none)"))
                    # Preserve the canonical opponent label ordering for later use
                    opponent_label_order = [t[0] for t in opponents]

                    for eps in eps_list:
                        # initialize counters per (category, injection, opponent)
                        agg = {}
                        for r in rows:
                            cat = r['category']
                            inj = r['injection']
                            # determine dataset task for direction
                            higher = True if dataset_task_map.get(r['dataset'],'binclass') in ('binclass','multiclass') else False
                            a = r['metric']
                            for opp_label, opp_comp in opponents:
                                # opponent metric value from row (we stored ref fields as keys like 'xgboost__few')
                                if opp_comp.startswith(f"{p}("):
                                    # primary baseline/full handled separately
                                    if 'ratio='+small_ratio in opp_comp:
                                        b = r.get('baseline_metric')
                                    else:
                                        b = r.get('full_metric')
                                else:
                                    # ref models stored as keys like 'xgboost__few' in row
                                    ref_key = opp_comp.split('(')[0]
                                    if 'ratio='+small_ratio in opp_comp:
                                        keyname = f"{ref_key}__few"
                                    else:
                                        keyname = f"{ref_key}__full"
                                    b = r.get(keyname)
                                key = (eps, cat, inj, opp_label)
                                agg.setdefault(key, {'beats_strict':0,'ties':0,'total':0})
                                better = is_better(a, b, eps, higher, args.eps_mode, args.eps_behavior)
                                if better is True:
                                    agg[key]['beats_strict'] += 1
                                elif better is None:
                                    agg[key]['ties'] += 1
                                agg[key]['total'] += 1
                        for (eps_v, cat, inj, opp), vals in agg.items():
                            beats_total = vals.get('beats_strict', 0) + vals.get('ties', 0)
                            sens_rows.append({'eps':eps_v,'category':cat,'injection':inj,'opponent':opp,'beats_strict':vals.get('beats_strict',0),'ties':vals.get('ties',0),'beats':beats_total,'total':vals['total']})

                    out_csv = os.path.join(RESULT_DIR, f'{p}_sensitivity_{metric_name}.csv')
                    # write tall CSV
                    with open(out_csv, 'w', encoding='utf-8', newline='') as csvf:
                        fieldnames = ['eps','category','injection','opponent','beats_strict','ties','beats','total']
                        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in sens_rows:
                            # ensure all keys exist
                            for k in fieldnames:
                                if k not in r:
                                    r[k] = 0
                            writer.writerow({k: r.get(k, 0) for k in fieldnames})
                    print('Wrote', out_csv)

                    # Optionally generate wide-format sensitivity tables (overall across categories)
                    if args.make_wide_sensitivity:
                        # sens_rows: list of dicts with eps,category,injection,opponent,beats_strict,ties,beats,total
                        for eps in eps_list:
                            # aggregate across categories for this eps
                            agg2 = {}  # (injection, opponent) -> {'beats':int,'total':int}
                            for r in sens_rows:
                                if float(r['eps']) != float(eps):
                                    continue
                                inj = r['injection']
                                opp = r['opponent']
                                key = (inj, opp)
                                agg2.setdefault(key, {'beats':0,'total':0})
                                agg2[key]['beats'] += int(r.get('beats', 0))
                                agg2[key]['total'] += int(r.get('total', 0))

                            # build wide table: rows=injection, cols=opponents
                            injections = sorted(set(k[0] for k in agg2.keys()))
                            # Keep opponent columns in the canonical order defined earlier,
                            # but only include those present in agg2 for this eps.
                            opponents_present = set(k[1] for k in agg2.keys())
                            opponents = [lbl for lbl in opponent_label_order if lbl in opponents_present]
                            wide_rows = []
                            for inj in injections:
                                row = {'injection': inj}
                                for opp in opponents:
                                    key = (inj, opp)
                                    vals = agg2.get(key, {'beats':0,'total':0})
                                    beats = vals['beats']
                                    total = vals['total']
                                    prop = (beats / total) if total > 0 else 0.0
                                    # store as "beats/total (prop)"
                                    row[f"{opp}_frac"] = f"{beats}/{total} ({prop:.3f})"
                                    row[f"{opp}_prop"] = f"{prop:.3f}"
                                wide_rows.append(row)

                            out_wide = os.path.join(RESULT_DIR, f'{p}_sensitivity_wide_eps{eps}_{metric_name}.csv')
                            # fieldnames: injection + for each opponent two fields
                            fieldnames = ['injection']
                            for opp in opponents:
                                fieldnames.append(f"{opp}_frac")
                                fieldnames.append(f"{opp}_prop")
                            with open(out_wide, 'w', encoding='utf-8', newline='') as wf:
                                writer = csv.DictWriter(wf, fieldnames=fieldnames)
                                writer.writeheader()
                                for r in wide_rows:
                                    # ensure all fields
                                    for k in fieldnames:
                                        if k not in r:
                                            r[k] = ''
                                    writer.writerow({k: r.get(k, '') for k in fieldnames})
                            print('Wrote wide sensitivity:', out_wide)

                        # If requested, produce a per-primary markdown that contains per-category wide tables
                        if args.wide_by_category:
                            md_path = os.path.join(RESULT_DIR, f'{p}_sensitivity_split_category_{metric_name}.md')
                            cats = sorted(set(r['category'] for r in sens_rows))
                            opponents = sorted(set(r['opponent'] for r in sens_rows))
                            with open(md_path, 'w', encoding='utf-8') as md:
                                md.write(f"# Sensitivity by category — primary: {p} (metric={metric_name})\n\n")
                                md.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
                                for cat in cats:
                                    md.write(f"## Category: {cat}\n\n")
                                    for eps in eps_list:
                                        md.write(f"### eps = {eps}\n\n")
                                        # opponents for this category/eps
                                        # Keep opponent column order consistent with canonical order
                                        opps_present = set(r['opponent'] for r in sens_rows if r['category']==cat and float(r['eps'])==float(eps))
                                        opps = [lbl for lbl in opponent_label_order if lbl in opps_present]
                                        injections = sorted(set(r['injection'] for r in sens_rows if r['category']==cat and float(r['eps'])==float(eps)))
                                        # header
                                        hdr = "| Injection "
                                        for opp in opps:
                                            hdr += f"| {opp} "
                                        hdr += "|\n"
                                        sep = "|---"
                                        sep += "|---" * len(opps)
                                        sep += "|\n"
                                        md.write(hdr)
                                        md.write(sep)
                                        for inj in injections:
                                            line = f"| {inj} "
                                            for opp in opps:
                                                # find row
                                                found = next((r for r in sens_rows if float(r['eps'])==float(eps) and r['category']==cat and r['injection']==inj and r['opponent']==opp), None)
                                                if found:
                                                    beats_strict = int(found.get('beats_strict',0))
                                                    ties = int(found.get('ties',0))
                                                    total = int(found.get('total',0))
                                                    beats = beats_strict + ties
                                                    prop = (beats/total) if total>0 else 0.0
                                                    cell = f"{beats_strict}/{ties}/{total} ({prop:.3f})"
                                                else:
                                                    cell = "-"
                                                line += f"| {cell} "
                                            line += "|\n"
                                            md.write(line)
                                        md.write("\n")
                            print('Wrote per-category markdown:', md_path)

                        # Also produce a single markdown that aggregates across all categories
                        # (one combined table per eps). This shows overall beats_strict/ties/total.
                        combined_md_path = os.path.join(RESULT_DIR, f'{p}_sensitivity_aggregate_category_{metric_name}.md')
                        # aggregate across categories
                        combined_agg = {}
                        for r in sens_rows:
                            key = (float(r['eps']), r['injection'], r['opponent'])
                            ag = combined_agg.setdefault(key, {'beats_strict':0, 'ties':0, 'total':0})
                            ag['beats_strict'] += int(r.get('beats_strict', 0))
                            ag['ties'] += int(r.get('ties', 0))
                            ag['total'] += int(r.get('total', 0))

                        # write combined md
                        with open(combined_md_path, 'w', encoding='utf-8') as cmd:
                            cmd.write(f"# Combined (all categories) sensitivity — primary: {p} (metric={metric_name})\n\n")
                            cmd.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
                            for eps in eps_list:
                                cmd.write(f"## eps = {eps}\n\n")
                                # determine injections and opponents present for this eps
                                inj_set = sorted(set(k[1] for k in combined_agg.keys() if k[0] == float(eps)))
                                opp_set = [lbl for lbl in opponent_label_order if any((float(k[0])==float(eps) and k[2]==lbl) for k in combined_agg.keys())]
                                # header
                                hdr = "| Injection "
                                for opp in opp_set:
                                    hdr += f"| {opp} "
                                hdr += "|\n"
                                sep = "|---"
                                sep += "|---" * len(opp_set)
                                sep += "|\n"
                                cmd.write(hdr)
                                cmd.write(sep)
                                for inj in inj_set:
                                    line = f"| {inj} "
                                    for opp in opp_set:
                                        key = (float(eps), inj, opp)
                                        vals = combined_agg.get(key)
                                        if vals:
                                            bs = vals['beats_strict']
                                            ties = vals['ties']
                                            total = vals['total']
                                            prop = (bs + ties) / total if total > 0 else 0.0
                                            cell = f"{bs}/{ties}/{total} ({prop:.3f})"
                                        else:
                                            cell = "-"
                                        line += f"| {cell} "
                                    line += "|\n"
                                    cmd.write(line)
                                cmd.write("\n")
                        print('Wrote combined-all-categories markdown:', combined_md_path)
                        combined_md_paths_all_primaries.append(combined_md_path)
                        # save aggregated numbers for group-level summary
                        group_agg_by_primary[p] = combined_agg

                    # end per-primary loop: after generating combined md per primary, create an overall file
                if combined_md_paths_all_primaries:
                    overall_path = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_category_split_primary_{metric_name}.md')
                    with open(overall_path, 'w', encoding='utf-8') as outf_all:
                        outf_all.write(f"# All primaries — combined all categories sensitivity (metric={metric_name})\n\n")
                        for p_md in combined_md_paths_all_primaries:
                            try:
                                text = open(p_md, 'r', encoding='utf-8').read()
                            except Exception:
                                text = ''
                            # write section separator and the content
                            outf_all.write(f"<!-- START {os.path.basename(p_md)} -->\n\n")
                            outf_all.write(text)
                            outf_all.write(f"\n<!-- END {os.path.basename(p_md)} -->\n\n")
                    print('Wrote overall combined markdown for all primaries:', overall_path)

                    # --- write group-level combined markdown for all primaries
                    # columns: Injection, {primary}_few, {primary}_full, tree_few, tree_full, gnn_few, gnn_full, tabpfn_few, tabpfn_full
                    group_md_path = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_category_split_primary_{metric_name}_group_level.md')
                    with open(group_md_path, 'w', encoding='utf-8') as gmd:
                        gmd.write(f"# All primaries — group-level combined (metric={metric_name})\n\n")
                        gmd.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
                        for p in primaries_list:
                            combined_agg = group_agg_by_primary.get(p)
                            if not combined_agg:
                                continue
                            gmd.write(f"## Primary: {p}\n\n")
                            for eps in eps_list:
                                gmd.write(f"### eps = {eps}\n\n")
                                hdr = "| Injection | {pf}_few | {pf}_full | tree_few | tree_full | gnn_few | gnn_full | tabpfn_few | tabpfn_full |\n".format(pf=p)
                                sep = "|---|---|---|---|---|---|---|---|---|\n"
                                gmd.write(hdr)
                                gmd.write(sep)
                                # determine injections present
                                inj_set = sorted(set(k[1] for k in combined_agg.keys() if float(k[0])==float(eps)))
                                for inj in inj_set:
                                    # helper to sum across opponent labels
                                    def get_vals_for_labels(labels):
                                        bs = 0
                                        ties = 0
                                        tot = 0
                                        for lbl in labels:
                                            key = (float(eps), inj, lbl)
                                            v = combined_agg.get(key)
                                            if v:
                                                bs += int(v.get('beats_strict', 0))
                                                ties += int(v.get('ties', 0))
                                                tot += int(v.get('total', 0))
                                        return bs, ties, tot

                                    # primary few/full labels
                                    pf_few_lbl = f"{p}_few"
                                    pf_full_lbl = f"{p}_full"
                                    # tree and gnn groups
                                    tree_few_lbls = ['xgboost_few','catboost_few','lightgbm_few']
                                    tree_full_lbls = ['xgboost_full','catboost_full','lightgbm_full']
                                    gnn_few_lbls = ['t2g-former_few','tabgnn_few']
                                    gnn_full_lbls = ['t2g-former_full','tabgnn_full']
                                    tabpfn_few_lbls = ['tabpfn_few']
                                    tabpfn_full_lbls = ['tabpfn_full']

                                    pf_few_bs, pf_few_ties, pf_few_tot = get_vals_for_labels([pf_few_lbl])
                                    pf_full_bs, pf_full_ties, pf_full_tot = get_vals_for_labels([pf_full_lbl])
                                    tree_few_bs, tree_few_ties, tree_few_tot = get_vals_for_labels(tree_few_lbls)
                                    tree_full_bs, tree_full_ties, tree_full_tot = get_vals_for_labels(tree_full_lbls)
                                    gnn_few_bs, gnn_few_ties, gnn_few_tot = get_vals_for_labels(gnn_few_lbls)
                                    gnn_full_bs, gnn_full_ties, gnn_full_tot = get_vals_for_labels(gnn_full_lbls)
                                    tabpfn_few_bs, tabpfn_few_ties, tabpfn_few_tot = get_vals_for_labels(tabpfn_few_lbls)
                                    tabpfn_full_bs, tabpfn_full_ties, tabpfn_full_tot = get_vals_for_labels(tabpfn_full_lbls)

                                    # format: beats_strict/ties/total (prop) where prop uses (beats_strict+ties)/total
                                    def fmt(bs, ties, tot):
                                        if tot <= 0:
                                            return '-'
                                        prop = (bs + ties) / tot if tot > 0 else 0.0
                                        return f"{bs}/{ties}/{tot} ({prop:.3f})"

                                    line = f"| {inj} | {fmt(pf_few_bs,pf_few_ties,pf_few_tot)} | {fmt(pf_full_bs,pf_full_ties,pf_full_tot)} | {fmt(tree_few_bs,tree_few_ties,tree_few_tot)} | {fmt(tree_full_bs,tree_full_ties,tree_full_tot)} | {fmt(gnn_few_bs,gnn_few_ties,gnn_few_tot)} | {fmt(gnn_full_bs,gnn_full_ties,gnn_full_tot)} | {fmt(tabpfn_few_bs,tabpfn_few_ties,tabpfn_few_tot)} | {fmt(tabpfn_full_bs,tabpfn_full_ties,tabpfn_full_tot)} |\n"
                                    gmd.write(line)
                                gmd.write("\n")
                    print('Wrote group-level combined markdown for all primaries:', group_md_path)

                    # --- aggregate per-category tables across all primaries (robust CSV-based approach)
                    try:
                        from collections import defaultdict
                        per_category_agg = {}  # category -> eps -> injection -> colname -> counters
                        # We'll read the tall per-primary CSVs `{primary}_sensitivity_{metric_name}.csv`
                        csv_pattern = f"_sensitivity_{metric_name}.csv"
                        csv_files = [os.path.join(RESULT_DIR, fn) for fn in os.listdir(RESULT_DIR) if fn.endswith(csv_pattern) and not fn.startswith('all_models')]
                        # canonical output columns: aggregate primary few/full + reference models
                        ref_cols = ['xgboost_few','xgboost_full','catboost_few','catboost_full','lightgbm_few','lightgbm_full',
                                    't2g-former_few','t2g-former_full','tabgnn_few','tabgnn_full','tabpfn_few','tabpfn_full']

                        for cf in csv_files:
                            try:
                                df = pd.read_csv(cf)
                            except Exception:
                                continue
                            # df columns: eps,category,injection,opponent,beats_strict,ties,beats,total
                            for _, r in df.iterrows():
                                try:
                                    eps_val = float(r['eps'])
                                except Exception:
                                    continue
                                cat = r['category']
                                inj = r['injection']
                                opp = r['opponent']
                                bs = int(r.get('beats_strict', 0))
                                ties = int(r.get('ties', 0))
                                tot = int(r.get('total', 0))
                                # normalize opponent column names: if opponent corresponds to a primary, map to primary_few/full
                                cname = opp
                                if cname.endswith('_few') or cname.endswith('_full'):
                                    base = cname.rsplit('_', 1)[0]
                                    if base in primaries_list:
                                        cname = 'primary_few' if cname.endswith('_few') else 'primary_full'

                                per_category_agg.setdefault(cat, {}).setdefault(eps_val, {}).setdefault(inj, {}).setdefault(cname, {'beats_strict':0,'ties':0,'total':0})
                                cell = per_category_agg[cat][eps_val][inj][cname]
                                cell['beats_strict'] += bs
                                cell['ties'] += ties
                                cell['total'] += tot

                        # write aggregated markdown
                        out_cat_md = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_primary_split_category_{metric_name}.md')
                        # compute dataset counts per category from dataset_category_map
                        counts_by_category = {}
                        try:
                            for _ds, _cat in dataset_category_map.items():
                                counts_by_category[_cat] = counts_by_category.get(_cat, 0) + 1
                        except Exception:
                            counts_by_category = {}

                        with open(out_cat_md, 'w', encoding='utf-8') as outm:
                            outm.write(f"# All models — sensitivity aggregated by category (metric={metric_name})\n\n")
                            outm.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
                            for cat, eps_dict in sorted(per_category_agg.items()):
                                cnt = counts_by_category.get(cat, 0)
                                outm.write(f"## Category: {cat} ({cnt})\n\n")
                                for eps in sorted(eps_dict.keys()):
                                    outm.write(f"### eps = {eps}\n\n")
                                    # header: Injection + primary_few/primary_full + refs
                                    hdr_cols = ['Injection','primary_few','primary_full'] + ref_cols
                                    hdr = "| " + " | ".join(hdr_cols) + " |\n"
                                    sep = "|" + "---|" * len(hdr_cols) + "\n"
                                    outm.write(hdr)
                                    outm.write(sep)
                                    inj_set = sorted(eps_dict[eps].keys())
                                    for inj in inj_set:
                                        line = f"| {inj} "
                                        for cname in hdr_cols[1:]:
                                            vals = eps_dict[eps][inj].get(cname)
                                            if not vals or vals.get('total',0) <= 0:
                                                cell = '-'
                                            else:
                                                bs = vals.get('beats_strict',0)
                                                ties = vals.get('ties',0)
                                                tot = vals.get('total',0)
                                                prop = (bs + ties)/tot if tot>0 else 0.0
                                                cell = f"{bs}/{ties}/{tot} ({prop:.3f})"
                                            line += f"| {cell} "
                                        line += "|\n"
                                        outm.write(line)
                                    outm.write("\n")
                        print('Wrote aggregated-by-category markdown:', out_cat_md)
                    except Exception as e:
                        print('Failed to aggregate per-category across primaries (csv-based):', e)

                    # --- produce group-level per-category aggregation from the combined-by-category file
                    try:
                        src_cat_md = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_primary_split_category_{metric_name}.md')
                        if os.path.exists(src_cat_md):
                            text = open(src_cat_md, 'r', encoding='utf-8').read()
                            # parse categories
                            cat_sections = re.split(r"\n## Category: ", text)
                            group_md_path = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_primary_split_category_{metric_name}_group_level.md')
                            with open(group_md_path, 'w', encoding='utf-8') as gfm:
                                gfm.write(f"# All models — group-level by category (metric={metric_name})\n\n")
                                gfm.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
                                # skip header chunk
                                for sec in cat_sections[1:]:
                                    lines = sec.splitlines()
                                    if not lines:
                                        continue
                                    cat_name = lines[0].strip()
                                    rest = "\n".join(lines[1:])
                                    gfm.write(f"## Category: {cat_name}\n\n")
                                    # split eps blocks
                                    eps_blocks = re.split(r"\n### eps = ", rest)
                                    for eb in eps_blocks[1:]:
                                        eb_lines = eb.splitlines()
                                        if not eb_lines:
                                            continue
                                        # first token is eps value
                                        eps_val_line = eb_lines[0].strip()
                                        try:
                                            eps_val = float(eps_val_line.split()[0])
                                        except Exception:
                                            try:
                                                eps_val = float(eps_val_line)
                                            except Exception:
                                                continue
                                        gfm.write(f"### eps = {eps_val}\n\n")
                                        # find table header
                                        table_start_idx = None
                                        for i, l in enumerate(eb_lines[1:], start=1):
                                            if l.strip().startswith('|') and 'Injection' in l:
                                                table_start_idx = i
                                                break
                                        if table_start_idx is None:
                                            continue
                                        header_line = eb_lines[table_start_idx].strip()
                                        cols = [c.strip() for c in header_line.strip('|').split('|')]
                                        # collect data rows until blank
                                        data_lines = []
                                        for dl in eb_lines[table_start_idx+2:]:
                                            if not dl.strip():
                                                break
                                            data_lines.append(dl)

                                        # columns we will produce
                                        out_cols = ['Injection','primary_few','primary_full','tree_few','tree_full','gnn_few','gnn_full','tabpfn_few','tabpfn_full']
                                        # write header
                                        hdr = "| " + " | ".join(out_cols) + " |\n"
                                        sep = "|" + "---|" * len(out_cols) + "\n"
                                        gfm.write(hdr)
                                        gfm.write(sep)

                                        # parse each data row and compute grouped sums
                                        for row in data_lines:
                                            parts = [p.strip() for p in row.strip('|').split('|')]
                                            if not parts:
                                                continue
                                            inj = parts[0]
                                            # map column name -> (bs,ties,tot)
                                            col_map = {}
                                            for ci, colname in enumerate(cols[1:], start=1):
                                                if ci >= len(parts):
                                                    continue
                                                cell = parts[ci]
                                                m = re.search(r"(\d+)\/(\d+)\/(\d+)", cell)
                                                if m:
                                                    bs = int(m.group(1)); ties = int(m.group(2)); tot = int(m.group(3))
                                                    col_map[colname] = {'bs':bs,'ties':ties,'tot':tot}
                                            # helper to sum labels
                                            def sum_labels(labels):
                                                s_bs = 0; s_ties = 0; s_tot = 0
                                                for lbl in labels:
                                                    v = col_map.get(lbl)
                                                    if v:
                                                        s_bs += v['bs']; s_ties += v['ties']; s_tot += v['tot']
                                                if s_tot == 0:
                                                    return None
                                                return (s_bs, s_ties, s_tot)

                                            # primary_few/full are present as columns named 'primary_few'/'primary_full'
                                            pf_few = sum_labels(['primary_few'])
                                            pf_full = sum_labels(['primary_full'])
                                            tree_few = sum_labels(['xgboost_few','catboost_few','lightgbm_few'])
                                            tree_full = sum_labels(['xgboost_full','catboost_full','lightgbm_full'])
                                            gnn_few = sum_labels(['t2g-former_few','tabgnn_few'])
                                            gnn_full = sum_labels(['t2g-former_full','tabgnn_full'])
                                            tabpfn_few = sum_labels(['tabpfn_few'])
                                            tabpfn_full = sum_labels(['tabpfn_full'])

                                            def fmt_cell(v):
                                                if not v:
                                                    return '-'
                                                bs, ties, tot = v
                                                prop = (bs + ties) / tot if tot>0 else 0.0
                                                return f"{bs}/{ties}/{tot} ({prop:.3f})"

                                            line = f"| {inj} | {fmt_cell(pf_few)} | {fmt_cell(pf_full)} | {fmt_cell(tree_few)} | {fmt_cell(tree_full)} | {fmt_cell(gnn_few)} | {fmt_cell(gnn_full)} | {fmt_cell(tabpfn_few)} | {fmt_cell(tabpfn_full)} |\n"
                                            gfm.write(line)
                                        gfm.write('\n')
                            print('Wrote group-level by-category markdown:', group_md_path)
                    except Exception as e:
                        print('Failed to create group-level by-category markdown:', e)

                    # --- New: aggregate (cross-category) and split by baseline (columns = primaries)
                    try:
                        # group_agg_by_primary holds per-primary combined aggregates: {primary: {(eps,inj,opp): {'beats_strict', 'ties','total'}}}
                        # Build canonical baseline list (exclude per-primary opponent labels like 'excelformer_few')
                        trees = ['xgboost','catboost','lightgbm']
                        gnn_refs = ['t2g-former','tabgnn']
                        ref_models = trees + gnn_refs + ['tabpfn']
                        baseline_list = []
                        # primary group baselines
                        baseline_list.extend(['primary_few','primary_full'])
                        # add per-ref few/full baselines (e.g., xgboost_few)
                        for r in ref_models:
                            baseline_list.append(f"{r}_few")
                            baseline_list.append(f"{r}_full")

                        out_path1 = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_category_split_baseline_{metric_name}_group_level.md')
                        with open(out_path1, 'w', encoding='utf-8') as out1:
                            out1.write(f"# All primaries — aggregate categories, split by baseline (metric={metric_name})\n\n")
                            out1.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
                            for baseline in baseline_list:
                                out1.write(f"## Baseline: {baseline}\n\n")
                                for eps in eps_list:
                                    out1.write(f"### eps = {eps}\n\n")
                                    # header: Injection + primaries
                                    hdr = "| Injection "
                                    for p in primaries_list:
                                        hdr += f"| {p} "
                                    hdr += "|\n"
                                    sep = "|---" + "|---" * len(primaries_list) + "|\n"
                                    out1.write(hdr)
                                    out1.write(sep)
                                    # use canonical injection stages order
                                    inj_order = ['columnwise','none','decoding','encoding','start','materialize']
                                    for inj in inj_order:
                                        line = f"| {inj} "
                                        for p in primaries_list:
                                            agg = group_agg_by_primary.get(p, {})
                                            # determine which opponent labels to sum for this baseline
                                            opp_labels = []
                                            if baseline == 'primary_few':
                                                opp_labels = [f"{p}_few"]
                                            elif baseline == 'primary_full':
                                                opp_labels = [f"{p}_full"]
                                            elif baseline.endswith('_few') and baseline.split('_')[0] in ref_models:
                                                opp_labels = [baseline]
                                            elif baseline.endswith('_full') and baseline.split('_')[0] in ref_models:
                                                opp_labels = [baseline]
                                            else:
                                                opp_labels = [baseline]

                                            # sum across opp_labels
                                            bs_sum = 0; ties_sum = 0; tot_sum = 0
                                            for opp in opp_labels:
                                                key = (float(eps), inj, opp)
                                                v = agg.get(key)
                                                if v:
                                                    bs_sum += int(v.get('beats_strict',0))
                                                    ties_sum += int(v.get('ties',0))
                                                    tot_sum += int(v.get('total',0))
                                            if tot_sum <= 0:
                                                cell = '-'
                                            else:
                                                prop = (bs_sum + ties_sum) / tot_sum if tot_sum>0 else 0.0
                                                cell = f"{bs_sum}/{ties_sum}/{tot_sum} ({prop:.3f})"
                                            line += f"| {cell} "
                                        line += "|\n"
                                        out1.write(line)
                                    out1.write("\n")
                        print('Wrote aggregate-category-split-by-baseline markdown:', out_path1)
                    except Exception as e:
                        print('Failed to write aggregate-category-split-baseline file:', e)

                    # --- New: aggregate (cross-primary) and split by baseline (columns = dataset categories)
                    try:
                        # Try candidate source files in order; if none exist build per_cat from per-primary CSVs
                        candidate_agg = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_primary_split_category_{metric_name}.md')
                        candidate_bycat = os.path.join(RESULT_DIR, f'all_models_sensitivity_by_category_{metric_name}.md')
                        per_cat = {}
                        if os.path.exists(candidate_agg):
                            text = open(candidate_agg, 'r', encoding='utf-8').read()
                            cat_sections = re.split(r"\n## Category: ", text)
                            for sec in cat_sections[1:]:
                                lines = sec.splitlines()
                                if not lines:
                                    continue
                                cat_line = lines[0].strip()
                                m = re.match(r"(.+?)\s*\((\d+)\)$", cat_line)
                                if m:
                                    cat_name = m.group(1).strip()
                                else:
                                    cat_name = cat_line
                                rest = "\n".join(lines[1:])
                                eps_blocks = re.split(r"\n### eps = ", rest)
                                for eb in eps_blocks[1:]:
                                    eb_lines = eb.splitlines()
                                    if not eb_lines:
                                        continue
                                    eps_val_line = eb_lines[0].strip()
                                    try:
                                        eps_val = float(eps_val_line.split()[0])
                                    except Exception:
                                        try:
                                            eps_val = float(eps_val_line)
                                        except Exception:
                                            continue
                                    table_start_idx = None
                                    for i, l in enumerate(eb_lines[1:], start=1):
                                        if l.strip().startswith('|') and 'Injection' in l:
                                            table_start_idx = i
                                            break
                                    if table_start_idx is None:
                                        continue
                                    header_line = eb_lines[table_start_idx].strip()
                                    cols = [c.strip() for c in header_line.strip('|').split('|')]
                                    data_lines = []
                                    for dl in eb_lines[table_start_idx+2:]:
                                        if not dl.strip():
                                            break
                                        data_lines.append(dl)
                                    for row in data_lines:
                                        parts = [p.strip() for p in row.strip('|').split('|')]
                                        if not parts:
                                            continue
                                        inj = parts[0]
                                        per_cat.setdefault(cat_name, {}).setdefault(float(eps_val), {}).setdefault(inj, {})
                                        for ci, colname in enumerate(cols[1:], start=1):
                                            if ci >= len(parts):
                                                continue
                                            cell = parts[ci]
                                            m = re.search(r"(\d+)\/(\d+)\/(\d+)", cell)
                                            if not m:
                                                continue
                                            bs = int(m.group(1)); ties = int(m.group(2)); tot = int(m.group(3))
                                            per_cat[cat_name][float(eps_val)][inj][colname] = {'bs':bs,'ties':ties,'tot':tot}
                        elif os.path.exists(candidate_bycat):
                            text = open(candidate_bycat, 'r', encoding='utf-8').read()
                            cat_sections = re.split(r"\n## Category: ", text)
                            for sec in cat_sections[1:]:
                                lines = sec.splitlines()
                                if not lines:
                                    continue
                                cat_name = lines[0].strip()
                                rest = "\n".join(lines[1:])
                                eps_blocks = re.split(r"\n### eps = ", rest)
                                for eb in eps_blocks[1:]:
                                    eb_lines = eb.splitlines()
                                    if not eb_lines:
                                        continue
                                    eps_val_line = eb_lines[0].strip()
                                    try:
                                        eps_val = float(eps_val_line.split()[0])
                                    except Exception:
                                        try:
                                            eps_val = float(eps_val_line)
                                        except Exception:
                                            continue
                                    table_start_idx = None
                                    for i, l in enumerate(eb_lines[1:], start=1):
                                        if l.strip().startswith('|') and 'Injection' in l:
                                            table_start_idx = i
                                            break
                                    if table_start_idx is None:
                                        continue
                                    header_line = eb_lines[table_start_idx].strip()
                                    cols = [c.strip() for c in header_line.strip('|').split('|')]
                                    data_lines = []
                                    for dl in eb_lines[table_start_idx+2:]:
                                        if not dl.strip():
                                            break
                                        data_lines.append(dl)
                                    for row in data_lines:
                                        parts = [p.strip() for p in row.strip('|').split('|')]
                                        if not parts:
                                            continue
                                        inj = parts[0]
                                        per_cat.setdefault(cat_name, {}).setdefault(float(eps_val), {}).setdefault(inj, {})
                                        for ci, colname in enumerate(cols[1:], start=1):
                                            if ci >= len(parts):
                                                continue
                                            cell = parts[ci]
                                            m = re.search(r"(\d+)\/(\d+)\/(\d+)", cell)
                                            if not m:
                                                continue
                                            bs = int(m.group(1)); ties = int(m.group(2)); tot = int(m.group(3))
                                            per_cat[cat_name][float(eps_val)][inj][colname] = {'bs':bs,'ties':ties,'tot':tot}
                        else:
                            # fallback: build per_cat directly from per-primary CSVs
                            per_cat = {}
                            csv_pattern = f"_sensitivity_{metric_name}.csv"
                            csv_files = [os.path.join(RESULT_DIR, fn) for fn in os.listdir(RESULT_DIR) if fn.endswith(csv_pattern) and not fn.startswith('all_models')]
                            for cf in csv_files:
                                try:
                                    df = pd.read_csv(cf)
                                except Exception:
                                    continue
                                for _, r in df.iterrows():
                                    try:
                                        eps_val = float(r['eps'])
                                    except Exception:
                                        continue
                                    cat = r['category']
                                    inj = r['injection']
                                    opp = r['opponent']
                                    bs = int(r.get('beats_strict', 0))
                                    ties = int(r.get('ties', 0))
                                    tot = int(r.get('total', 0))
                                    per_cat.setdefault(cat, {}).setdefault(eps_val, {}).setdefault(inj, {})
                                    per_cat[cat][eps_val][inj].setdefault(opp, {'bs':0,'ties':0,'tot':0})
                                    cell = per_cat[cat][eps_val][inj][opp]
                                    cell['bs'] += bs
                                    cell['ties'] += ties
                                    cell['tot'] += tot

                        # Now produce file per baseline where columns are categories
                        baseline_set = set()
                        for cat, epsd in per_cat.items():
                            for epsv, injd in epsd.items():
                                for inj, colsmap in injd.items():
                                    for cname in colsmap.keys():
                                        baseline_set.add(cname)
                        baseline_list = sorted(baseline_set)

                        out_path2 = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_primary_split_baseline_{metric_name}_group_level.md')
                        with open(out_path2, 'w', encoding='utf-8') as out2:
                            out2.write(f"# All primaries aggregated — split by baseline across dataset categories (metric={metric_name})\n\n")
                            out2.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
                            categories = sorted(per_cat.keys())
                            for baseline in baseline_list:
                                out2.write(f"## Baseline: {baseline}\n\n")
                                for eps in eps_list:
                                    out2.write(f"### eps = {eps}\n\n")
                                    # header: Injection + categories
                                    hdr = "| Injection "
                                    for cat in categories:
                                        hdr += f"| {cat} "
                                    hdr += "|\n"
                                    sep = "|---" + "|---" * len(categories) + "|\n"
                                    out2.write(hdr)
                                    out2.write(sep)
                                    inj_order = ['columnwise','none','decoding','encoding','start','materialize']
                                    for inj in inj_order:
                                        line = f"| {inj} "
                                        for cat in categories:
                                            cell = '-'
                                            try:
                                                cmap = per_cat.get(cat, {}).get(float(eps), {}).get(inj, {})
                                                v = cmap.get(baseline)
                                                if v and int(v.get('tot',0))>0:
                                                    bs = int(v.get('bs',0)); ties = int(v.get('ties',0)); tot = int(v.get('tot',0))
                                                    prop = (bs + ties) / tot if tot>0 else 0.0
                                                    cell = f"{bs}/{ties}/{tot} ({prop:.3f})"
                                            except Exception:
                                                cell = '-'
                                            line += f"| {cell} "
                                        line += "|\n"
                                        out2.write(line)
                                    out2.write("\n")
                        print('Wrote aggregate-primary-split-by-baseline markdown:', out_path2)
                    except Exception as e:
                        print('Failed to write aggregate-primary-split-baseline file:', e)

            # After computing for both metrics, run the comparison generator for test
            # (embed the logic from generate_comparisons_by_test.py)
            try:
                from pathlib import Path as _Path
                import math as _math
                # read primaries_all_datasets_ranking_by_test.md
                RESULT_DIR_P = os.path.join(BASE_DIR, 'gnn_injection_analysis', 'per_model_result')
                INPUT_MD = _Path(RESULT_DIR_P) / f'primaries_all_datasets_ranking_by_{metric_name}.md'
                OUT_MD = _Path(RESULT_DIR_P) / f'primaries_comparison_by_{metric_name}.md'
                eps = 0.1
                # groups
                primaries_list = ['excelformer', 'fttransformer', 'resnet', 'scarf', 'subtab',
                                'tabm', 'tabnet', 'tabtransformer', 'trompt', 'vime']
                trees = ['xgboost', 'catboost', 'lightgbm']
                gnn_refs = ['t2g-former', 'tabgnn']
                tabpfn = 'tabpfn'

                text = INPUT_MD.read_text(encoding='utf-8')
                section_re = re.compile(r"## Primary: (\S+)\n\n(.*?)\n(?=## Primary:|$)", re.S)
                row_re = re.compile(r"\|\s*\d+\s*\|\s*([^\|]+)\|\s*([0-9]+\.?[0-9]*)\s*\|")
                sections = {m.group(1): m.group(2) for m in section_re.finditer(text)}
                lookup = {}
                for p, body in sections.items():
                    d = {}
                    for r in row_re.finditer(body):
                        comp = r.group(1).strip()
                        avg = float(r.group(2))
                        d[comp] = avg
                    lookup[p] = d

                def find_avg(d, base, ratio, stage):
                    label = f"{base}(ratio={ratio}, gnn_stage={stage})"
                    return d.get(label)

                injection_stages = ['columnwise','none','decoding','encoding','start','materialize']
                small_ratio = '0.05/0.15/0.8'
                large_ratio = '0.8/0.15/0.05'

                out_lines = ["# Primaries comparison (based on test metrics)\n", "(eps for ties = {:.2f})\n\n".format(eps)]

                for p in primaries_list:
                    out_lines.append(f"## Primary: {p}\n\n")
                    out_lines.append("| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |")
                    out_lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
                    d = lookup.get(p, {})
                    baseline_avg = find_avg(d, p, small_ratio, 'none')
                    full_avg = find_avg(d, p, large_ratio, 'none')
                    tree_few = {t: find_avg(d, t, small_ratio, 'none') for t in trees}
                    tree_full = {t: find_avg(d, t, large_ratio, 'none') for t in trees}
                    gnn_few = {g: find_avg(d, g, small_ratio, 'none') for g in gnn_refs}
                    gnn_full = {g: find_avg(d, g, large_ratio, 'none') for g in gnn_refs}
                    tabpfn_few = find_avg(d, tabpfn, small_ratio, 'none')
                    tabpfn_full = find_avg(d, tabpfn, large_ratio, 'none')

                    rows = []
                    for inj in injection_stages:
                        avg = None
                        if inj == 'none':
                            avg = baseline_avg
                        else:
                            avg = find_avg(d, p, small_ratio, inj)
                        avg_val = avg if avg is not None else float('inf')
                        avg_str = f"{avg:.2f}" if avg is not None else 'N/A'
                        if avg is None or baseline_avg is None:
                            beat_baseline = 'N/A'
                        else:
                            if inj == 'none':
                                beat_baseline = 'No'
                            else:
                                if avg < baseline_avg - 1e-12:
                                    beat_baseline = 'Yes'
                                elif abs(avg - baseline_avg) <= eps:
                                    beat_baseline = 'Yes (tie)'
                                else:
                                    beat_baseline = 'No'
                        if avg is None or full_avg is None:
                            beat_full = 'N/A'
                        else:
                            diff = full_avg - avg
                            if diff > eps:
                                beat_full = 'Yes'
                            elif abs(diff) <= eps:
                                beat_full = 'Yes (tie)'
                            else:
                                beat_full = 'No'
                        tree_few_beats = 0
                        for t, val in tree_few.items():
                            if avg is None or val is None:
                                continue
                            if avg < val - eps:
                                tree_few_beats += 1
                        tree_full_beats = 0
                        for t, val in tree_full.items():
                            if avg is None or val is None:
                                continue
                            if avg < val - eps:
                                tree_full_beats += 1
                        gnn_few_beats = 0
                        for g, val in gnn_few.items():
                            if avg is None or val is None:
                                continue
                            if avg < val - eps:
                                gnn_few_beats += 1
                        gnn_full_beats = 0
                        for g, val in gnn_full.items():
                            if avg is None or val is None:
                                continue
                            if avg < val - eps:
                                gnn_full_beats += 1
                        tabpfn_few_beat = 'N/A'
                        if avg is not None and tabpfn_few is not None:
                            if avg < tabpfn_few - eps:
                                tabpfn_few_beat = 'Yes'
                            elif abs(avg - tabpfn_few) <= eps:
                                tabpfn_few_beat = 'Yes (tie)'
                            else:
                                tabpfn_few_beat = 'No'
                        tabpfn_full_beat = 'N/A'
                        if avg is not None and tabpfn_full is not None:
                            if avg < tabpfn_full - eps:
                                tabpfn_full_beat = 'Yes'
                            elif abs(avg - tabpfn_full) <= eps:
                                tabpfn_full_beat = 'Yes (tie)'
                            else:
                                tabpfn_full_beat = 'No'

                        row_line = f"| {inj} | {avg_str} | {beat_baseline} | {beat_full} | {tree_few_beats}/3 | {tree_full_beats}/3 | {gnn_few_beats}/2 | {gnn_full_beats}/2 | {tabpfn_few_beat} | {tabpfn_full_beat} |"
                        rows.append((avg_val, row_line))

                    # sort rows by avg_val (N/A last) and append
                    rows.sort(key=lambda x: x[0])
                    for _, line in rows:
                        out_lines.append(line)
                    out_lines.append("\n")

                OUT_MD.write_text('\n'.join(out_lines), encoding='utf-8')
                print('Wrote', OUT_MD)
            except Exception as e:
                print('Comparison generation failed:', e)
    else:
        print(f"No summary_results directory at {SUMMARY_DIR}; skipping detailed aggregation.")
