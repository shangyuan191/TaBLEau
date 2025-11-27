#!/usr/bin/env python3
import os
import re
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "gnn_injection_analysis", "per_model_result")


def parse_baseline_md(path):
    """Parse a baseline-split markdown into nested dict:
    data[baseline][eps][inj][colname] = {'bs':int,'ties':int,'tot':int}
    Also return ordered list of columns for each eps/block (but we'll collect union).
    """
    text = open(path, 'r', encoding='utf-8').read()
    sections = re.split(r"\n## Baseline: ", text)
    data = {}
    for sec in sections[1:]:
        lines = sec.splitlines()
        if not lines:
            continue
        baseline = lines[0].strip()
        rest = "\n".join(lines[1:])
        eps_blocks = re.split(r"\n### eps = ", rest)
        data.setdefault(baseline, {})
        for eb in eps_blocks[1:]:
            eb_lines = eb.splitlines()
            if not eb_lines:
                continue
            eps_line = eb_lines[0].strip()
            try:
                eps = float(eps_line.split()[0])
            except Exception:
                try:
                    eps = float(eps_line)
                except Exception:
                    continue
            # find table header
            table_start = None
            for i, l in enumerate(eb_lines[1:], start=1):
                if l.strip().startswith('|') and 'Injection' in l:
                    table_start = i
                    break
            if table_start is None:
                continue
            header = eb_lines[table_start].strip()
            cols = [c.strip() for c in header.strip('|').split('|')]
            colnames = cols[1:]
            # parse rows
            idx = table_start+2
            while idx < len(eb_lines):
                row = eb_lines[idx]
                if not row.strip():
                    break
                parts = [p.strip() for p in row.strip('|').split('|')]
                if not parts:
                    idx += 1
                    continue
                inj = parts[0]
                data[baseline].setdefault(eps, {}).setdefault(inj, {})
                for ci, col in enumerate(colnames, start=1):
                    if ci >= len(parts):
                        continue
                    cell = parts[ci]
                    m = re.search(r"(\d+)\/(\d+)\/(\d+)", cell)
                    if not m:
                        continue
                    bs = int(m.group(1)); ties = int(m.group(2)); tot = int(m.group(3))
                    data[baseline][eps][inj][col] = {'bs':bs,'ties':ties,'tot':tot}
                idx += 1
    return data


def write_grouped_md(src_data, out_path, grouped_baselines, eps_list, columns_order, title):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(title + '\n\n')
        f.write(f"eps values: {', '.join(str(e) for e in eps_list)}\n\n")
        for g in grouped_baselines:
            f.write(f"## Baseline: {g}\n\n")
            for eps in eps_list:
                f.write(f"### eps = {eps}\n\n")
                hdr = "| Injection "
                for c in columns_order:
                    hdr += f"| {c} "
                hdr += "|\n"
                sep = "|---" + "|---" * len(columns_order) + "|\n"
                f.write(hdr)
                f.write(sep)
                inj_order = ['columnwise','none','decoding','encoding','start','materialize']
                for inj in inj_order:
                    line = f"| {inj} "
                    for col in columns_order:
                        cell = '-'
                        try:
                            # src_data: baseline -> eps -> inj -> col
                            entries = src_data
                            # entries will be mapping of original baselines to their data dicts
                            # grouped baseline g corresponds to list of original baselines
                            orig_labels = grouped_baselines[g]
                            bs_sum = ties_sum = tot_sum = 0
                            for lbl in orig_labels:
                                v = entries.get(lbl, {}).get(eps, {}).get(inj, {}).get(col)
                                if v:
                                    bs_sum += int(v.get('bs',0))
                                    ties_sum += int(v.get('ties',0))
                                    tot_sum += int(v.get('tot',0))
                            if tot_sum and tot_sum>0:
                                prop = (bs_sum + ties_sum) / tot_sum if tot_sum>0 else 0.0
                                cell = f"{bs_sum}/{ties_sum}/{tot_sum} ({prop:.3f})"
                        except Exception:
                            cell = '-'
                        line += f"| {cell} "
                    line += "|\n"
                    f.write(line)
                f.write("\n")
    return out_path


if __name__ == '__main__':
    metrics = ['test','val']
    # grouped baseline mapping: grouped_name -> list of original labels
    trees = ['xgboost','catboost','lightgbm']
    gnn = ['t2g-former','tabgnn']

    for metric in metrics:
        # 1) handle aggregate_category_split_baseline_{metric}.md -> produce group_level
        src1 = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_category_split_baseline_{metric}.md')
        if os.path.exists(src1):
            data = parse_baseline_md(src1)
            # collect eps list and columns ordering from first baseline
            all_eps = sorted({eps for b in data for eps in data[b].keys()})
            # columns: primaries in header; try to find from any section
            sample_baseline = next(iter(data))
            sample_eps = next(iter(data[sample_baseline]))
            sample_inj = next(iter(data[sample_baseline][sample_eps]))
            cols = sorted({c for b in data for eps in data[b] for inj in data[b][eps] for c in data[b][eps][inj].keys()})
            # define grouped mapping
            grouped_map = {
                'primary_few':[ 'primary_few' ],
                'primary_full':[ 'primary_full' ],
                'tree_few':[f"{t}_few" for t in trees],
                'tree_full':[f"{t}_full" for t in trees],
                'gnn_few':[f"{g}_few" for g in gnn],
                'gnn_full':[f"{g}_full" for g in gnn],
                'tabpfn_few':['tabpfn_few'],
                'tabpfn_full':['tabpfn_full']
            }
            out1 = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_category_split_baseline_{metric}_group_level.md')
            write_grouped_md(data, out1, grouped_map, all_eps, cols, f"# All primaries — aggregate categories, split by grouped baseline (metric={metric})")
            print('Wrote', out1)

        # 2) handle aggregate_primary_split_baseline_{metric}.md -> produce group_level (columns = categories)
        src2 = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_primary_split_baseline_{metric}.md')
        if os.path.exists(src2):
            data2 = parse_baseline_md(src2)
            all_eps2 = sorted({eps for b in data2 for eps in data2[b].keys()})
            cols2 = sorted({c for b in data2 for eps in data2[b] for inj in data2[b][eps] for c in data2[b][eps][inj].keys()})
            grouped_map2 = {
                'primary_few':[ 'primary_few' ],
                'primary_full':[ 'primary_full' ],
                'tree_few':[f"{t}_few" for t in trees],
                'tree_full':[f"{t}_full" for t in trees],
                'gnn_few':[f"{g}_few" for g in gnn],
                'gnn_full':[f"{g}_full" for g in gnn],
                'tabpfn_few':['tabpfn_few'],
                'tabpfn_full':['tabpfn_full']
            }
            out2 = os.path.join(RESULT_DIR, f'all_models_sensitivity_aggregate_primary_split_baseline_{metric}_group_level.md')
            write_grouped_md(data2, out2, grouped_map2, all_eps2, cols2, f"# All primaries aggregated — split by grouped baseline across dataset categories (metric={metric})")
            print('Wrote', out2)

    print('Done')
