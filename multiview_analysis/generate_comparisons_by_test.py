import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
metric_name = "test"
RESULT_DIR = BASE / 'gnn_injection_analysis' / 'per_model_result'
INPUT_MD = RESULT_DIR / f'primaries_all_datasets_ranking_by_{metric_name}.md'
OUT_MD = RESULT_DIR / f'primaries_comparison_by_{metric_name}.md'

eps = 0.1  # tolerance for tie for non-few-shot comparisons

# groups
primaries = ['excelformer', 'fttransformer', 'resnet', 'scarf', 'subtab',
             'tabm', 'tabnet', 'tabtransformer', 'trompt', 'vime']
trees = ['xgboost', 'catboost', 'lightgbm']
gnn_refs = ['t2g-former', 'tabgnn']
tabpfn = 'tabpfn'

# parse input
text = INPUT_MD.read_text(encoding='utf-8')
section_re = re.compile(r"## Primary: (\S+)\n\n(.*?)\n(?=## Primary:|$)", re.S)
row_re = re.compile(r"\|\s*\d+\s*\|\s*([^\|]+)\|\s*([0-9]+\.?[0-9]*)\s*\|")

sections = {m.group(1): m.group(2) for m in section_re.finditer(text)}

# build lookup: primary -> competitor_label -> avg
lookup = {}
for p, body in sections.items():
    d = {}
    for r in row_re.finditer(body):
        comp = r.group(1).strip()
        avg = float(r.group(2))
        d[comp] = avg
    lookup[p] = d

# helper to find avg for a given base, ratio, stage
def find_avg(d, base, ratio, stage):
    label = f"{base}(ratio={ratio}, gnn_stage={stage})"
    return d.get(label)

# injection stages to report (order)
injection_stages = ['columnwise','none','decoding','encoding','start','materialize']
small_ratio = '0.05/0.15/0.8'
large_ratio = '0.8/0.15/0.05'

out_lines = ["# Primaries comparison (based on test metrics)\n", "(eps for ties = {:.2f})\n\n".format(eps)]

for p in primaries:
    out_lines.append(f"## Primary: {p}\n\n")
    out_lines.append("| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |")
    out_lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    d = lookup.get(p, {})
    # find baseline and full-non-gnn
    baseline_avg = find_avg(d, p, small_ratio, 'none')
    full_avg = find_avg(d, p, large_ratio, 'none')
    # tree avgs
    tree_few = {t: find_avg(d, t, small_ratio, 'none') for t in trees}
    tree_full = {t: find_avg(d, t, large_ratio, 'none') for t in trees}
    # gnn refs
    gnn_few = {g: find_avg(d, g, small_ratio, 'none') for g in gnn_refs}
    gnn_full = {g: find_avg(d, g, large_ratio, 'none') for g in gnn_refs}
    # tabpfn
    tabpfn_few = find_avg(d, tabpfn, small_ratio, 'none')
    tabpfn_full = find_avg(d, tabpfn, large_ratio, 'none')

    # collect rows then sort by numeric avg (None -> place last)
    rows = []
    for inj in injection_stages:
        # for 'none' we use small_ratio none as few-shot baseline entry
        avg = find_avg(d, p, small_ratio if inj!='none' else small_ratio, inj) if inj!='none' else baseline_avg
        avg_val = avg if avg is not None else float('inf')
        avg_str = f"{avg:.2f}" if avg is not None else 'N/A'
        # beats few-shot baseline: strict
        if avg is None or baseline_avg is None:
            beat_baseline = 'N/A'
        else:
            if inj == 'none':
                beat_baseline = 'No (tie, few-shot strict)'
            else:
                if avg < baseline_avg - 1e-12:
                    beat_baseline = 'Yes'
                elif abs(avg - baseline_avg) <= eps:
                    beat_baseline = 'No (tie, few-shot strict)'
                else:
                    beat_baseline = 'No'
        # beats full-non-gnn
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
        # trees few-shot count
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
        # gnn few
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
        # tabpfn
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
