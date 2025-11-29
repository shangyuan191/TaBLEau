import re
from pathlib import Path
p = Path('gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_primary_split_category_test_group_level.md')
text = p.read_text()
ratio_re = re.compile(r"\(([0-9]*\.?[0-9]+)\)")

# parse categories: split by '## Category:'
cats = re.split(r"\n## Category:", text)[1:]

cols = None
# columns stats: {col: {inj: [ratios per category]}}
cols_stats = {}
for c in cats:
    lines = c.splitlines()
    # first line contains category name
    catname = lines[0].strip()
    # find table lines starting with |
    table_lines = [ln for ln in lines if ln.strip().startswith('|')]
    if not table_lines:
        continue
    header = table_lines[0]
    headers = [h.strip() for h in header.split('|')[1:-1]]
    if cols is None:
        cols = headers[1:]
        for col in cols:
            cols_stats[col] = {}
    # rows: skip separator line
    row_lines = []
    for ln in table_lines[1:]:
        if set(ln.strip()) <= set('|-: '):
            continue
        row_lines.append(ln)
    injection_names = []
    cell_matrix = []
    for r in row_lines:
        cells = [c.strip() for c in r.split('|')[1:-1]]
        inj = cells[0]
        injection_names.append(inj)
        nums = []
        for cell in cells[1:]:
            m = ratio_re.search(cell)
            if m:
                nums.append(float(m.group(1)))
            else:
                nums.append(None)
        cell_matrix.append((inj, nums))
    # populate cols_stats per column index
    for col_idx, col in enumerate(cols):
        for inj, nums in cell_matrix:
            val = nums[col_idx]
            if inj not in cols_stats[col]:
                cols_stats[col][inj] = []
            cols_stats[col][inj].append(val)

# compute summary
import statistics
print('Columns:', cols)
print()
summary = {}
for col in cols:
    data = cols_stats[col]
    # per category best counts
    n_cats = len(next(iter(data.values())))
    best_counts = {inj:0 for inj in data.keys()}
    mean_ratios = {}
    for inj, arr in data.items():
        # compute mean over non-None
        vals = [v for v in arr if v is not None]
        mean_ratios[inj] = statistics.mean(vals) if vals else None
    # for each category index find best injection
    for i in range(n_cats):
        best = None
        best_val = -1
        for inj, arr in data.items():
            v = arr[i]
            if v is None:
                continue
            if v > best_val:
                best_val = v
                best = inj
        if best is not None:
            best_counts[best] += 1
    # compute mean delta vs none per inj
    none_arr = data.get('none')
    mean_delta = {}
    for inj, arr in data.items():
        deltas = []
        for a, b in zip(arr, none_arr):
            if a is None or b is None:
                continue
            deltas.append(a-b)
        mean_delta[inj] = statistics.mean(deltas) if deltas else None
    summary[col] = {'n_cats': n_cats, 'best_counts': best_counts, 'mean_ratios': mean_ratios, 'mean_delta_vs_none': mean_delta}

# print readable summary
for col, s in summary.items():
    print('===', col, ' (n_cats=', s['n_cats'], ')')
    print('Top counts (best per category):')
    for inj, cnt in sorted(s['best_counts'].items(), key=lambda x:-x[1]):
        print(f'  {inj:12} : {cnt}')
    print('Mean ratio per injection:')
    for inj, mr in sorted(s['mean_ratios'].items(), key=lambda x: (x[1] is None, -x[1] if x[1] is not None else 0)):
        print(f'  {inj:12} : {mr:.3f}' if mr is not None else f'  {inj:12} : None')
    print('Mean delta vs none:')
    for inj, md in sorted(s['mean_delta_vs_none'].items(), key=lambda x: (x[1] is None, -x[1] if x[1] is not None else 0)):
        print(f'  {inj:12} : {md:.3f}' if md is not None else f'  {inj:12} : None')
    print()

# Also summarize across columns: which injections perform well overall
overall = {}
for col in cols:
    for inj, cnt in summary[col]['best_counts'].items():
        overall[inj] = overall.get(inj,0) + cnt
print('Overall best-counts across all columns:')
for inj, cnt in sorted(overall.items(), key=lambda x:-x[1]):
    print(f'  {inj:12} : {cnt}')

