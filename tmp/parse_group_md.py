import re
from pathlib import Path

FILES = [
    'gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_primary_split_baseline_test_group_level.md',
    'gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_primary_split_baseline_val_group_level.md',
    'gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_category_split_baseline_test_group_level.md',
    'gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_category_split_baseline_val_group_level.md',
]

ROOT = Path('')

ratio_re = re.compile(r"\(([0-9]*\.?[0-9]+)\)\s*$")


def parse_table(lines):
    # lines: list of lines for one table, starting with header row (| Injection | ... |)
    header_line = None
    rows = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith('|'):
            if header_line is None:
                header_line = ln.strip()
            else:
                # skip the separator line if present
                if set(ln.strip()) <= set('|-: '):
                    continue
                rows.append(ln.strip())
    if header_line is None:
        return None
    headers = [h.strip() for h in header_line.strip().split('|')[1:-1]]
    parsed = []
    for r in rows:
        cells = [c.strip() for c in r.split('|')[1:-1]]
        parsed.append(cells)
    return headers, parsed


def extract_ratio(cell):
    # find last parenthesized number
    m = ratio_re.search(cell)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None


for f in FILES:
    p = ROOT / f
    if not p.exists():
        print(f"MISSING: {f}")
        continue
    txt = p.read_text()
    # split by Baseline sections
    sections = re.split(r"\n## Baseline:", txt)
    title = sections[0].strip().splitlines()[0] if sections else ''
    print('\n' + '='*60)
    print(f"File: {f}\n{title}")
    for sec in sections[1:]:
        # sec starts with ' baseline_name\n\n### eps...\n\n| Injection | ...'
        lines = sec.splitlines()
        baseline_name = lines[0].strip()
        # find the table start index
        table_lines = [ln for ln in lines if ln.strip().startswith('|')]
        parsed = parse_table(table_lines)
        if not parsed:
            continue
        headers, rows = parsed
        # first column is Injection
        cats = headers[1:]
        # build dict: injection -> list of ratios in same order as cats
        inj_map = {}
        for r in rows:
            inj = r[0]
            ratios = [extract_ratio(c) for c in r[1:]]
            inj_map[inj] = ratios
        # compute best injection per category
        best_per_cat = []
        for ci, cat in enumerate(cats):
            best = None
            best_val = -1
            for inj, arr in inj_map.items():
                val = arr[ci]
                if val is None:
                    continue
                if val > best_val:
                    best_val = val
                    best = inj
            best_per_cat.append((cat, best, best_val))
        # determine none row values for diff
        none_vals = inj_map.get('none')
        print('\nBaseline: %s   (categories: %d)' % (baseline_name, len(cats)))
        print('Top injection per category (cat | best | value | none_value | delta)')
        for cat, best, best_val in best_per_cat:
            none_v = None
            if none_vals is not None:
                idx = cats.index(cat)
                none_v = none_vals[idx]
            delta = None if none_v is None or best_val is None else (best_val - none_v)
            print(f"- {cat} | {best} | {best_val} | {none_v} | {delta}")
        # also compute how many categories each injection is best across
        counts = {}
        for cat, best, best_val in best_per_cat:
            counts[best] = counts.get(best, 0) + 1
        print('\nCounts of best-in-category for this baseline:')
        for inj, c in sorted(counts.items(), key=lambda x:-x[1]):
            print(f"- {inj}: {c}")

print('\nDone.')
