#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).parent
models = [
    'excelformer', 'fttransformer', 'resnet', 'scarf', 'subtab',
    'tabm', 'tabnet', 'tabtransformer', 'trompt', 'vime'
]
files = {m: ROOT / f"{m}_gnn_enhancement_summary.md" for m in models}

# Targets (columns) expected in the tables (order in many files)
cols = [
    'injection', 'avg_rank',
    'beats_few_shot_non_gnn', 'beats_full_non_gnn',
    'beats_few_shot_tree', 'beats_full_tree',
    'beats_few_shot_gnn', 'beats_full_gnn',
    'beats_few_shot_tabpfn', 'beats_full_tabpfn'
]

# helper predicates

def is_yes_cell(cell):
    if cell is None:
        return False
    c = cell.lower()
    if 'yes' in c:
        return True
    # if fraction like 1/3, 2/3 etc -> yes if numerator > 0
    m = re.search(r"(\d+)\s*/\s*(\d+)", cell)
    if m:
        return int(m.group(1)) > 0
    return False


def parse_table_block(lines):
    # lines: header row + separator + data rows
    # return list of dict rows mapping col->cell
    header = lines[0]
    # split header into columns count
    headers = [h.strip().lower() for h in header.strip().strip('|').split('|')]
    # normalize some common variations to our target keys
    # find mapping from column index to our expected field
    mapping = {}
    for i,h in enumerate(headers):
        if 'injection' in h:
            mapping[i] = 'injection'
        elif 'avg' in h:
            mapping[i] = 'avg_rank'
        elif 'few-shot-non-gnn' in h or 'few-shot non-gnn' in h or 'few-shot' in h and 'non-gnn' in h:
            mapping[i] = 'beats_few_shot_non_gnn'
        elif 'full-non-gnn' in h or 'full non-gnn' in h:
            mapping[i] = 'beats_full_non_gnn'
        elif 'few-shot tree' in h or 'few-shot tree' in h or 'few-shot tree' in h:
            mapping[i] = 'beats_few_shot_tree'
        elif 'full tree' in h:
            mapping[i] = 'beats_full_tree'
        elif 'few-shot gnn' in h or 'few-shot gnn' in h:
            mapping[i] = 'beats_few_shot_gnn'
        elif 'full gnn' in h:
            mapping[i] = 'beats_full_gnn'
        elif 'few-shot tabpfn' in h or 'few-shot tabpfn' in h:
            mapping[i] = 'beats_few_shot_tabpfn'
        elif 'full tabpfn' in h or 'full tabpfn' in h:
            mapping[i] = 'beats_full_tabpfn'
        else:
            # fallback: map by position later
            mapping[i] = headers[i]
    rows = []
    for r in lines[2:]:
        if not r.strip().startswith('|'):
            continue
        parts = [p.strip() for p in r.strip().strip('|').split('|')]
        if len(parts) < 3:
            continue
        row = {k: None for k in cols}
        for i,cell in enumerate(parts):
            if i in mapping:
                key = mapping[i]
                # normalize key to our column names
                if key in cols:
                    row[key] = cell
                else:
                    # try to map common variations
                    lk = key.lower()
                    if 'few-shot' in lk and 'non' in lk:
                        row['beats_few_shot_non_gnn'] = cell
                    elif 'full' in lk and 'non' in lk:
                        row['beats_full_non_gnn'] = cell
                    elif 'tree' in lk and 'few' in lk:
                        row['beats_few_shot_tree'] = cell
                    elif 'tree' in lk and 'full' in lk:
                        row['beats_full_tree'] = cell
                    elif 'gnn' in lk and 'few' in lk:
                        row['beats_few_shot_gnn'] = cell
                    elif 'gnn' in lk and 'full' in lk:
                        row['beats_full_gnn'] = cell
                    elif 'tabpfn' in lk and 'few' in lk:
                        row['beats_few_shot_tabpfn'] = cell
                    elif 'tabpfn' in lk and 'full' in lk:
                        row['beats_full_tabpfn'] = cell
                    elif 'avg' in lk:
                        row['avg_rank'] = cell
                    elif 'injection' in lk:
                        row['injection'] = cell
        # normalize injection name
        if row['injection']:
            inj = re.sub(r"\(.*\)", "", row['injection']).strip()
            row['injection'] = inj
        rows.append(row)
    return rows


def extract_model_data(path):
    s = path.read_text(encoding='utf8')
    lines = s.splitlines()
    data = {}
    i = 0
    # Accept headings like '## Category: name' or '## large_datasets+binclass+numerical ...'
    cat_heading_re = re.compile(r"^##\s*(?:Category:\s*)?(.+)$")
    while i < len(lines):
        line = lines[i]
        m = cat_heading_re.match(line)
        if m:
            candidate = m.group(1).strip()
            # we only treat this as a dataset-category heading when it looks like the known pattern
            # pattern: contains at least two '+' separators (e.g., 'large_datasets+binclass+numerical')
            if candidate.count('+') >= 2:
                cat = candidate
            else:
                # not a category heading, skip
                i += 1
                continue

            # find next table header: flexible search for a pipe table that mentions 'Injection'
            j = i + 1
            while j < len(lines) and not (lines[j].strip().startswith('|') and 'injection' in lines[j].lower()):
                j += 1
            if j >= len(lines):
                i = j
                continue
            # gather table lines (header + separator + rows until blank line or next heading)
            tbl_lines = []
            k = j
            while k < len(lines) and (lines[k].strip().startswith('|') or lines[k].strip() == ''):
                if lines[k].strip() == '' and tbl_lines:
                    break
                tbl_lines.append(lines[k])
                k += 1
            try:
                rows = parse_table_block(tbl_lines)
            except Exception:
                rows = []
            data[cat] = rows
            i = k
        else:
            i += 1
    return data

# collect per-model parsed data
all_data = {}
for m,f in files.items():
    if not f.exists():
        print('Missing', f)
        continue
    all_data[m] = extract_model_data(f)

# determine set of categories to aggregate (intersection of models)
cats = set()
for m,d in all_data.items():
    cats.update(d.keys())
cats = sorted(cats)

# injections of interest
injections = ['start','materialize','encoding','columnwise','decoding','none']

# prepare aggregation structure
from collections import defaultdict

report = []
for cat in cats:
    # for each injection, count models with yes for each of 8 metrics
    agg = {inj: {c:0 for c in cols[2:]} for inj in injections}
    model_count = 0
    for m in models:
        model_tables = all_data.get(m,{})
        if cat not in model_tables:
            continue
        model_count += 1
        rows = model_tables[cat]
        # map injection names to rows
        inj_map = {}
        for r in rows:
            name = r.get('injection')
            if not name:
                continue
            key = name.strip().lower()
            inj_map[key] = r
        # for each injection of interest, find matching row or skip
        for inj in injections:
            row = None
            if inj == 'none':
                # few-shot baseline label sometimes 'none (few-shot baseline)' or 'none (few)'
                candidates = [k for k in inj_map.keys() if k.startswith('none')]
                if candidates:
                    row = inj_map[candidates[0]]
            else:
                # find key that contains inj
                candidates = [k for k in inj_map.keys() if inj in k]
                if candidates:
                    row = inj_map[candidates[0]]
            if row:
                # for each metric col, check if yes
                for col in cols[2:]:
                    cell = row.get(col)
                    if is_yes_cell(cell):
                        agg[inj][col] += 1
    report.append((cat, model_count, agg))

# print markdown report
out = []
out.append('# Cross-model GNN injection aggregation (10 models)\n')
out.append('This report counts, for each dataset category and each GNN injection stage, how many of the parsed models (out of the models that include that category) had the injection "beat" each reference group.\n')
for cat,model_count,agg in report:
    out.append(f'## Category: {cat} ({model_count} models parsed)\n')
    out.append('| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for inj in injections:
        row = agg[inj]
        cells = []
        for col in cols[2:]:
            cells.append(f"{row[col]}/{model_count}, {row[col]*100/model_count:.0f}%")
        out.append(f"| {inj} | {' | '.join(cells)} |")
    out.append('\n')

out_text = '\n'.join(out)
# write outputs
ROOT.joinpath('gnn_injection_crossmodel_aggregation.md').write_text(out_text, encoding='utf8')
ROOT.joinpath('gnn_injection_crossmodel_aggregation.json').write_text(str(report), encoding='utf8')
print('Wrote gnn_injection_crossmodel_aggregation.md and .json')
