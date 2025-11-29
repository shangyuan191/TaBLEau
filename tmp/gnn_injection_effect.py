import re
from pathlib import Path

FILES = [
    'gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_primary_split_baseline_test_group_level.md',
    'gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_primary_split_baseline_val_group_level.md',
]
ROOT = Path('')
ratio_re = re.compile(r"\(([0-9]*\.?[0-9]+)\)")


def parse_tables(txt):
    sections = re.split(r"\n## Baseline:", txt)
    out = {}
    for sec in sections[1:]:
        lines = sec.splitlines()
        baseline_name = lines[0].strip()
        table_lines = [ln for ln in lines if ln.strip().startswith('|')]
        if not table_lines:
            continue
        header = table_lines[0].strip()
        headers = [h.strip() for h in header.split('|')[1:-1]]
        rows = []
        for ln in table_lines[1:]:
            if set(ln.strip()) <= set('|-: '):
                continue
            cells = [c.strip() for c in ln.strip().split('|')[1:-1]]
            rows.append(cells)
        out[baseline_name] = {'headers': headers, 'rows': rows}
    return out


def extract_ratio(cell):
    m = ratio_re.search(cell)
    if not m:
        return None
    return float(m.group(1))


def analyze_file(p, injection_candidates=('start','encoding','materialize','columnwise','decoding')):
    txt = p.read_text()
    tables = parse_tables(txt)
    report = {}
    for baseline, t in tables.items():
        headers = t['headers']
        cats = headers[1:]
        inj_map = {}
        for r in t['rows']:
            inj = r[0]
            vals = [extract_ratio(c) for c in r[1:]]
            inj_map[inj] = vals
        none_vals = inj_map.get('none')
        stats = {}
        for inj in injection_candidates:
            if inj not in inj_map:
                continue
            vals = inj_map[inj]
            wins = 0
            ties = 0
            losses = 0
            deltas = []
            for i, cat in enumerate(cats):
                v = vals[i]
                n = none_vals[i] if none_vals is not None else None
                if v is None or n is None:
                    continue
                delta = v - n
                deltas.append(delta)
                if delta > 1e-9:
                    wins += 1
                elif abs(delta) <= 1e-9:
                    ties += 1
                else:
                    losses += 1
            stats[inj] = {'wins': wins, 'ties': ties, 'losses': losses, 'mean_delta': (sum(deltas)/len(deltas) if deltas else None), 'n_cats': len(deltas)}
        report[baseline] = stats
    return report


def main():
    agg = {}
    for f in FILES:
        p = ROOT / f
        if not p.exists():
            print('Missing', f)
            continue
        key = p.name
        agg[key] = analyze_file(p)
    # Print summary
    for fname, rep in agg.items():
        print('\n' + '='*60)
        print('File:', fname)
        for baseline, stats in rep.items():
            print('\nBaseline:', baseline)
            if not stats:
                print('  (no matching injections)')
                continue
            for inj, s in stats.items():
                print(f"  Injection: {inj:12}  wins={s['wins']:2}  ties={s['ties']:2}  losses={s['losses']:2}  n_cat={s['n_cats']:2}  mean_delta={s['mean_delta']}")
    print('\nDone')

if __name__ == '__main__':
    main()
