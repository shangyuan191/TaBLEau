import re
from pathlib import Path

p = Path('gnn_injection_analysis/per_model_result/all_models_sensitivity_aggregate_primary_split_baseline_test_group_level.md')
text = p.read_text()

# We'll parse sections by '## Baseline:'
sections = re.split(r"\n## Baseline:", text)[1:]
ratio_re = re.compile(r"\(([0-9]*\.?[0-9]+)\)")
nums_re = re.compile(r"(\d+)\/(\d+)\/(\d+)")

report = {}
for sec in sections:
    lines = sec.splitlines()
    baseline = lines[0].strip()
    table_lines = [ln for ln in lines if ln.strip().startswith('|')]
    if not table_lines:
        continue
    header = table_lines[0]
    headers = [h.strip() for h in header.split('|')[1:-1]]
    cats = headers[1:]
    # parse rows
    rows = []
    for ln in table_lines[1:]:
        if set(ln.strip()) <= set('|-: '):
            continue
        cells = [c.strip() for c in ln.split('|')[1:-1]]
        rows.append(cells)
    # build per-injection aggregates
    inj_stats = {}
    for r in rows:
        inj = r[0]
        wins = []
        ties = []
        totals = []
        for cell in r[1:]:
            m = nums_re.search(cell)
            if not m:
                wins.append(None); ties.append(None); totals.append(None)
            else:
                a,b,c = int(m.group(1)), int(m.group(2)), int(m.group(3))
                # interpretation: wins / ties / total (user clarified)
                wins.append(a)
                ties.append(b)
                totals.append(c)
        inj_stats[inj] = {'wins': wins, 'ties': ties, 'totals': totals}
    # compute per-injection totals and proportions across categories (sum wins / sum totals)
    summary = {}
    for inj, d in inj_stats.items():
        wins_sum = sum([v for v in d['wins'] if v is not None])
        ties_sum = sum([v for v in d['ties'] if v is not None])
        totals_sum = sum([v for v in d['totals'] if v is not None])
        # proportion of strictly-beat across all datasets in these categories
        prop = wins_sum / totals_sum if totals_sum>0 else None
        summary[inj] = {'wins_sum': wins_sum, 'ties_sum': ties_sum, 'totals_sum': totals_sum, 'prop': prop}
    # also compute per-category which injection has most strict wins proportion in that category
    per_cat_best = []
    n_cats = len(cats)
    for ci, cat in enumerate(cats):
        best_inj = None
        best_prop = -1
        cat_tot = None
        for inj, d in inj_stats.items():
            w = d['wins'][ci]
            t = d['totals'][ci]
            if w is None or t is None:
                continue
            pval = w / t if t>0 else 0
            if pval > best_prop:
                best_prop = pval
                best_inj = inj
                cat_tot = t
        per_cat_best.append({'category': cat, 'best_inj': best_inj, 'best_prop': best_prop, 'total': cat_tot})

    report[baseline] = {'summary': summary, 'per_cat_best': per_cat_best, 'categories': cats}

# Print readable report
for baseline, info in report.items():
    print('\n' + '='*60)
    print('Baseline:', baseline)
    print('Overall strictly-beat totals and proportions (wins / total):')
    for inj, s in sorted(info['summary'].items(), key=lambda x: (- (x[1]['prop'] or 0), -x[1]['wins_sum'])):
        print(f"  {inj:12} wins={s['wins_sum']:4} ties={s['ties_sum']:4} total={s['totals_sum']:4} prop={s['prop']:.3f}")
    print('\nPer-category best (by wins/total proportion):')
    for c in info['per_cat_best']:
        print(f"  {c['category'][:50]:50} | {c['best_inj']:12} | {c['best_prop']:.3f} (n={c['total']})")

print('\nDone')
