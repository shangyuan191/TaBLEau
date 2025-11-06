#!/usr/bin/env python3
import re
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'gnn_enhancement_analysis'
out = Path(__file__).resolve().parents[1] / 'analysis' / 'gnn_enhancement_parsed.csv'
out.parent.mkdir(parents=True, exist_ok=True)

md_files = sorted(root.glob('*_gnn_enhancement.md'))

group_header_re = re.compile(r"\* #### 分類: ([^\(]+) \(包含 (\d+) 個資料集\)")
table_header_re = re.compile(r"\|\s*排名\s*\|\s*競爭者\s*\|\s*平均排名\s*\|\s*資料集數\s*\|")
row_re = re.compile(r"\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|")
comp_re = re.compile(r"([^<]+)(?:<br>\s*\(ratio=([^,]+),\s*gnn_stage=([^\)]+)\))?")
# competitor format in files actually like: name<br>(ratio=... , gnn_stage=...)
comp_re2 = re.compile(r"(.+?)<br>\(ratio=([^,]+),\s*gnn_stage=([^\)]+)\)")

rows = []
for md in md_files:
    text = md.read_text()
    model = md.stem.replace('_gnn_enhancement','')
    # iterate through group headers
    pos = 0
    while True:
        m = group_header_re.search(text, pos)
        if not m:
            break
        group = m.group(1).strip()
        group_n = int(m.group(2))
        # find table header after this
        th = table_header_re.search(text, m.end())
        if not th:
            pos = m.end()
            continue
        table_start = th.end()
        # find rows following
        cur = table_start
        while True:
            r = row_re.search(text, cur)
            if not r:
                break
            # ensure row belongs to this table (by position - stop when next blank line followed by '*' or next group)
            row_start = r.start()
            # but we'll accept consecutive rows until next '\n* ####'
            # parse competitor
            rank = int(r.group(1))
            comp_raw = r.group(2).strip()
            avg_rank = float(r.group(3))
            ds_count = int(r.group(4))
            # try comp_re2
            comp_name = ''
            ratio = ''
            gnn_stage = ''
            m2 = comp_re2.search(comp_raw)
            if m2:
                comp_name = m2.group(1).strip()
                ratio = m2.group(2).strip()
                gnn_stage = m2.group(3).strip()
            else:
                # fallback: strip html tags and parentheses
                comp_name = re.sub(r'<.*?>','', comp_raw).strip()
                # try to extract ratio/gnn_stage by searching parentheses
                m3 = re.search(r'ratio=([^,\)]+)', comp_raw)
                m4 = re.search(r'gnn_stage=([^\)]+)', comp_raw)
                if m3:
                    ratio = m3.group(1).strip()
                if m4:
                    gnn_stage = m4.group(1).strip()
            rows.append({
                'model': model,
                'group': group,
                'group_dataset_count': group_n,
                'rank_pos': rank,
                'competitor': comp_name,
                'ratio': ratio,
                'gnn_stage': gnn_stage,
                'avg_rank': avg_rank,
                'competitor_dataset_count': ds_count,
                'source_file': str(md)
            })
            cur = r.end()
            # check if next chunk contains another table header or group header before next row start
            # simple approach: continue until we hit a blank line followed by '* ####' or '```' or end
            # break condition handled by failing to find a row
        pos = r.end() if r else m.end()

# write CSV
fieldnames = ['model','group','group_dataset_count','rank_pos','competitor','ratio','gnn_stage','avg_rank','competitor_dataset_count','source_file']
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f'Parsed {len(rows)} rows into {out}')
