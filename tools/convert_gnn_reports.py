#!/usr/bin/env python3
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'gnn_enhancement_analysis'
files = [
    'excelformer_gnn_enhancement.txt',
    'fttransformer_gnn_enhancement.txt',
    'resnet_gnn_enhancement.txt',
    'scarf_gnn_enhancement.txt',
    'subtab_gnn_enhancement.txt',
    'tabm_gnn_enhancement.txt',
    'tabnet_gnn_enhancement.txt',
    'tabtransformer_gnn_enhancement.txt',
    'trompt_gnn_enhancement.txt',
    'vime_gnn_enhancement.txt',
    'all_models_ranking_by_classification.txt'
]

sep_line_re = re.compile(r"^={10,}$")
section_header_re = re.compile(r"^分類:")

for fname in files:
    src = root / fname
    if not src.exists():
        print(f"skip missing {src}")
        continue
    text = src.read_text(encoding='utf-8')
    out_lines = []
    i = 0
    lines = text.splitlines()
    n = len(lines)
    while i < n:
        line = lines[i]
        # detect the table header line starting with '排名' (may have spaces)
        if line.strip().startswith('排名') and '競爭者' in line:
            # consume header and the dashed separator
            # write markdown table header
            out_lines.append('| 排名 | 競爭者 | 平均排名 | 資料集數 |')
            out_lines.append('|---:|---|---:|---:|')
            # skip next line if it's a dashed line
            i += 1
            # skip dashes line if present
            if i < n and re.match(r'^-+', lines[i].strip()):
                i += 1
            # now process rows until blank line or separator of equals or next section
            while i < n:
                l = lines[i]
                if not l.strip():
                    # blank line ends table
                    i += 1
                    break
                if sep_line_re.match(l.strip()):
                    # don't consume the sep here; it will be handled in main loop
                    break
                if section_header_re.match(l.strip()):
                    break
                # attempt to parse row: rank, competitor (which may contain many spaces), mean_rank, n
                # We'll split by 2+ spaces
                parts = re.split(r"\s{2,}", l.strip())
                # sometimes the competitor contains extra spaces; we expect 4 parts
                if len(parts) >= 4:
                    rank = parts[0]
                    mean = parts[-2]
                    num = parts[-1]
                    comp = '  '.join(parts[1:-2 + 1]) if len(parts) > 4 else parts[1]
                    # But above logic is brittle; better: rank, competitor, mean, num = first, middle, second last, last
                    comp = '  '.join(parts[1:-2+1]) if False else parts[1] if len(parts)==4 else '  '.join(parts[1:-2])
                    # If the competitor is empty due to parsing, fallback to reconstruct from original using positions
                    if not comp.strip():
                        # try regex: rank at line start, last two numeric columns
                        m = re.match(r"^(\d+)\s+(.*)\s+(\d+\.?\d*)\s+(\d+)$", l.strip())
                        if m:
                            rank, comp, mean, num = m.groups()
                    # insert a HTML line break before first '(' to allow the renderer to wrap model/config
                    try:
                        comp = re.sub(r"\(", "<br>(", comp, count=1)
                    except Exception:
                        pass
                    out_lines.append(f'| {rank} | {comp} | {mean} | {num} |')
                else:
                    # If can't parse, include line in code block for manual inspection
                    out_lines.append('')
                    out_lines.append('```')
                    out_lines.append(l)
                    out_lines.append('```')
                i += 1
            continue
        else:
            # skip separator lines consisting of ========
            if sep_line_re.match(line.strip()):
                i += 1
                continue
            # convert section headers like '分類: ...' into '* #### 分類: ...'
            if section_header_re.match(line.strip()):
                out_lines.append(f"* #### {line.strip()}")
                i += 1
                continue
            out_lines.append(line)
            i += 1
    out_text = '\n'.join(out_lines) + '\n'
    dst = src.with_suffix('.md')
    dst.write_text(out_text, encoding='utf-8')
    print(f'Wrote {dst}')

print('Done')
