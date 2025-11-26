#!/usr/bin/env python3
"""Run a sequence of `main.py` experiments and log outputs.

Usage examples:
  python scripts/run_experiments.py --models excelformer --datasets kaggle_Audit_Data,eye,house --epochs 2 --gpu 0
  python scripts/run_experiments.py --models excelformer,trompt --epochs 2

This script runs `python main.py` from the repository root and writes
stdout+stderr to `results/{dataset}_models_{model}_gnn_stages_all_epochs_{epochs}_gpu_{gpu}.txt`.
"""
from __future__ import annotations

import argparse
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import List


def build_command(repo_root: Path, dataset: str, model: str, epochs: int, gpu: int, gnn_stages: str, extra: str) -> List[str]:
    cmd = [
        "python",
        str(repo_root / "main.py"),
        "--dataset", dataset,
        "--models", model,
        "--gnn_stages", gnn_stages,
        "--epochs", str(epochs),
        "--gpu", str(gpu),
    ]
    if extra:
        # allow passing additional flags as a single string (will be split)
        cmd += extra.split()
    return cmd


def run_and_log(repo_root: Path, dataset: str, model: str, epochs: int, gpu: int, gnn_stages: str, extra: str, results_dir: Path):
    cmd = build_command(repo_root, dataset, model, epochs, gpu, gnn_stages, extra)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"dataset_{dataset}_models_{model}_gnn_stages_{gnn_stages}_epochs_{epochs}_gpu_{gpu}_{timestamp}.txt"
    out_path = results_dir / fname

    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {out_path}")

    # Run command from repository root so relative imports / paths work
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('COMMAND: ' + ' '.join(cmd) + '\n\n')
        fh.write('RETURN CODE: ' + str(proc.returncode) + '\n\n')
        fh.write('STDOUT:\n')
        fh.write(proc.stdout)
        fh.write('\n\nSTDERR:\n')
        fh.write(proc.stderr)

    if proc.returncode != 0:
        print(f"Command failed with return code {proc.returncode}; see {out_path}")
    else:
        print(f"Command completed; output written to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=str, default='excelformer',
                   help='Comma-separated model names (e.g. excelformer,trompt)')
    p.add_argument('--datasets', type=str,
                   default='kaggle_Audit_Data,eye,house,credit,openml_The_Office_Dataset',
                   help='Comma-separated dataset names')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--gnn_stages', type=str, default='all')
    p.add_argument('--extra', type=str, default='',
                   help='Extra flags to append to main.py invocation (single string)')
    p.add_argument('--results-dir', type=str, default='results', help='Directory to write logs')
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]

    for model in models:
        for dataset in datasets:
            run_and_log(repo_root, dataset, model, args.epochs, args.gpu, args.gnn_stages, args.extra, results_dir)


if __name__ == '__main__':
    main()
