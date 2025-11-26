#!/usr/bin/env python3
"""Run the fixed unit-test experiment set (six commands) and log outputs.

This script is intentionally hard-coded to run these six jobs (for CI/unit
testing) and will write logs into the repository `results/` folder.

Jobs (hard-coded):
  python main.py --dataset kaggle_Audit_Data --models resnet --gnn_stages all --epochs 2
  python main.py --dataset eye --models resnet --gnn_stages all --epochs 2
  python main.py --dataset house --models resnet --gnn_stages all --epochs 2
  python main.py --dataset credit --models resnet --gnn_stages all --epochs 2
  python main.py --dataset openml_The_Office_Dataset --models resnet --gnn_stages all --epochs 2

"""
from __future__ import annotations

import subprocess
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm


JOBS = [
    ("kaggle_Audit_Data", "resnet"),
    ("openml_The_Office_Dataset","resnet"),
    ("credit", "resnet"),
    ("eye", "resnet"),
    ("helena", "resnet"),
    ("house", "resnet")
]


def run_job(repo_root: Path, dataset: str, model: str, epochs: int, gpu: int, gnn_stages: str):
    cmd = [
        "python",
        str(repo_root / "main.py"),
        "--dataset", dataset,
        "--models", model,
        "--gnn_stages", gnn_stages,
        "--epochs", str(epochs),
        "--gpu", str(gpu),
    ]

    # run and return proc
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    return proc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--gnn_stages', type=str, default='all')
    p.add_argument('--model', '--models', dest='model', type=str, default=None,
                   help='If set, override the model for all jobs (e.g. "resnet")')
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # run jobs with a tqdm progress bar and show the current dataset/model
    with tqdm(total=len(JOBS)) as pbar:
        for dataset, model in JOBS:
            # allow overriding the hard-coded model via CLI
            model_to_use = args.model if args.model is not None else model
            pbar.set_description(f"Running {dataset} [{model_to_use}]")
            proc = run_job(repo_root, dataset, model_to_use, args.epochs, args.gpu, args.gnn_stages)
            if proc.returncode != 0:
                pbar.write(f"FAILED: {dataset} ({model})  rc={proc.returncode}")
                # also print stderr for quick debugging
                if proc.stderr:
                    pbar.write(proc.stderr)
            else:
                pbar.write(f"OK: {dataset} ({model})")
            pbar.update(1)


if __name__ == '__main__':
    main()
