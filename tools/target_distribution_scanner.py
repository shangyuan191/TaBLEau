#!/usr/bin/env python3
"""
Scan regression CSV files under datasets/*/regression, detect target column,
compute distributional statistics and heuristics, classify the target distribution,
and save a summary CSV plus per-dataset histogram+KDE PNGs.

Usage:
    python tools/target_distribution_scanner.py --root datasets --out analysis/target_distributions

Dependencies: pandas, numpy, scipy, matplotlib
"""
import argparse
import os
import sys
import math
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
except Exception as e:
    print("Missing Python dependencies. Please install: pandas, numpy, scipy, matplotlib")
    print("Import error:", e)
    sys.exit(2)


KEY_TARGET_NAMES = [
    "target", "y", "label", "price", "saleprice", "value", "median_house_value",
    "target_value", "target_y", "y_true", "Y"
]


def find_csv_files(root: Path):
    files = []
    for p in root.rglob("*.csv"):
        # only under regression directories
        if "/regression/" in str(p):
            files.append(p)
    return sorted(files)


def detect_target_column(df: pd.DataFrame):
    # Prefer columns with common target-like names
    cols = list(df.columns)
    lower_map = {c: c.lower() for c in cols}
    for key in KEY_TARGET_NAMES:
        for c, cl in lower_map.items():
            if key == cl or key in cl:
                # ensure numeric
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
    # fallback: choose last numeric column
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        return None
    return numeric_cols[-1]


def compute_stats(s: pd.Series):
    s_clean = s.dropna().astype(float)
    n = len(s_clean)
    if n == 0:
        return None
    unique_count = s_clean.nunique()
    unique_ratio = unique_count / n
    zero_prop = (s_clean == 0).sum() / n
    mean = float(s_clean.mean())
    median = float(s_clean.median())
    std = float(s_clean.std())
    mn = float(s_clean.min())
    mx = float(s_clean.max())
    q25 = float(s_clean.quantile(0.25))
    q75 = float(s_clean.quantile(0.75))
    q50 = median
    q99 = float(s_clean.quantile(0.99))
    iqr = q75 - q25
    skewness = float(stats.skew(s_clean)) if n >= 3 else 0.0
    kurt = float(stats.kurtosis(s_clean, fisher=False)) if n >= 4 else 0.0

    # quantile ratio: how heavy the tails are
    qratio = (q99 - q50) / (q75 - q25 + 1e-12)

    # detect integer-like
    is_integer_like = np.all(np.isclose(s_clean, np.round(s_clean), atol=1e-8))

    # KDE peaks
    peaks = 0
    try:
        if unique_count > 1 and n > 10 and mx > mn:
            vals = s_clean.values
            grid = np.linspace(mn, mx, 512)
            kde = stats.gaussian_kde(vals)
            dens = kde(grid)
            peak_idx, _ = find_peaks(dens, prominence=0.01 * dens.max())
            peaks = len(peak_idx)
    except Exception:
        peaks = 0

    return dict(
        n=n,
        unique_count=int(unique_count),
        unique_ratio=float(unique_ratio),
        zero_prop=float(zero_prop),
        mean=mean,
        median=median,
        std=std,
        min=mn,
        max=mx,
        q25=q25,
        q50=q50,
        q75=q75,
        q99=q99,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurt,
        qratio=float(qratio),
        integer_like=bool(is_integer_like),
        peaks=int(peaks),
    )


def classify(stats: dict):
    labels = []
    if stats is None:
        return ["invalid"]
    if stats["n"] == 0:
        return ["invalid"]
    if stats["unique_ratio"] < 0.01 or stats["std"] == 0 or (stats["max"] - stats["min"]) == 0:
        labels.append("constant")
    if stats["integer_like"] and stats["unique_ratio"] < 0.05:
        labels.append("discrete")
    if stats["zero_prop"] >= 0.2:
        labels.append("zero_inflated")
    if stats["peaks"] >= 2:
        labels.append("multimodal")
    # skewness-based
    s = stats["skewness"]
    if abs(s) < 0.5:
        labels.append("approx_normal")
    elif abs(s) < 1.0:
        labels.append("moderately_skewed")
    else:
        labels.append("highly_skewed")
    # heavy tail
    if stats["kurtosis"] > 10 or stats["qratio"] > 5:
        labels.append("heavy_tailed")
    # bounded
    if stats["min"] >= 0 and stats["max"] <= 1:
        labels.append("bounded_0_1")
    # fallback
    if len(labels) == 0:
        labels.append("unknown")
    return labels


def plot_distribution(s: pd.Series, out_png: Path, title: str = None):
    s_clean = s.dropna().astype(float)
    if len(s_clean) == 0:
        return
    plt.figure(figsize=(6, 4))
    try:
        sns = None
        # Histogram
        plt.hist(s_clean, bins=60, density=True, alpha=0.6, color="C0")
        # KDE
        try:
            vals = s_clean.values
            grid = np.linspace(s_clean.min(), s_clean.max(), 512)
            kde = stats.gaussian_kde(vals)
            plt.plot(grid, kde(grid), color="C1")
        except Exception:
            pass
        plt.title(title or "target distribution")
        plt.xlabel("value")
        plt.ylabel("density")
        plt.tight_layout()
        plt.savefig(out_png)
    finally:
        plt.close()


def process_file(csv_path: Path, out_dir: Path, args):
    rel = csv_path.relative_to(Path.cwd())
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return dict(csv=str(csv_path), error=str(e))

    target_col = detect_target_column(df)
    if target_col is None:
        return dict(csv=str(csv_path), error="no numeric column detected")

    stats = compute_stats(df[target_col])
    labels = classify(stats)

    # save plot
    out_png = out_dir / (csv_path.stem + "__" + target_col + ".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        plot_distribution(df[target_col], out_png, title=f"{csv_path.stem} :: {target_col}")
    except Exception:
        pass

    rec = dict(
        csv=str(csv_path),
        dataset=str(csv_path.parent),
        target_col=target_col,
        labels=";".join(labels),
    )
    if stats is not None:
        rec.update(stats)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets", help="root datasets folder")
    parser.add_argument("--out", default="analysis/target_distributions", help="output folder for summary and plots")
    parser.add_argument("--recursive", action="store_true", default=True, help="scan recursively")
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    csvs = find_csv_files(root)
    if len(csvs) == 0:
        print("No CSV files found under regression subfolders in", root)
        sys.exit(0)

    print(f"Found {len(csvs)} CSV files. Processing...")
    rows = []
    for p in csvs:
        print("processing", p)
        rec = process_file(p, out, args)
        rows.append(rec)

    df = pd.DataFrame(rows)
    summary_csv = out / "target_distribution_summary.csv"
    df.to_csv(summary_csv, index=False)
    print("Wrote summary to", summary_csv)


if __name__ == "__main__":
    main()
