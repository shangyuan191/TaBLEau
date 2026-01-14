#!/usr/bin/env bash
set -euo pipefail

# Run LDS-GNN across dataset filters in TaBLEau, using a TF1 subprocess env.
#
# Usage:
#   nohup bash scripts/run_lds_gnn_all.sh > lds_gnn_all.log 2>&1 &
#
# Optional overrides (env vars):
#   LDS_GNN_PYTHON  Path to TF1 python (default: /home/skyler/miniconda3/envs/lds_gnn_tf1/bin/python)
#   LDS_GNN_REPO    Path to LDS-GNN repo   (default: /home/skyler/ModelComparison/LDS-GNN)
#   LDS_GCN_REPO    Path to Kipf gcn repo  (default: $LDS_GNN_REPO/deps/gcn)
#   CONDA_ENV       TaBLEau env name       (default: tableau)
#   DATASET_SIZE    (default: all)
#   TASK_TYPE       (default: all)
#   FEATURE_TYPE    (default: all)
#
# Notes:
# - LDS-GNN is a comparison/self-contained model, so --gnn_stages is ignored.

ROOT="/home/skyler/ModelComparison/TaBLEau"

export LDS_GNN_PYTHON="${LDS_GNN_PYTHON:-/home/skyler/miniconda3/envs/lds_gnn_tf1/bin/python}"
export LDS_GNN_REPO="${LDS_GNN_REPO:-/home/skyler/ModelComparison/LDS-GNN}"
export LDS_GCN_REPO="${LDS_GCN_REPO:-${LDS_GNN_REPO}/deps/gcn}"

CONDA_ENV="${CONDA_ENV:-tableau}"
DATASET_SIZE="${DATASET_SIZE:-all}"
TASK_TYPE="${TASK_TYPE:-all}"
FEATURE_TYPE="${FEATURE_TYPE:-all}"

cd "$ROOT"

echo "[run_lds_gnn_all] LDS_GNN_PYTHON=$LDS_GNN_PYTHON"
echo "[run_lds_gnn_all] LDS_GNN_REPO=$LDS_GNN_REPO"
echo "[run_lds_gnn_all] LDS_GCN_REPO=$LDS_GCN_REPO"
echo "[run_lds_gnn_all] CONDA_ENV=$CONDA_ENV"
echo "[run_lds_gnn_all] Filters: dataset_size=$DATASET_SIZE task_type=$TASK_TYPE feature_type=$FEATURE_TYPE"

timestamp="$(date +%Y%m%d_%H%M%S)"
echo "[run_lds_gnn_all] Starting at $timestamp"

conda run -n "$CONDA_ENV" python main.py \
  --dataset_size "$DATASET_SIZE" \
  --task_type "$TASK_TYPE" \
  --feature_type "$FEATURE_TYPE" \
  --models lds_gnn \
  --gnn_stages none

echo "[run_lds_gnn_all] Done"
