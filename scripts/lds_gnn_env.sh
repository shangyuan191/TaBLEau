#!/usr/bin/env bash
# Usage:
#   source scripts/lds_gnn_env.sh
#
# Points TaBLEau's LDS-GNN wrapper to a dedicated TF1 environment.
#
# NOTE: This file is intended to be *sourced* into your current shell.
# Do not enable `set -e/-u/-o pipefail` here, because that can cause the
# entire interactive terminal to exit if a later command returns non-zero.

export LDS_GNN_PYTHON="/home/skyler/miniconda3/envs/lds_gnn_tf1/bin/python"
export LDS_GNN_REPO="/home/skyler/ModelComparison/LDS-GNN"
export LDS_GCN_REPO="/home/skyler/ModelComparison/LDS-GNN/deps/gcn"

# Keep native BLAS/OMP thread counts conservative; this prevents some OpenBLAS
# builds from exhausting per-thread metadata ("too many memory regions") on
# large runs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# (Optional) help Kipf gcn imports if other tooling runs in-process.
export PYTHONPATH="${LDS_GCN_REPO}:${LDS_GNN_REPO}:${PYTHONPATH:-}"

echo "LDS_GNN_PYTHON=${LDS_GNN_PYTHON}"
echo "LDS_GNN_REPO=${LDS_GNN_REPO}"
echo "LDS_GCN_REPO=${LDS_GCN_REPO}"
