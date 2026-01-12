#!/usr/bin/env bash
# Usage:
#   source scripts/lds_gnn_env.sh
#
# Points TaBLEau's LDS-GNN wrapper to a dedicated TF1 environment.

set -euo pipefail

export LDS_GNN_PYTHON="/home/skyler/miniconda3/envs/lds_gnn_tf1/bin/python"
export LDS_GNN_REPO="/home/skyler/ModelComparison/LDS-GNN"
export LDS_GCN_REPO="/home/skyler/ModelComparison/LDS-GNN/deps/gcn"

# (Optional) help Kipf gcn imports if other tooling runs in-process.
export PYTHONPATH="${LDS_GCN_REPO}:${LDS_GNN_REPO}:${PYTHONPATH:-}"

echo "LDS_GNN_PYTHON=${LDS_GNN_PYTHON}"
echo "LDS_GNN_REPO=${LDS_GNN_REPO}"
echo "LDS_GCN_REPO=${LDS_GCN_REPO}"
