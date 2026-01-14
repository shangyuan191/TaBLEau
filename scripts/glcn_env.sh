#!/usr/bin/env bash
# Usage:
#   source scripts/glcn_env.sh
#
# Points TaBLEau's GLCN wrapper to a dedicated TF1 environment.
#
# NOTE: This file is intended to be *sourced* into your current shell.
# Do not enable `set -e/-u/-o pipefail` here, because that can cause the
# entire interactive terminal to exit if a later command returns non-zero.

# Reuse the existing TF1 environment used by LDS-GNN, unless you want a separate one.
export GLCN_PYTHON="/home/skyler/miniconda3/envs/lds_gnn_tf1/bin/python"
export GLCN_TF_REPO="/home/skyler/ModelComparison/GLCN-tf"

# 防止 OpenBLAS 因 thread metadata 分配過多而崩潰（helena 之類大資料集常見）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# Important: do NOT export GLCN into PYTHONPATH here.
# TaBLEau's `main.py` imports `utils.*`, and the GLCN repo also contains a top-level
# `utils.py` that imports TensorFlow; adding it to PYTHONPATH will break TaBLEau.

# If an older version of this script already polluted PYTHONPATH in the current shell,
# clean it up proactively.
if [[ -n "${PYTHONPATH:-}" ]]; then
	_new_pp=""
	IFS=':' read -r -a _pp_parts <<< "${PYTHONPATH}"
	for _p in "${_pp_parts[@]}"; do
		[[ -z "${_p}" ]] && continue
		if [[ "${_p}" == "${GLCN_TF_REPO}" || "${_p}" == "${GLCN_TF_REPO}/glcn" ]]; then
			continue
		fi
		if [[ -z "${_new_pp}" ]]; then
			_new_pp="${_p}"
		else
			_new_pp="${_new_pp}:${_p}"
		fi
	done
	export PYTHONPATH="${_new_pp}"
	unset _new_pp _pp_parts _p
fi

echo "GLCN_PYTHON=${GLCN_PYTHON}"
echo "GLCN_TF_REPO=${GLCN_TF_REPO}"
