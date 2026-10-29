#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${AX_PYTHON:-$ROOT/../envs/jslee_py311/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
export AX_SEMANTIC_COMPLETION_GPU_WORKERS="${AX_SEMANTIC_COMPLETION_GPU_WORKERS:-4}"
export AX_SEMANTIC_COMPLETION_CPU_WORKERS="${AX_SEMANTIC_COMPLETION_CPU_WORKERS:-8}"
export AX_CATBOOST_GPU_RAM_PART="${AX_CATBOOST_GPU_RAM_PART:-0.08}"
export AX_RF_THREADS="${AX_RF_THREADS:-4}"
export AX_GPU_HOST_THREADS="${AX_GPU_HOST_THREADS:-2}"
# The container exposes 72 CPUs but has an 8-core cgroup quota.
# Bound BLAS per worker without changing grids, seeds, or selection policy.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
cd "$ROOT/.."
"$PYTHON_BIN" -u "$ROOT/ax_completion_watchdog.py" \
  --state-dir "$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_completion_v2" -- \
  "$PYTHON_BIN" -u -m AXData.ax_catalog_experiments.exhaustive_e6_semantic_completion all "$@"
