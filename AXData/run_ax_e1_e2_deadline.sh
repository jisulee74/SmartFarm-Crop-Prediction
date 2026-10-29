#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${AX_PYTHON:-$ROOT/../envs/jslee_py311/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export AX_DEADLINE_SELECTION_WORKERS="${AX_DEADLINE_SELECTION_WORKERS:-12}"
export AX_DEADLINE_CPU_WORKERS="${AX_DEADLINE_CPU_WORKERS:-16}"
export AX_DEADLINE_GPU_WORKERS="${AX_DEADLINE_GPU_WORKERS:-16}"
cd "$ROOT/.."
exec "$PYTHON_BIN" -u -m AXData.ax_catalog_experiments.exhaustive_e1_e2_deadline run
