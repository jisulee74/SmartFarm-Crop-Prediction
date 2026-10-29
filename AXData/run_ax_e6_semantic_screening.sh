#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${AX_PYTHON:-$ROOT/../envs/jslee_py311/bin/python}"
OUTPUT="$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_screening_v1"
SOURCE="$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_completion_v2"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export AX_DEADLINE_CAMPAIGN_ID="exhaustive_e6_semantic_screening_v1"
export AX_DEADLINE_METHOD="semantic_single_seed_epoch_capped_screening"
export AX_DEADLINE_SOURCE_ROOT="$SOURCE"
export AX_DEADLINE_SEEDS="42"
export AX_DEADLINE_SELECTION_WORKERS="${AX_DEADLINE_SELECTION_WORKERS:-8}"
export AX_DEADLINE_CPU_WORKERS="${AX_DEADLINE_CPU_WORKERS:-12}"
export AX_DEADLINE_GPU_WORKERS="${AX_DEADLINE_GPU_WORKERS:-4}"
export AX_DEADLINE_EPOCH_CAP_CATBOOST="${AX_DEADLINE_EPOCH_CAP_CATBOOST:-50}"
export AX_DEADLINE_EPOCH_CAP_MLP="${AX_DEADLINE_EPOCH_CAP_MLP:-30}"
export AX_DEADLINE_EPOCH_CAP_TABM="${AX_DEADLINE_EPOCH_CAP_TABM:-20}"
export AX_DEADLINE_EPOCH_CAP_TFT="${AX_DEADLINE_EPOCH_CAP_TFT:-3}"
export AX_DEADLINE_BATCH_SIZE_TFT="${AX_DEADLINE_BATCH_SIZE_TFT:-512}"
export AX_CATBOOST_GPU_RAM_PART="${AX_CATBOOST_GPU_RAM_PART:-0.05}"

mkdir -p "$OUTPUT"
ln -sfn "$SOURCE/prepared" "$OUTPUT/prepared"
ln -sfn "$SOURCE/prepared_manifest.json" "$OUTPUT/prepared_manifest.json"

cd "$ROOT/.."
exec "$PYTHON_BIN" -u -m AXData.ax_catalog_experiments.exhaustive_e1_e2_deadline \
  run --output "$OUTPUT"
