#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/../envs/jslee_py311/bin/python"
MODULE="AXData.ax_catalog_experiments.exhaustive_e1_e2_deadline"
OUTPUT="$PROJECT_ROOT/ax_catalog_experiments/outputs/exhaustive_e1_e2_deadline_quick_v1"
EXPECTED=6096
COMMON_ENV=(
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  PYTHONWARNINGS=ignore
  AX_DEADLINE_CAMPAIGN_ID=exhaustive_e1_e2_deadline_quick_v1
  AX_DEADLINE_METHOD=deadline_single_seed_epoch_capped_screening
  AX_DEADLINE_SEEDS=42
  AX_DEADLINE_EPOCH_CAP_CATBOOST=50
  AX_DEADLINE_EPOCH_CAP_MLP=30
  AX_DEADLINE_EPOCH_CAP_TABM=20
  AX_DEADLINE_EPOCH_CAP_TFT=3
  AX_DEADLINE_BATCH_SIZE_TFT=512
)

cd "$PROJECT_ROOT/.."

count_results() {
  find "$OUTPUT/summaries" -type f -name '*.json' 2>/dev/null | wc -l
}

while true; do
  completed="$(count_results)"
  printf '%s quick_completed=%s/%s\n' "$(date '+%F %T')" "$completed" "$EXPECTED"
  if [ "$completed" -ge "$EXPECTED" ]; then
    env "${COMMON_ENV[@]}" "$PYTHON" -u -m "$MODULE" finalize --output "$OUTPUT" || exit $?
    printf '%s quick campaign finalized; resuming three-seed campaign supervisor\n' "$(date '+%F %T')"
    exec bash "$PROJECT_ROOT/supervise_ax_e1_e2_deadline.sh"
  fi

  if ! pgrep -f '[e]xhaustive_e1_e2_deadline (run|gpu|cpu)' >/dev/null; then
    printf '%s no active worker; starting quick catch-up queues\n' "$(date '+%F %T')"
    env "${COMMON_ENV[@]}" AX_DEADLINE_CPU_WORKERS=16 \
      "$PYTHON" -u -m "$MODULE" cpu --output "$OUTPUT"
    env "${COMMON_ENV[@]}" AX_DEADLINE_MODELS=catboost AX_DEADLINE_GPU_WORKERS=4 AX_CATBOOST_GPU_RAM_PART=0.10 \
      "$PYTHON" -u -m "$MODULE" gpu --output "$OUTPUT"
    env "${COMMON_ENV[@]}" AX_DEADLINE_MODELS=mlp AX_DEADLINE_GPU_WORKERS=4 \
      "$PYTHON" -u -m "$MODULE" gpu --output "$OUTPUT"
    env "${COMMON_ENV[@]}" AX_DEADLINE_MODELS=tabm AX_DEADLINE_GPU_WORKERS=4 \
      "$PYTHON" -u -m "$MODULE" gpu --output "$OUTPUT"
    env "${COMMON_ENV[@]}" AX_DEADLINE_MODELS=tft AX_DEADLINE_GPU_WORKERS=12 \
      "$PYTHON" -u -m "$MODULE" gpu --output "$OUTPUT"
  fi
  sleep 60
done
