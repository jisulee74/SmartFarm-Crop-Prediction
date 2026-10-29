#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/../envs/jslee_py311/bin/python"
MODULE="AXData.ax_catalog_experiments.exhaustive_e1_e2_deadline"
OUTPUT="$PROJECT_ROOT/ax_catalog_experiments/outputs/exhaustive_e1_e2_deadline_v1"
EXPECTED=6096

cd "$PROJECT_ROOT/.."

count_results() {
  find "$OUTPUT/summaries" -type f -name '*.json' 2>/dev/null | wc -l
}

while true; do
  completed="$(count_results)"
  printf '%s completed=%s/%s\n' "$(date '+%F %T')" "$completed" "$EXPECTED"
  if [ "$completed" -ge "$EXPECTED" ]; then
    "$PYTHON" -u -m "$MODULE" finalize --output "$OUTPUT"
    exit $?
  fi

  if ! pgrep -f '[e]xhaustive_e1_e2_deadline (run|gpu|cpu)' >/dev/null; then
    printf '%s no active worker; starting catch-up queues\n' "$(date '+%F %T')"
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore \
      AX_DEADLINE_CPU_WORKERS=16 "$PYTHON" -u -m "$MODULE" cpu --output "$OUTPUT"
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore \
      AX_DEADLINE_MODELS=catboost AX_DEADLINE_GPU_WORKERS=4 AX_CATBOOST_GPU_RAM_PART=0.10 \
      "$PYTHON" -u -m "$MODULE" gpu --output "$OUTPUT"
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore \
      AX_DEADLINE_MODELS=mlp,tabm AX_DEADLINE_GPU_WORKERS=4 \
      "$PYTHON" -u -m "$MODULE" gpu --output "$OUTPUT"
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore \
      AX_DEADLINE_MODELS=tft AX_DEADLINE_GPU_WORKERS=4 \
      "$PYTHON" -u -m "$MODULE" gpu --output "$OUTPUT"
  fi
  sleep 60
done
