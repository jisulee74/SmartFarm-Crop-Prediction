#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${AX_PYTHON:-$ROOT/../envs/jslee_py311/bin/python}"
OUTPUT="$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_screening_v1"
STATUS="$OUTPUT/supervisor_status.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AX_GPU_HOST_THREADS=1 AX_RF_THREADS=1
MAX_RESTARTS="${AX_SCREENING_MAX_RESTARTS:-3}"

write_status() {
  local state="$1" message="$2" attempts="$3"
  mkdir -p "$OUTPUT"
  "$PYTHON_BIN" -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"Status":sys.argv[2],"Message":sys.argv[3],"Restart_Attempts":int(sys.argv[4]),"Updated_At":time.time()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")' "$STATUS" "$state" "$message" "$attempts"
}

attempt=0
while true; do
  write_status running "127개 의미 통합 1시드 선별 실행" "$attempt"
  if bash "$ROOT/run_ax_e6_semantic_screening.sh"; then
    break
  fi
  attempt=$((attempt + 1))
  if (( attempt > MAX_RESTARTS )); then
    write_status failed "빠른 선별 실행 재시도 한도 초과" "$attempt"
    exit 1
  fi
  write_status restarting "완료 결과를 보존하고 빠른 선별 재개" "$attempt"
done

export AX_DEADLINE_CAMPAIGN_ID="exhaustive_e6_semantic_screening_v1"
export AX_DEADLINE_METHOD="semantic_single_seed_epoch_capped_screening"
export AX_DEADLINE_SOURCE_ROOT="$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_completion_v2"
export AX_DEADLINE_SEEDS="42"

write_status finalizing "1시드 순위 생성 및 우승 후보 3시드 Test 평가" "$attempt"
cd "$ROOT/.."
"$PYTHON_BIN" -u -m AXData.ax_catalog_experiments.exhaustive_e1_e2_deadline \
  finalize --output "$OUTPUT"
"$PYTHON_BIN" -u -m AXData.ax_catalog_experiments.exhaustive_e6_semantic_screening \
  finalize --output "$OUTPUT"

if [[ "${AX_SCREENING_AUTO_PUBLISH:-1}" == "0" ]]; then
  write_status ready "Test 평가 완료·배포 검증 대기" "$attempt"
  exit 0
fi

write_status publishing "잠정 선별 결과 대시보드 반영 및 재배포" "$attempt"
bash "$ROOT/finalize_ax_semantic_screening_dashboard.sh"
write_status finished "잠정 선별 결과 대시보드 배포 완료" "$attempt"
