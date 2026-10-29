#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_completion_v2"
LOCK="$OUTPUT/campaign.lock"
LOG="$ROOT/ax_e6_semantic_completion.log"
PID_FILE="$ROOT/ax_e6_semantic_completion.pid"
STATE_FILE="$OUTPUT/supervisor_status.json"
MAX_RESTARTS="${AX_SEMANTIC_COMPLETION_MAX_RESTARTS:-20}"

campaign_status() {
  python - "$OUTPUT/scheduler.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
try:
    print(json.loads(p.read_text()).get('Status','missing'))
except Exception:
    print('missing')
PY
}

write_state() {
  local state="$1" attempts="$2" message="$3"
  python - "$STATE_FILE" "$state" "$attempts" "$message" <<'PY'
import json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'Status':sys.argv[2],'Restart_Attempts':int(sys.argv[3]),'Message':sys.argv[4],'Updated_At':time.time()},ensure_ascii=False,indent=2))
os.replace(tmp,p)
PY
}

attempts=0
while [[ "$(campaign_status)" != "finished" ]]; do
  while fuser "$LOCK" >/dev/null 2>&1; do sleep 15; done
  if [[ "$(campaign_status)" == "finished" ]]; then break; fi
  attempts=$((attempts + 1))
  if (( attempts > MAX_RESTARTS )); then
    write_state "failed" "$attempts" "자동 재개 허용 횟수 초과"
    exit 1
  fi
  write_state "running" "$attempts" "48개 의미 통합 조합 실험 및 최종 Test 자동 실행"
  echo "[$(date '+%F %T')] supervisor start $attempts/$MAX_RESTARTS" >> "$LOG"
  bash "$ROOT/run_ax_e6_semantic_completion.sh" >> "$LOG" 2>&1 &
  child=$!
  echo "$child" > "$PID_FILE"
  wait "$child" || true
  sleep 10
done

write_state "experiment_finished" "$attempts" "127개 순위와 최종 Test 산출물 생성 완료"
if [[ -x "$ROOT/finalize_ax_semantic_dashboard.sh" ]]; then
  if ! bash "$ROOT/finalize_ax_semantic_dashboard.sh" >> "$LOG" 2>&1; then
    write_state "dashboard_deploy_failed" "$attempts" "실험은 완료됐으나 대시보드 검증 또는 배포 실패"
    exit 1
  fi
fi
write_state "finished" "$attempts" "실험·대시보드 동기화·배포 완료"
