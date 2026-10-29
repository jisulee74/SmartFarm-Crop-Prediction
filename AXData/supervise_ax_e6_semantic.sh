#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_v2"
LOCK="$OUTPUT/campaign.lock"
LOG="$ROOT/ax_e6_semantic.log"
PID_FILE="$ROOT/ax_e6_semantic.pid"
STATE_FILE="$OUTPUT/supervisor_status.json"
MAX_RESTARTS=5

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
p=Path(sys.argv[1]); tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'Status':sys.argv[2],'Restart_Attempts':int(sys.argv[3]),'Message':sys.argv[4],'Updated_At':time.time()},ensure_ascii=False,indent=2))
os.replace(tmp,p)
PY
}

initial_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
write_state "monitoring" 0 "현재 실행 종료를 감시 중"
while [[ -n "$initial_pid" ]] && kill -0 "$initial_pid" 2>/dev/null; do sleep 30; done

attempts=0
while [[ "$(campaign_status)" != "finished" ]]; do
  while fuser "$LOCK" >/dev/null 2>&1; do sleep 10; done
  if [[ "$(campaign_status)" == "finished" ]]; then break; fi
  attempts=$((attempts + 1))
  if (( attempts > MAX_RESTARTS )); then
    write_state "failed" "$attempts" "자동 재개 허용 횟수 초과"
    exit 1
  fi
  write_state "restarting" "$attempts" "미완료 캠페인 자동 재개"
  echo "[$(date '+%F %T')] supervisor restart $attempts/$MAX_RESTARTS" >> "$LOG"
  bash "$ROOT/run_ax_e6_semantic.sh" >> "$LOG" 2>&1 &
  child=$!
  echo "$child" > "$PID_FILE"
  wait "$child" || true
  sleep 10
done

python "$ROOT/flower_fruit_prediction_exhaustive/sync_e6_semantic_dashboard.py" >> "$LOG" 2>&1
write_state "finished" "$attempts" "실험 및 대시보드 최종 동기화 완료"
