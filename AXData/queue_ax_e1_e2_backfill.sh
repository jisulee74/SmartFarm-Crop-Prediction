#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E6_OUTPUT="$ROOT/ax_catalog_experiments/outputs/exhaustive_e6_semantic_v2"
OUTPUT="$ROOT/ax_catalog_experiments/outputs/exhaustive_e1_e2_backfill_v1"
LOG="$ROOT/ax_e1_e2_backfill.log"
PID_FILE="$ROOT/ax_e1_e2_backfill.pid"
QUEUE_STATE="$OUTPUT/queue_status.json"
MAX_RESTARTS=5
mkdir -p "$OUTPUT"
echo "$$" > "$PID_FILE"

write_state() {
  python - "$QUEUE_STATE" "$1" "$2" <<'PY'
import json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1]); tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'Status':sys.argv[2],'Message':sys.argv[3],'Updated_At':time.time()},ensure_ascii=False,indent=2))
os.replace(tmp,p)
PY
}

e6_status() {
  python - "$E6_OUTPUT/scheduler.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
try: print(json.loads(p.read_text()).get('Status','missing'))
except Exception: print('missing')
PY
}

write_state "waiting_for_e6" "E6 의미 통합 캠페인 완료 후 자동 시작 대기"
while [[ "$(e6_status)" != "finished" ]]; do sleep 30; done

backfill_status() {
  python - "$OUTPUT/scheduler.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
try: print(json.loads(p.read_text()).get('Status','missing'))
except Exception: print('missing')
PY
}

attempts=0
while [[ "$(backfill_status)" != "finished" ]]; do
  attempts=$((attempts + 1))
  if (( attempts > MAX_RESTARTS )); then
    write_state "failed" "자동 재개 허용 횟수 초과"
    exit 1
  fi
  write_state "running" "E1·E2 보완 캠페인 실행 중 (시도 $attempts/$MAX_RESTARTS)"
  echo "[$(date '+%F %T')] E1/E2 backfill start $attempts/$MAX_RESTARTS" >> "$LOG"
  bash "$ROOT/run_ax_e1_e2_backfill.sh" >> "$LOG" 2>&1 || true
  sleep 10
done
write_state "finished" "E1·E2 보완 캠페인 완료"
