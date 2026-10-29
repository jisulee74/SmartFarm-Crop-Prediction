#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_REPO="${AX_DEPLOY_REPO:-$ROOT/../repo_deploy/repo}"
PYTHON_BIN="${AX_PYTHON:-$ROOT/../envs/jslee_py311/bin/python}"

exec 9>"$ROOT/ax_catalog_experiments/outputs/dashboard_publish.lock"
flock 9

cd "$ROOT/ax_data_validation"
if "$PYTHON_BIN" -c 'import json; from pathlib import Path; p=Path("cache/campaigns/semantic_control/experiment_index.json"); d=json.loads(p.read_text()); raise SystemExit(0 if d.get("campaign",{}).get("internal_campaign_id")=="exhaustive_e6_semantic_completion_v2" and d.get("campaign",{}).get("status")=="completed" else 1)'; then
  echo "Formal results already published; do not overwrite with screening"
  exit 0
fi
"$PYTHON_BIN" sync_semantic_screening_dashboard.py
"$PYTHON_BIN" sync_screening_analysis.py
"$PYTHON_BIN" -m pytest -q test_semantic_screening_static.py test_screening_analysis.py
node --check src/experiment_api.js
node --check src/experiment_view.js
node --check src/analysis_view.js

if [[ ! -d "$DEPLOY_REPO/.git" ]]; then
  echo "Deploy repository not found: $DEPLOY_REPO" >&2
  exit 3
fi

mkdir -p "$DEPLOY_REPO/ax_data_validation"
rsync -a --delete \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$ROOT/ax_data_validation/" "$DEPLOY_REPO/ax_data_validation/"

cd "$DEPLOY_REPO"
git add ax_data_validation
if git diff --cached --quiet; then
  echo "No dashboard changes to deploy"
  exit 0
fi
git commit -q -m "Update all AX diagnostic views from screening Test results"
git push

commit_sha="$(git rev-parse HEAD)"
public_url="${AX_PUBLIC_DASHBOARD_URL:-https://jisulee74.github.io/SmartFarm-Crop-Prediction/ax_data_validation/}"
index_url="${public_url%/}/ax_direction/dashboard_bundle.json"
expected_hash="$(sha256sum "$ROOT/ax_data_validation/ax_direction/dashboard_bundle.json" | cut -d " " -f 1)"
for attempt in $(seq 1 60); do
  if remote_hash="$(curl -fsSL "${index_url}?deploy=${commit_sha}" | sha256sum | cut -d " " -f 1)" && [[ "$remote_hash" == "$expected_hash" ]]; then
    echo "GitHub Pages screening dashboard verified: $public_url ($commit_sha)"
    exit 0
  fi
  sleep 15
done
echo "Git push succeeded but screening dashboard verification timed out: $public_url ($commit_sha)" >&2
exit 4
