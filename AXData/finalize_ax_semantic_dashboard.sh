#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_REPO="${AX_DEPLOY_REPO:-$ROOT/../repo_deploy/repo}"
PYTHON_BIN="${AX_PYTHON:-$ROOT/../envs/jslee_py311/bin/python}"

exec 9>"$ROOT/ax_catalog_experiments/outputs/dashboard_publish.lock"
flock 9

cd "$ROOT/ax_data_validation"
"$PYTHON_BIN" sync_semantic_final_dashboard.py
"$PYTHON_BIN" -m pytest -q test_semantic_final_static.py
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
git commit -m "Publish final semantic AX validation results"
git push

commit_sha="$(git rev-parse HEAD)"
public_url="${AX_PUBLIC_DASHBOARD_URL:-https://jisulee74.github.io/SmartFarm-Crop-Prediction/ax_data_validation/}"
for attempt in $(seq 1 60); do
  if curl -fsSL "${public_url}?deploy=${commit_sha}" | grep -F "127개 조합의 성능과 최종 Test 예측" >/dev/null; then
    echo "GitHub Pages verified: $public_url ($commit_sha)"
    exit 0
  fi
  sleep 15
done
echo "Git push succeeded but GitHub Pages verification timed out: $public_url ($commit_sha)" >&2
exit 4
