#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${AX_PYTHON:-$ROOT/../envs/jslee_py311/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
cd "$ROOT/.."
"$PYTHON_BIN" -u -m AXData.ax_catalog_experiments.exhaustive_e1_e2_backfill all "$@"
