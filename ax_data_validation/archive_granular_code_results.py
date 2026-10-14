#!/usr/bin/env python3
"""Create a compact, checksum-protected GitHub archive of raw-code results."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEST = HERE / "technical_archive" / "granular_control_codes_v1"
RAW_ROOT = WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1"
BACKFILL_ROOT = WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_deadline_v1"
INDEX = HERE / "cache" / "experiment_index.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build() -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    copies = {
        RAW_ROOT / "campaign_manifest.json": DEST / "original_campaign_manifest.json",
        RAW_ROOT / "combination_catalog.csv": DEST / "combination_catalog.csv",
        RAW_ROOT / "data_quality.csv": DEST / "data_quality.csv",
        BACKFILL_ROOT / "campaign_manifest.json": DEST / "e1_e2_backfill_campaign_manifest.json",
        BACKFILL_ROOT / "completion_report.json": DEST / "e1_e2_backfill_completion_report.json",
    }
    for source, destination in copies.items():
        if source.exists():
            shutil.copy2(source, destination)

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    rows = []
    for target_id, target in index["targets"].items():
        for key, result in target.get("results", {}).items():
            combination_id, model, split = key.split("::")
            if split != "validation":
                continue
            metric = result.get("bounded") or {}
            rows.append(
                {
                    "Target_ID": target_id,
                    "Model_Type": model,
                    "Combination_ID": combination_id,
                    "Variable_Groups": "|".join(target["groups"][combination_id].get("category_ids", [])),
                    "Status": result.get("status"),
                    "Source": result.get("source"),
                    "Completed_Seeds": result.get("completed_seeds", 0),
                    "Feature_Count": result.get("feature_count"),
                    "Validation_RMSE_Mean": metric.get("rmse_mean"),
                    "Validation_RMSE_Std": metric.get("rmse_std"),
                    "Validation_MAE_Mean": metric.get("mae_mean"),
                    "Validation_MAE_Std": metric.get("mae_std"),
                    "Validation_R2_Mean": metric.get("r2_mean"),
                    "Validation_R2_Std": metric.get("r2_std"),
                    "Validation_CCC_Mean": metric.get("ccc_mean"),
                    "Validation_CCC_Std": metric.get("ccc_std"),
                }
            )
    fields = list(rows[0])
    with (DEST / "validation_results_compact.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    readme = """# 시설별 세분 제어코드 실험 기술 보관본

이 폴더는 대시보드에서 제외한 `방향·층수 세분 코드` 기준 결과의 재현·감사용 기술 자료입니다.

- 원 데이터의 시설별 제어 코드를 그대로 구분한 기준 실험입니다.
- 방향·층수 조합이 시설마다 달라 E6 후보가 지나치게 희소해졌고, 원 코드 E6 후보 408개가 변수 선택 단계에서 모두 제외되는 문제가 확인됐습니다.
- 사용자용 대시보드는 시설 간 비교가 가능한 `장치 의미 통합` 결과만 제공합니다.
- 원 코드와 원 실험 결과는 삭제하거나 덮어쓰지 않았습니다. 대용량 run/checkpoint 파일은 서버의 고정 캠페인 출력에 보존합니다.

## 파일

- `original_campaign_manifest.json`: 최초 전수실험의 고정 설정과 해시
- `e1_e2_backfill_campaign_manifest.json`: E1·E2 보완 실험 설정
- `validation_results_compact.csv`: 8개 타깃·6개 모델·127개 조합의 세분 코드 Validation 요약
- `combination_catalog.csv`: 조합 ID와 변수군 구성
- `data_quality.csv`: 원 데이터 품질 요약
- `checksums.json`: 이 보관본의 SHA-256
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in DEST.iterdir() if path.is_file() and path.name != "checksums.json")
    checksums = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in files},
    }
    (DEST / "checksums.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return DEST


if __name__ == "__main__":
    print(build())

