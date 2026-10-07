#!/usr/bin/env python3
"""Build the public interim E1/E2 three-seed validation snapshot.

The snapshot is deliberately separated from the final campaign publication.  It
only compares models whose 127-combination lanes are complete and labels every
number as provisional until all six models finish.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CAMPAIGN_DIR = (
    WORKSPACE
    / "ax_catalog_experiments"
    / "outputs"
    / "exhaustive_e1_e2_deadline_v1"
)
BUNDLE_PATH = HERE / "ax_direction" / "dashboard_bundle.json"

TOTAL_COMBINATIONS_PER_MODEL = 8 * 127
TOTAL_RUNS = TOTAL_COMBINATIONS_PER_MODEL * 6
E12 = {"E1", "E2"}

TARGET_LABELS = {
    "tomato_first": "토마토 1화방 꽃수",
    "tomato_second": "토마토 2화방 꽃수",
    "tomato_third": "토마토 3화방 꽃수",
    "tomato_sum123": "토마토 1~3화방 합계 꽃수",
    "strawberry_first": "딸기 1화방 착과수",
    "strawberry_second": "딸기 2화방 착과수",
    "strawberry_third": "딸기 3화방 착과수",
    "strawberry_sum123": "딸기 1~3화방 합계 착과수",
}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _candidate_key(item: dict) -> tuple:
    groups = item["Variable_Groups"]
    return (
        item["Validation_RMSE_Mean"],
        len(groups),
        "+".join(groups),
        item["Model_Type"],
    )


def _best(items: list[dict], with_e12: bool) -> dict:
    candidates = [
        item
        for item in items
        if bool(E12.intersection(item["Variable_Groups"])) is with_e12
    ]
    if not candidates:
        raise RuntimeError(f"No candidate found (with_e12={with_e12})")
    return min(candidates, key=_candidate_key)


def build_snapshot() -> dict:
    summary_root = CAMPAIGN_DIR / "summaries"
    files = list(summary_root.glob("*/*/COMBO_*.json"))
    model_counts = Counter(path.parent.name for path in files)
    completed_models = sorted(
        model for model, count in model_counts.items() if count == TOTAL_COMBINATIONS_PER_MODEL
    )
    pending_models = sorted(set(["poisson", "random_forest", "catboost", "mlp", "tabm", "tft"]) - set(completed_models))

    by_target_model: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_target: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("Model_Type") not in completed_models:
            continue
        if item.get("Status") != "success" or not _finite(item.get("Validation_RMSE_Mean")):
            continue
        by_target_model[(item["Target_ID"], item["Model_Type"])].append(item)
        by_target[item["Target_ID"]].append(item)

    pair_improvements: list[float] = []
    for key, items in sorted(by_target_model.items()):
        baseline = _best(items, with_e12=False)
        enhanced = _best(items, with_e12=True)
        pair_improvements.append(
            (baseline["Validation_RMSE_Mean"] - enhanced["Validation_RMSE_Mean"])
            / baseline["Validation_RMSE_Mean"]
            * 100
        )

    target_results = []
    for target_id in TARGET_LABELS:
        items = by_target[target_id]
        baseline = _best(items, with_e12=False)
        enhanced = _best(items, with_e12=True)
        baseline_rmse = baseline["Validation_RMSE_Mean"]
        enhanced_rmse = enhanced["Validation_RMSE_Mean"]
        improvement = (baseline_rmse - enhanced_rmse) / baseline_rmse * 100
        target_results.append(
            {
                "target": target_id,
                "target_label": TARGET_LABELS[target_id],
                "baseline_rmse": baseline_rmse,
                "baseline_model": baseline["Model_Type"],
                "baseline_groups": baseline["Variable_Groups"],
                "e12_rmse": enhanced_rmse,
                "e12_model": enhanced["Model_Type"],
                "e12_groups": enhanced["Variable_Groups"],
                "rmse_improvement_pct": improvement,
                "e12_improved": improvement > 0,
            }
        )

    now = datetime.now(ZoneInfo("Asia/Seoul"))
    processed = len(files)
    return {
        "campaign_id": "exhaustive_e1_e2_deadline_v1",
        "status": "in_progress",
        "snapshot_at_kst": now.isoformat(timespec="minutes"),
        "snapshot_label": now.strftime("%Y-%m-%d %H:%M KST"),
        "method": "fixed_configuration_three_seed_validation",
        "seeds": [42, 52, 62],
        "processed": processed,
        "total": TOTAL_RUNS,
        "remaining": TOTAL_RUNS - processed,
        "progress_percent": round(processed / TOTAL_RUNS * 100, 2),
        "model_counts": dict(sorted(model_counts.items())),
        "completed_models": completed_models,
        "pending_models": pending_models,
        "completed_model_count": len(completed_models),
        "total_model_count": 6,
        "comparison_pair_count": len(pair_improvements),
        "e12_improved_pair_count": sum(value > 0 for value in pair_improvements),
        "e12_improved_pair_rate": round(sum(value > 0 for value in pair_improvements) / len(pair_improvements) * 100, 2),
        "median_rmse_improvement_pct": statistics.median(pair_improvements),
        "mean_rmse_improvement_pct": statistics.mean(pair_improvements),
        "target_results": target_results,
        "limitations": [
            "TFT가 진행 중이므로 최종 우승 모델·조합과 기준 RMSE는 변경될 수 있음",
            "현재 산출물은 조합별 요약 지표이며 행별 관측값·예측값은 최종 우승 모델 확정 후 별도 생성",
            "E1·E2 효과는 기존 생육 이력 활용 개선이며 신규 AX 수집 데이터의 효과가 아님",
            "신규 AX 데이터 효과는 동일 신규 코호트에서 입력 추가 전후를 다시 비교해야 함",
        ],
    }


def main() -> None:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(BUNDLE_PATH)
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    snapshot = build_snapshot()
    bundle.setdefault("tables", {})["interim_e1_e2_validation"] = snapshot
    BUNDLE_PATH.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"updated {BUNDLE_PATH}: {snapshot['processed']}/{snapshot['total']} "
        f"({snapshot['progress_percent']}%), completed models={snapshot['completed_models']}"
    )


if __name__ == "__main__":
    main()
