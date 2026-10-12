#!/usr/bin/env python3
"""Integrate the completed E1/E2 three-seed campaign into the two dashboard views.

The completed deadline campaign uses fixed target/model configurations and the
facility-specific (raw-code) E6 features.  Consequently:

* all combinations containing E1 or E2 replace the unavailable rows in the
  stored-control-code view;
* combinations containing E1/E2 but not E6 are also reusable in the semantic
  view because their inputs are identical;
* combinations containing both E1/E2 and E6 remain outside the semantic-E6
  scope until they are actually retrained with semantic E6 features.

The script is idempotent and preserves the original experiment outputs.  It
only rewrites dashboard cache files and the KPI planning reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import sync_e1_e2_quick_dashboard as detail


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CACHE = HERE / "cache"
OUTPUT = WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_deadline_v1"
PREPARED = WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_backfill_v1"
BASE_INDEX = CACHE / "experiment_index.json"
SEMANTIC_INDEX = CACHE / "campaigns" / "semantic_control" / "experiment_index.json"
CATALOG = CACHE / "combination_catalog.json"
CAMPAIGN_CATALOG = CACHE / "campaign_catalog.json"
BASE_FEATURES = CACHE / "feature_selections"
SEMANTIC_FEATURES = CACHE / "campaigns" / "semantic_control" / "feature_selections"
KPI_BUNDLE = HERE / "ax_direction" / "dashboard_bundle.json"
MODELS = detail.MODELS
TARGETS = list(detail.TARGET_NAMES_KR)
SEEDS = [42, 52, 62]
SOURCE = "e1_e2_fixed_config_3seed"
SOURCE_LABEL = "E1·E2 보완 고정설정 3시드"
SEMANTIC_PENDING_REASON = (
    "E1·E2와 장치 의미 통합 E6를 함께 사용한 재학습은 아직 수행되지 않았습니다. "
    "현재 완료 결과의 E6는 시설별 제어코드 기준입니다."
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if pretty:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summary_records() -> dict[tuple[str, str, str], dict]:
    report = read_json(OUTPUT / "completion_report.json")
    if report.get("Completed") != 6096 or report.get("Total") != 6096:
        raise ValueError(f"Final E1/E2 campaign is incomplete: {report}")
    records = {}
    for path in sorted((OUTPUT / "summaries").glob("*/*/COMBO_*.json")):
        item = read_json(path)
        key = (str(item["Target_ID"]), str(item["Combination_ID"]), str(item["Model_Type"]))
        if key in records:
            raise ValueError(f"Duplicate summary: {key}")
        records[key] = item
    if len(records) != 6096:
        raise ValueError(f"Expected 6096 summaries, found {len(records)}")
    return records


def aggregate_runs(summary: dict, variant: str) -> dict:
    values = {metric: [] for metric in ("rmse", "mae", "r2", "ccc")}
    for run in summary.get("Runs") or []:
        block = run.get(variant) or {}
        for metric in values:
            value = finite(block.get(metric))
            if value is not None:
                values[metric].append(value)
    result = {}
    for metric, items in values.items():
        if items:
            result[f"{metric}_mean"] = statistics.fmean(items)
            result[f"{metric}_std"] = statistics.pstdev(items) if len(items) > 1 else 0.0
    return result


def result_from_summary(summary: dict, n_eval: int) -> dict:
    status = str(summary.get("Status", "unknown"))
    success_runs = [run for run in summary.get("Runs") or [] if run.get("Status") == "success"]
    return {
        "group_id": str(summary["Combination_ID"]),
        "model": str(summary["Model_Type"]),
        "split": "validation",
        "completed_seeds": len(success_runs),
        "status": status,
        "reason": summary.get("Reason"),
        "params": summary.get("Params") or {},
        "best_epoch": summary.get("Fixed_Epochs"),
        "n_eval": n_eval,
        "feature_count": summary.get("Feature_Count"),
        "bounded": aggregate_runs(summary, "Bounded") if status == "success" else {},
        "raw": aggregate_runs(summary, "Raw") if status == "success" else {},
        "seeds": {
            str(int(run["Seed"])): {"seed": int(run["Seed"]), "status": "success"}
            for run in success_runs
        },
        "source": SOURCE,
        "source_label": SOURCE_LABEL,
        "method": summary.get("Method"),
        "comparability": summary.get("Comparability"),
        "config_source_combination": summary.get("Config_Source_Combination"),
    }


def combo_groups(catalog: dict) -> dict[str, list[str]]:
    return {str(item["combination_id"]): list(item["groups"]) for item in catalog["combinations"]}


def evaluation_rows(target: dict) -> int:
    for result in target.get("results", {}).values():
        if result.get("split") == "validation" and int(result.get("n_eval") or 0) > 0:
            return int(result["n_eval"])
    return 0


def winner(target: dict, model: str) -> str | None:
    candidates = []
    for key, result in target.get("results", {}).items():
        combo_id, result_model, split = key.split("::")
        metric = result.get("bounded") or {}
        rmse = finite(metric.get("rmse_mean"))
        if result_model != model or split != "validation" or result.get("status") != "success" or rmse is None:
            continue
        mae = finite(metric.get("mae_mean"))
        feature_count = result.get("feature_count")
        candidates.append((
            rmse,
            mae if mae is not None else math.inf,
            int(feature_count) if feature_count is not None else math.inf,
            combo_id,
        ))
    return min(candidates)[3] if candidates else None


def refresh_winners(index: dict) -> None:
    for target in index["targets"].values():
        target["frozen"] = {"overall": None, "winners": {}}
        for model in MODELS:
            combo_id = winner(target, model)
            if combo_id:
                target["frozen"]["winners"][model] = {"group": combo_id}


def validation_status_counts(index: dict) -> Counter:
    return Counter(
        result.get("status", "unknown")
        for target in index["targets"].values()
        for result in target.get("results", {}).values()
        if result.get("split") == "validation"
    )


def build_feature_detail(target_id: str, combo: dict, summary: dict, labels: dict, feature_group: dict) -> dict:
    combo_id = str(combo["combination_id"])
    source = read_json(OUTPUT / "feature_selection" / target_id / f"{combo_id}.json")
    selected_ids = [
        item for item in source.get("selected_numeric", []) + source.get("selected_categorical", [])
        if item != "item_code"
    ]
    categorical = set(source.get("selected_categorical", []))
    selected = []
    for variable_id in selected_ids:
        item = detail._feature_parts(variable_id, feature_group.get(variable_id, "-"), labels)
        item["variable_type"] = "categorical" if variable_id in categorical else "numeric"
        selected.append(item)

    excluded = []
    for row in source.get("exclusions", []):
        variable_id = str(row.get("Variable_ID", ""))
        if not variable_id or variable_id == "item_code":
            continue
        item = detail._feature_parts(variable_id, feature_group.get(variable_id, "-"), labels)
        related = row.get("Related_Variable")
        related_item = (
            detail._feature_parts(str(related), feature_group.get(str(related), "-"), labels)
            if related else None
        )
        reason = str(row.get("Reason", "unknown"))
        item.update(
            reason=reason,
            reason_label=detail.REASON_LABELS.get(reason, reason),
            missing_rate=finite(row.get("Missing_Rate")),
            related_variable=str(related) if related else None,
            related_variable_name=related_item.get("source_name") if related_item else None,
            correlation=finite(row.get("Correlation")),
            iteration=int(row["Iteration"]) if row.get("Iteration") is not None else None,
            vif=finite(row.get("VIF")),
            vif_is_infinite=bool(row.get("VIF_Is_Infinite", False)) or str(row.get("VIF", "")).lower() == "inf",
        )
        excluded.append(item)

    reasons = Counter(item["reason"] for item in excluded)
    status = "success" if selected else str(summary.get("Status", "ineligible"))
    return {
        "combination_id": combo_id,
        "label": combo["label"],
        "groups": combo["groups"],
        "status": status,
        "reason": None if selected else summary.get("Reason"),
        "stage": "validation",
        "fold": None,
        "fit_hash": None,
        "candidate_count": len(selected) + len(excluded),
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "selected": selected,
        "excluded": excluded,
        "exclusion_summary": [
            {"reason": reason, "reason_label": detail.REASON_LABELS.get(reason, reason), "count": count}
            for reason, count in reasons.most_common()
        ],
        "source_label": f"{SOURCE_LABEL} 검증 학습 구간 변수 선택 기록",
        "exclusion_note": "같은 타깃·조합의 6개 모델이 공통으로 사용한 변수 선택 결과입니다.",
    }


def merge_feature_selections(records: dict, catalog: dict, groups: dict) -> None:
    labels = detail._source_labels()
    prepared = read_json(PREPARED / "prepared_manifest.json")
    combo_lookup = {item["combination_id"]: item for item in catalog["combinations"]}
    for target_id in TARGETS:
        target_groups = prepared["targets"][target_id].get("groups", {})
        feature_group = {
            variable: group for group, variables in target_groups.items() for variable in variables
        }
        details = {}
        for combo_id, combo in combo_lookup.items():
            if not ({"E1", "E2"} & set(groups[combo_id])):
                continue
            summary = records[(target_id, combo_id, MODELS[0])]
            details[combo_id] = build_feature_detail(target_id, combo, summary, labels, feature_group)

        base_path = BASE_FEATURES / f"{target_id}.json"
        base = read_json(base_path)
        base["campaign"] = "stored_control_codes_with_e1_e2_backfill"
        base["combinations"].update(deepcopy(details))
        write_json(base_path, base)

        semantic_path = SEMANTIC_FEATURES / f"{target_id}.json"
        semantic = read_json(semantic_path)
        for combo_id, item in details.items():
            combo_set = set(groups[combo_id])
            if "E6" not in combo_set:
                reused = deepcopy(item)
                reused["source_label"] = f"{SOURCE_LABEL} 결과 재사용(E6 미포함)"
                semantic["combinations"][combo_id] = reused
            else:
                pending = semantic["combinations"].setdefault(combo_id, {})
                pending.update({
                    "combination_id": combo_id,
                    "label": combo_lookup[combo_id]["label"],
                    "groups": groups[combo_id],
                    "status": "not_in_scope",
                    "reason": SEMANTIC_PENDING_REASON,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "excluded_count": 0,
                    "selected": [],
                    "excluded": [],
                    "exclusion_summary": [],
                    "source_label": "장치 의미 통합 E6 재학습 필요",
                })
        write_json(semantic_path, semantic)


def merge_indexes(records: dict, catalog: dict, groups: dict) -> tuple[dict, dict]:
    base = read_json(BASE_INDEX)
    semantic = read_json(SEMANTIC_INDEX)
    for target_id in TARGETS:
        base_target = base["targets"][target_id]
        semantic_target = semantic["targets"][target_id]
        n_eval = evaluation_rows(base_target)
        for combo_id, combo_group_list in groups.items():
            combo_set = set(combo_group_list)
            if not ({"E1", "E2"} & combo_set):
                continue
            for model in MODELS:
                key = f"{combo_id}::{model}::validation"
                converted = result_from_summary(records[(target_id, combo_id, model)], n_eval)
                base_target["results"][key] = converted
                if "E6" not in combo_set:
                    reused = deepcopy(converted)
                    reused["source"] = "reused_e1_e2_fixed_config_no_e6"
                    reused["source_label"] = f"{SOURCE_LABEL} 결과 재사용(E6 미포함)"
                    semantic_target["results"][key] = reused
                else:
                    semantic_target["results"][key] = {
                        "group_id": combo_id,
                        "model": model,
                        "split": "validation",
                        "completed_seeds": 0,
                        "status": "not_in_scope",
                        "reason": SEMANTIC_PENDING_REASON,
                        "params": {},
                        "bounded": {},
                        "raw": {},
                        "seeds": {},
                        "source": "semantic_e1_e2_e6_not_run",
                        "source_label": "장치 의미 통합 E6 재학습 필요",
                    }
            base_target["groups"][combo_id]["campaign_role"] = "e1_e2_backfill"
            semantic_target["groups"][combo_id]["campaign_role"] = (
                "reused_e1_e2_no_e6" if "E6" not in combo_set else "not_in_scope"
            )
        base_target["status"] = "completed"
        semantic_target["status"] = "completed"

    refresh_winners(base)
    refresh_winners(semantic)
    now = datetime.now(timezone.utc).isoformat()
    base_counts = validation_status_counts(base)
    base["campaign"].update({
        "id": "stored_control_codes_with_e1_e2_backfill",
        "name": "시설별 제어코드 기준",
        "description": "시설별 제어코드 기준 결과에 E1·E2 보완 고정설정 3시드 결과를 통합한 127개 조합 비교",
        "status": "completed",
        "partial_results": False,
        "snapshot_generated_at": now,
        "snapshot_note": (
            "E1·E2 포함 96개 조합은 타깃·모델별 고정 하이퍼파라미터를 사용한 3시드 결과입니다. "
            "나머지 조합은 기존 조합별 튜닝 결과를 보존합니다."
        ),
        "method": "hybrid_original_cv_and_e1_e2_fixed_configuration_3seed",
        "e1_e2_backfill_campaign": "exhaustive_e1_e2_deadline_v1",
        "e1_e2_backfilled_combinations": 96,
        "processed_combination_summaries": 6096,
        "progress_percent": 100.0,
        "summary_status_counts": dict(sorted(base_counts.items())),
        "published_result_groups": 6096,
        "published_success_groups": base_counts.get("success", 0),
    })

    semantic_counts = validation_status_counts(semantic)
    semantic["campaign"].update({
        "status": "completed",
        "partial_results": False,
        "snapshot_generated_at": now,
        "snapshot_note": (
            "장치 의미 통합 E6 실험 16개 조합과 E6 미포함 결과 63개 조합을 제공합니다. "
            "E1·E2와 의미 통합 E6를 함께 쓰는 48개 조합은 재학습 전입니다."
        ),
        "scope_note": (
            "장치 의미 통합 E6를 사용한 16개 조합을 신규 학습하고, E6가 없는 63개 조합은 "
            "시설별 코드 기준 또는 E1·E2 보완 3시드 결과를 재사용합니다. E1·E2와 E6를 함께 "
            "포함한 48개 조합은 의미 통합 기준 재학습 전이므로 결과를 표시하지 않습니다."
        ),
        "reused_combinations": 63,
        "out_of_scope_combinations": 48,
        "e1_e2_reused_no_e6_combinations": 48,
        "summary_status_counts": dict(sorted(semantic_counts.items())),
        "published_success_groups": semantic_counts.get("success", 0),
        "baseline_index_sha256": None,
        "integrated_baseline_index_sha256": "pending",
    })

    write_json(BASE_INDEX, base)
    semantic["campaign"]["integrated_baseline_index_sha256"] = hashlib.sha256(BASE_INDEX.read_bytes()).hexdigest()
    write_json(SEMANTIC_INDEX, semantic)
    return base, semantic


def update_campaign_catalog(base: dict, semantic: dict) -> None:
    base_counts = base["campaign"]["summary_status_counts"]
    semantic_counts = semantic["campaign"]["summary_status_counts"]
    payload = {
        "default_campaign": "stored_control_codes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaigns": [
            {
                "id": "stored_control_codes",
                "internal_campaign_id": "exhaustive_e1_e7_v1+exhaustive_e1_e2_deadline_v1",
                "name": "시설별 제어코드 기준",
                "short_name": "시설별 코드 기준",
                "description": "시설별 제어코드를 유지하고 E1·E2 보완 3시드 결과를 통합한 127개 조합 비교",
                "why_separate": "시설별로 저장된 원 제어코드를 그대로 사용한 기준입니다.",
                "scope_note": (
                    "E1·E2 포함 96개 조합은 고정설정 3시드로 보완했고, 나머지는 기존 조합별 튜닝 결과를 보존합니다. "
                    "두 결과의 하이퍼파라미터 탐색 방식은 서로 다릅니다."
                ),
                "status": "completed",
                "progress_percent": 100.0,
                "completed_results": sum(base_counts.values()),
                "successful_results": base_counts.get("success", 0),
                "failed_results": base_counts.get("failed", 0),
                "total_results": 6096,
                "total_seeds": 3,
                "seeds": SEEDS,
                "data_path": "./cache/experiment_index.json",
                "predictions_path": "./cache/predictions",
                "feature_selections_path": "./cache/feature_selections",
            },
            {
                "id": "semantic_control",
                "internal_campaign_id": "exhaustive_e6_semantic_v2+e1_e2_no_e6_reuse",
                "name": "장치 의미 통합 기준",
                "short_name": "장치 의미 통합",
                "description": "방향·층수 표현을 통합한 E6 결과와 E6 미포함 E1·E2 보완 결과를 함께 제공",
                "why_separate": semantic["campaign"]["why_separate"],
                "scope_note": semantic["campaign"]["scope_note"],
                "status": "completed",
                "progress_percent": 100.0,
                "completed_results": sum(semantic_counts.values()) - semantic_counts.get("not_in_scope", 0),
                "successful_results": semantic_counts.get("success", 0),
                "out_of_scope_results": semantic_counts.get("not_in_scope", 0),
                "total_results": 6096,
                "total_seeds": 3,
                "seeds": SEEDS,
                "completed_lanes": semantic["campaign"].get("completed_lanes", 48),
                "total_lanes": semantic["campaign"].get("total_lanes", 48),
                "remaining_lanes": 0,
                "data_path": "./cache/campaigns/semantic_control/experiment_index.json",
                "predictions_path": "./cache/campaigns/semantic_control/predictions",
                "baseline_predictions_path": "./cache/predictions",
                "feature_selections_path": "./cache/campaigns/semantic_control/feature_selections",
            },
        ],
    }
    write_json(CAMPAIGN_CATALOG, payload, pretty=True)


def update_kpi(records: dict) -> None:
    bundle = read_json(KPI_BUNDLE)
    rows = bundle["tables"]["kpi_planning_reference"]
    by_target: dict[str, list[tuple]] = {target: [] for target in TARGETS}
    for (target_id, combo_id, model), summary in records.items():
        if summary.get("Status") != "success":
            continue
        rmse = finite(summary.get("Validation_RMSE_Mean"))
        mae = finite(summary.get("Validation_MAE_Mean"))
        if rmse is None:
            continue
        by_target[target_id].append((
            rmse,
            mae if mae is not None else math.inf,
            int(summary.get("Feature_Count")) if summary.get("Feature_Count") is not None else math.inf,
            combo_id,
            model,
            list(summary.get("Variable_Groups") or []),
        ))
    snapshot = "2026-10-10 14:41 KST"
    for row in rows:
        best = min(by_target[row["target"]])
        row.update({
            "reference_rmse": best[0],
            "illustrative_10pct_rmse": best[0] * 0.9,
            "status": "completed fixed-configuration three-seed best",
            "reference_model": best[4],
            "reference_groups": best[5],
            "reference_combination_id": best[3],
            "reference_is_provisional": False,
            "reference_snapshot_label": snapshot,
            "reference_campaign_progress": {"processed": 6096, "total": 6096, "percent": 100.0},
            "reference_method_label": "E1·E2 보완 고정설정 3시드",
        })
    write_json(KPI_BUNDLE, bundle)


def main() -> None:
    records = summary_records()
    catalog = read_json(CATALOG)
    groups = combo_groups(catalog)
    base, semantic = merge_indexes(records, catalog, groups)
    merge_feature_selections(records, catalog, groups)
    update_campaign_catalog(base, semantic)
    update_kpi(records)
    print("Integrated E1/E2 final campaign into stored-control and semantic-control dashboard views")
    print("Stored-control validation statuses:", dict(validation_status_counts(base)))
    print("Semantic-control validation statuses:", dict(validation_status_counts(semantic)))


if __name__ == "__main__":
    main()
