#!/usr/bin/env python3
"""Publish the live E6 semantic-control campaign without changing the v1 cache.

The dashboard presents two human-facing experiment bases:

* stored_control_codes: the completed experiment using facility control codes as stored
* semantic_control: the follow-up experiment that unifies direction/layer codes by
  device meaning while preserving device type and measurement meaning

Only the 16 valid combinations containing E6 are retrained.  The 15 valid
non-E6 combinations are copied from the completed baseline so that the live
campaign can be ranked against the same alternatives.  Combinations involving
E1/E2 remain visible as out of scope for this follow-up campaign.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import sync_exhaustive_experiments as base
import sync_feature_selection_details as feature_detail


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
V1_INDEX = HERE / "cache" / "experiment_index.json"
CATALOG = HERE / "cache" / "combination_catalog.json"
CAMPAIGN_ROOT = HERE / "cache" / "campaigns" / "semantic_control"
INDEX_FILE = CAMPAIGN_ROOT / "experiment_index.json"
PREDICTIONS_DIR = CAMPAIGN_ROOT / "predictions"
FEATURE_SELECTION_DIR = CAMPAIGN_ROOT / "feature_selections"
CAMPAIGN_CATALOG = HERE / "cache" / "campaign_catalog.json"
V2_OUTPUT = WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e6_semantic_v2"

SEMANTIC_SCOPE_REASON = "시설 제어(E6)를 장치 의미로 통합해 다시 학습하는 16개 조합"
OUT_OF_SCOPE_REASON = "E6 의미 통합 재실험 범위 외: E1·E2 준비 보완 후 별도 실험 예정"


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _groups_by_id(catalog: dict) -> dict[str, list[str]]:
    return {item["combination_id"]: list(item["groups"]) for item in catalog["combinations"]}


def _is_semantic_scope(groups: list[str]) -> bool:
    return "E6" in groups and "E1" not in groups and "E2" not in groups


def _is_reused_scope(groups: list[str]) -> bool:
    return "E6" not in groups and "E1" not in groups and "E2" not in groups


def _build_raw_v2_index() -> dict:
    """Reuse the proven v1 publisher against the separate v2 output/cache paths."""
    saved = {
        "OUTPUTS_BASE": base.OUTPUTS_BASE,
        "CACHE_DIR": base.CACHE_DIR,
        "PREDICTIONS_CACHE_DIR": base.PREDICTIONS_CACHE_DIR,
        "INDEX_FILE": base.INDEX_FILE,
        "_index_cache": base._index_cache,
    }
    try:
        base.OUTPUTS_BASE = V2_OUTPUT
        base.CACHE_DIR = CAMPAIGN_ROOT
        base.PREDICTIONS_CACHE_DIR = PREDICTIONS_DIR
        base.INDEX_FILE = INDEX_FILE
        base._index_cache = None
        return base.build_index()
    finally:
        for key, value in saved.items():
            setattr(base, key, value)


def _validation_runs() -> dict[tuple[str, str, str], dict[int, dict]]:
    runs: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for path in V2_OUTPUT.glob("runs/*/result.json"):
        item = _read_json(path)
        if not item or item.get("Status") != "success" or item.get("Stage") != "validation":
            continue
        try:
            seed = int(item["Seed"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(item.get("Target_ID")), str(item.get("Combination_ID")), str(item.get("Model_Type")))
        runs[key][seed] = item
    return runs


def _winner(target: dict, model: str) -> str | None:
    candidates = []
    for key, result in target.get("results", {}).items():
        combo, result_model, split = key.split("::")
        rmse = result.get("bounded", {}).get("rmse_mean")
        if result_model == model and split == "validation" and rmse is not None:
            candidates.append((float(rmse), float(result.get("bounded", {}).get("mae_mean", float("inf"))), combo))
    return min(candidates)[2] if candidates else None


def _semantic_feature_lookup() -> dict[str, dict]:
    path = V2_OUTPUT / "semantic_feature_catalog.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path).fillna("")
    return {str(row.Variable_ID): row._asdict() for row in frame.itertuples(index=False)}


SEMANTIC_STAT_LABELS = {
    "active_fraction_mean": "장치별 작동시간 비율의 평균",
    "active_fraction_max": "장치별 작동시간 비율의 최댓값",
    "reporting_device_count": "기록 장치 수",
    "connection_rate": "데이터 연결률",
    "facility_mean": "시설 평균",
    "overall_max": "전체 최댓값",
    "overall_min": "전체 최솟값",
    "within_device_std": "장치 내 변동성",
    "mean": "평균",
    "std": "표준편차",
    "min": "최솟값",
    "max": "최댓값",
}


def _selected_item(variable_id: str, group: str, variable_type: str, semantic_lookup: dict[str, dict], source_labels: dict) -> dict:
    semantic = semantic_lookup.get(variable_id)
    if not semantic:
        item = feature_detail.feature_parts(variable_id, group, source_labels)
        item["variable_type"] = variable_type
        return item

    days = int(semantic.get("Window_Days") or 0)
    statistic = str(semantic.get("Statistic") or "")
    device = str(semantic.get("Device_Family_KR") or semantic.get("Device_Family") or "제어 장치")
    measurement = str(semantic.get("Measurement_Type_KR") or semantic.get("Measurement_Type") or "측정값")
    stat_kr = SEMANTIC_STAT_LABELS.get(statistic, statistic)
    source_name = f"{device} {measurement}"
    return {
        "variable_id": variable_id,
        "group": "E6",
        "base_variable": f"{semantic.get('Device_Family')} · {semantic.get('Measurement_Type')}",
        "window": f"직전 {days}일" if days else "예측 기준일 이전",
        "statistic": stat_kr,
        "source_name": source_name,
        "unit": "%" if semantic.get("Measurement_Type") == "opening_percent" else "",
        "meaning": f"예측 기준일 직전 {days}일 동안 방향·층수를 통합한 {source_name}의 {stat_kr}",
        "variable_type": variable_type,
    }


def _validation_selection_audit(target_id: str, combo_id: str) -> dict | None:
    directory = V2_OUTPUT / "feature_selection" / target_id / combo_id
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        item = _read_json(path)
        if item and item.get("identity", {}).get("stage") == "validation":
            return item
    return None


def _clean_float(value):
    try:
        value = float(value)
        return value if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def _build_feature_selection_cache(index: dict, catalog: dict, runs: dict) -> None:
    FEATURE_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    semantic_lookup = _semantic_feature_lookup()
    source_labels = feature_detail.load_source_labels()
    groups_by_id = _groups_by_id(catalog)
    prepared_manifest = _read_json(V2_OUTPUT / "prepared_manifest.json", {"targets": {}})

    for target_id in base.TARGET_NAMES_KR:
        target_groups = prepared_manifest.get("targets", {}).get(target_id, {}).get("groups", {})
        feature_group = {variable: group for group, variables in target_groups.items() for variable in variables}
        v1_path = HERE / "cache" / "feature_selections" / f"{target_id}.json"
        v1_payload = _read_json(v1_path, {"combinations": {}})
        combinations = {}
        for combo in catalog["combinations"]:
            combo_id = combo["combination_id"]
            groups = groups_by_id[combo_id]
            if _is_reused_scope(groups):
                item = copy.deepcopy(v1_payload.get("combinations", {}).get(combo_id, {}))
                item["source_label"] = "시설별 제어코드 기준 결과 재사용"
                combinations[combo_id] = item
                continue
            if not _is_semantic_scope(groups):
                combinations[combo_id] = {
                    "combination_id": combo_id,
                    "label": combo["label"],
                    "groups": groups,
                    "status": "not_in_scope",
                    "reason": OUT_OF_SCOPE_REASON,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "excluded_count": 0,
                    "selected": [],
                    "excluded": [],
                    "exclusion_summary": [],
                }
                continue

            audit = _validation_selection_audit(target_id, combo_id)
            selected = []
            excluded = []
            fit_hash = None
            if audit:
                fit_hash = audit.get("identity", {}).get("fit_hash")
                numeric = set(audit.get("selected_numeric") or [])
                categorical = set(audit.get("selected_categorical") or [])
                for variable_id in sorted(numeric | categorical):
                    variable_type = "categorical" if variable_id in categorical else "numeric"
                    group = "E6" if variable_id.startswith("E6S__") else feature_group.get(variable_id, "-")
                    selected.append(_selected_item(variable_id, group, variable_type, semantic_lookup, source_labels))
                for row in audit.get("exclusions") or []:
                    variable_id = str(row.get("Variable_ID", ""))
                    group = "E6" if variable_id.startswith("E6S__") else feature_group.get(variable_id, "-")
                    item = _selected_item(variable_id, group, "numeric", semantic_lookup, source_labels)
                    related = row.get("Related_Variable")
                    related_item = _selected_item(str(related), "E6" if str(related).startswith("E6S__") else feature_group.get(str(related), "-"), "numeric", semantic_lookup, source_labels) if related else None
                    reason = str(row.get("Reason", ""))
                    item.update(
                        reason=reason,
                        reason_label=feature_detail.REASON_LABELS.get(reason, reason),
                        missing_rate=_clean_float(row.get("Missing_Rate")),
                        related_variable=str(related) if related else None,
                        related_variable_name=related_item.get("source_name") if related_item else None,
                        correlation=_clean_float(row.get("Correlation")),
                        iteration=int(row["Iteration"]) if row.get("Iteration") is not None else None,
                        vif=_clean_float(row.get("VIF")),
                        vif_is_infinite=bool(row.get("VIF_Is_Infinite", False)),
                    )
                    excluded.append(item)
            reason_counts = Counter(item["reason"] for item in excluded)
            combinations[combo_id] = {
                "combination_id": combo_id,
                "label": combo["label"],
                "groups": groups,
                "status": "success" if audit else "running",
                "reason": None if audit else "의미 통합 변수 선택 기록이 아직 생성되지 않았습니다.",
                "stage": "validation",
                "fit_hash": fit_hash,
                "candidate_count": len(selected) + len(excluded),
                "selected_count": len(selected),
                "excluded_count": len(excluded),
                "selected": selected,
                "excluded": excluded,
                "exclusion_summary": [
                    {"reason": reason, "reason_label": feature_detail.REASON_LABELS.get(reason, reason), "count": count}
                    for reason, count in reason_counts.most_common()
                ],
                "source_label": "장치 의미 통합 기준 신규 학습",
            }

        payload = {
            "campaign": "semantic_control",
            "target": target_id,
            "selection_scope": "validation_train",
            "model_scope": "shared_all_models",
            "combinations": combinations,
        }
        _write_json_atomic(FEATURE_SELECTION_DIR / f"{target_id}.json", payload)


def build_index() -> dict:
    if not V1_INDEX.exists():
        raise FileNotFoundError(f"Completed baseline index is required: {V1_INDEX}")
    if not V2_OUTPUT.exists():
        raise FileNotFoundError(f"Semantic E6 output is required: {V2_OUTPUT}")

    v1 = _read_json(V1_INDEX)
    catalog = _read_json(CATALOG)
    index = _build_raw_v2_index()
    groups_by_id = _groups_by_id(catalog)
    runs = _validation_runs()
    scheduler = _read_json(V2_OUTPUT / "scheduler.json", {})
    completed_lanes = len(scheduler.get("Completed_Lanes", []))
    total_lanes = int(scheduler.get("Total_Lanes", len(base.TARGET_NAMES_KR) * len(base.MODELS)))
    finished = scheduler.get("Status") == "finished" or completed_lanes >= total_lanes

    for target_id, target in index["targets"].items():
        v1_target = v1["targets"][target_id]
        for combo_id, groups in groups_by_id.items():
            if _is_reused_scope(groups):
                for key, result in v1_target.get("results", {}).items():
                    if not key.startswith(f"{combo_id}::"):
                        continue
                    copied = copy.deepcopy(result)
                    copied["source"] = "reused_stored_control_codes"
                    copied["source_label"] = "시설별 제어코드 기준 결과 재사용"
                    target["results"][key] = copied
                target["groups"][combo_id]["campaign_role"] = "reused"
                continue

            if _is_semantic_scope(groups):
                target["groups"][combo_id]["campaign_role"] = "semantic_retrained"
                for model in base.MODELS:
                    key = f"{combo_id}::{model}::validation"
                    existing = target["results"].get(key)
                    if existing:
                        existing["source"] = "semantic_retrained"
                        existing["source_label"] = "장치 의미 통합 기준 신규 학습"
                        continue
                    seeded = runs.get((target_id, combo_id, model), {})
                    target["results"][key] = {
                        "group_id": combo_id,
                        "model": model,
                        "split": "validation",
                        "completed_seeds": len(seeded),
                        "status": "running" if seeded else "queued",
                        "reason": "현재 학습 중" if seeded else "해당 lane 실행 대기",
                        "params": {},
                        "bounded": {},
                        "raw": {},
                        "seeds": {str(seed): {"run_id": run["Run_ID"], "seed": seed, "status": "success"} for seed, run in seeded.items()},
                        "source": "semantic_retrained",
                        "source_label": "장치 의미 통합 기준 신규 학습",
                    }
                continue

            target["groups"][combo_id]["campaign_role"] = "not_in_scope"
            for model in base.MODELS:
                key = f"{combo_id}::{model}::validation"
                target["results"][key] = {
                    "group_id": combo_id,
                    "model": model,
                    "split": "validation",
                    "completed_seeds": 0,
                    "status": "not_in_scope",
                    "reason": OUT_OF_SCOPE_REASON,
                    "params": {},
                    "bounded": {},
                    "raw": {},
                    "seeds": {},
                    "source": "not_in_scope",
                    "source_label": "이번 재실험 범위 외",
                }

        target["status"] = "completed" if finished else "running"
        target["frozen"] = {"overall": None, "winners": {}}
        if finished:
            for model in base.MODELS:
                winner = _winner(target, model)
                if winner:
                    target["frozen"]["winners"][model] = {"group": winner}

    semantic_scope = sum(_is_semantic_scope(groups) for groups in groups_by_id.values())
    reused_scope = sum(_is_reused_scope(groups) for groups in groups_by_id.values())
    published_success = sum(
        1 for target in index["targets"].values() for result in target["results"].values()
        if result.get("status") == "success"
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    progress = round(completed_lanes / total_lanes * 100, 1) if total_lanes else 0.0
    baseline_index_sha256 = hashlib.sha256(V1_INDEX.read_bytes()).hexdigest()
    index["campaign"] = {
        "id": "semantic_control",
        "internal_campaign_id": "exhaustive_e6_semantic_v2",
        "name": "장치 의미 통합 기준",
        "short_name": "장치 의미 통합",
        "description": "시설마다 다른 제어 장치의 방향·층수 표현을 합치고, 장치 종류와 측정값 의미는 유지한 추가 검증",
        "why_separate": "완료된 시설별 코드 기준 결과를 보존하면서, 시설 간 코드 표현 차이가 E6 연결률과 예측 성능에 미친 영향을 공정하게 비교하기 위해 별도 실험으로 진행했습니다.",
        "scope_note": f"{SEMANTIC_SCOPE_REASON}. E6가 없는 {reused_scope}개 조합은 기존 결과를 재사용합니다.",
        "status": "completed" if finished else "running",
        "has_real_results": published_success > 0,
        "partial_results": not finished,
        "snapshot_generated_at": generated_at,
        "snapshot_note": "완료된 3개 시드 결과만 지표로 반영하며, 나머지는 진행 상태로 유지합니다." if not finished else "의미 통합 추가실험 전체 완료 결과입니다.",
        "total_combinations": len(groups_by_id),
        "semantic_retrained_combinations": semantic_scope,
        "reused_combinations": reused_scope,
        "out_of_scope_combinations": len(groups_by_id) - semantic_scope - reused_scope,
        "total_targets": len(base.TARGET_NAMES_KR),
        "total_models": len(base.MODELS),
        "total_seeds": len(base.SEEDS),
        "seeds": base.SEEDS,
        "completed_lanes": completed_lanes,
        "total_lanes": total_lanes,
        "remaining_lanes": max(total_lanes - completed_lanes, 0),
        "progress_percent": progress,
        "published_success_groups": published_success,
        "predictions_path": "./cache/campaigns/semantic_control/predictions",
        "feature_selections_path": "./cache/campaigns/semantic_control/feature_selections",
        "baseline_predictions_path": "./cache/predictions",
        "baseline_index_sha256": baseline_index_sha256,
    }

    _write_json_atomic(INDEX_FILE, index)
    _build_feature_selection_cache(index, catalog, runs)
    build_campaign_catalog(v1, index)
    print(f"[Semantic dashboard] {completed_lanes}/{total_lanes} lanes ({progress:.1f}%), {published_success} published result groups")
    return index


def build_campaign_catalog(v1: dict, semantic: dict) -> dict:
    payload = {
        "default_campaign": "stored_control_codes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaigns": [
            {
                "id": "stored_control_codes",
                "internal_campaign_id": "exhaustive_e1_e7_v1",
                "name": "시설별 제어코드 기준",
                "short_name": "시설별 코드 기준",
                "description": "DB에 저장된 시설 제어 코드를 구분한 최초 전수실험",
                "status": v1.get("campaign", {}).get("status", "completed"),
                "progress_percent": v1.get("campaign", {}).get("progress_percent", 100.0),
                "data_path": "./cache/experiment_index.json",
                "predictions_path": "./cache/predictions",
                "feature_selections_path": "./cache/feature_selections",
            },
            {
                "id": "semantic_control",
                "internal_campaign_id": "exhaustive_e6_semantic_v2",
                "name": "장치 의미 통합 기준",
                "short_name": "장치 의미 통합",
                "description": semantic["campaign"]["description"],
                "why_separate": semantic["campaign"]["why_separate"],
                "scope_note": semantic["campaign"]["scope_note"],
                "status": semantic["campaign"]["status"],
                "progress_percent": semantic["campaign"]["progress_percent"],
                "completed_lanes": semantic["campaign"]["completed_lanes"],
                "total_lanes": semantic["campaign"]["total_lanes"],
                "remaining_lanes": semantic["campaign"]["remaining_lanes"],
                "data_path": "./cache/campaigns/semantic_control/experiment_index.json",
                "predictions_path": "./cache/campaigns/semantic_control/predictions",
                "baseline_predictions_path": "./cache/predictions",
                "feature_selections_path": "./cache/campaigns/semantic_control/feature_selections",
            },
        ],
    }
    _write_json_atomic(CAMPAIGN_CATALOG, payload)
    return payload


if __name__ == "__main__":
    build_index()
