#!/usr/bin/env python3
"""Publish the completed 127-combination semantic-E6 campaign as the sole view."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import archive_granular_code_results
import sync_e6_semantic_dashboard as semantic_helpers
import sync_exhaustive_experiments as base
import sync_feature_selection_details as feature_detail


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
OUTPUT = (
    WORKSPACE
    / "ax_catalog_experiments"
    / "outputs"
    / "exhaustive_e6_semantic_completion_v2"
)
CACHE = HERE / "cache"
CATALOG_FILE = CACHE / "combination_catalog.json"
CAMPAIGN_ROOT = CACHE / "campaigns" / "semantic_control"
INDEX_FILE = CAMPAIGN_ROOT / "experiment_index.json"
PREDICTIONS_DIR = CAMPAIGN_ROOT / "predictions"
FEATURE_SELECTION_DIR = CAMPAIGN_ROOT / "feature_selections"
CAMPAIGN_CATALOG = CACHE / "campaign_catalog.json"
KPI_BUNDLE = HERE / "ax_direction" / "dashboard_bundle.json"
SEEDS = [42, 52, 62]


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: dict, pretty: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + ("\n" if pretty else ""),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def groups_by_id(catalog: dict) -> dict[str, list[str]]:
    return {
        str(item["combination_id"]): list(item["groups"])
        for item in catalog["combinations"]
    }


def is_completion_scope(groups: list[str]) -> bool:
    group_set = set(groups)
    return "E6" in group_set and bool(group_set & {"E1", "E2"})


def load_runs():
    values = defaultdict(dict)
    for path in OUTPUT.glob("runs/*/result.json"):
        run = read_json(path)
        if not run or run.get("Status") != "success":
            continue
        stage = str(run.get("Stage"))
        if stage not in {"validation", "test"}:
            continue
        try:
            seed = int(run["Seed"])
        except (KeyError, TypeError, ValueError):
            continue
        if seed not in SEEDS:
            continue
        key = (
            str(run.get("Target_ID")),
            str(run.get("Combination_ID")),
            str(run.get("Model_Type")),
            stage,
        )
        values[key][seed] = run
    return values


def metric_stats(runs: list[dict], variant: str):
    output = {}
    for metric in ("rmse", "mae", "r2", "ccc"):
        values = [
            finite(run.get("Metrics", {}).get(variant, {}).get(metric)) for run in runs
        ]
        values = [value for value in values if value is not None]
        if values:
            output[f"{metric}_mean"] = statistics.fmean(values)
            output[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
    return output


def result_entry(summary: dict, seeded: dict[int, dict], split: str, source: str):
    runs = [seeded[seed] for seed in SEEDS if seed in seeded]
    status = "success" if len(runs) == len(SEEDS) else str(summary.get("Status", "failed"))
    return {
        "group_id": str(summary["Combination_ID"]),
        "model": str(summary["Model_Type"]),
        "split": split,
        "completed_seeds": len(runs),
        "status": status,
        "reason": None if status == "success" else summary.get("Reason", "incomplete_seeds"),
        "params": summary.get("Params") or (runs[0].get("Params") if runs else {}),
        "best_epoch": summary.get("Best_Epoch") or summary.get("Fixed_Epochs"),
        "n_eval": int(round(statistics.fmean([int(run.get("N_Eval") or 0) for run in runs]))) if runs else 0,
        "feature_count": summary.get("Feature_Count") or (runs[0].get("Feature_Count") if runs else 0),
        "bounded": metric_stats(runs, "bounded"),
        "raw": metric_stats(runs, "raw"),
        "seeds": {
            str(seed): {
                "run_id": seeded[seed]["Run_ID"],
                "seed": seed,
                "status": "success",
            }
            for seed in SEEDS
            if seed in seeded
        },
        "source": source,
        "source_label": "장치 의미 통합 기준 최종 실험",
    }


def cache_prediction(run: dict):
    source = OUTPUT / "runs" / run["Run_ID"] / "predictions.parquet"
    if not source.exists():
        raise FileNotFoundError(f"Prediction parquet missing: {source}")
    frame = pd.read_parquet(source)
    payload = {
        "run_id": run["Run_ID"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": json.loads(frame.to_json(orient="records", date_format="iso")),
    }
    write_json(PREDICTIONS_DIR / f"{run['Run_ID']}.json", payload)


def update_index(catalog: dict, runs: dict):
    scheduler = read_json(OUTPUT / "scheduler.json", {})
    if scheduler.get("Status") != "finished":
        raise RuntimeError("Semantic completion campaign has not finished")
    rankings = pd.read_csv(OUTPUT / "validation_rankings.csv")
    if len(rankings) != 127 * len(base.TARGET_NAMES_KR) * len(base.MODELS):
        raise ValueError(f"Final ranking is incomplete: {len(rankings)}/6096")
    index = read_json(INDEX_FILE)
    if not index:
        raise FileNotFoundError(f"Semantic dashboard index missing: {INDEX_FILE}")
    if index.get("campaign", {}).get("internal_campaign_id") == "exhaustive_e6_semantic_screening_v1":
        baseline = WORKSPACE / "ax_catalog_experiments/outputs/dashboard_before_screening_20261014/semantic_control"
        original = read_json(baseline / "experiment_index.json")
        if not original:
            raise RuntimeError("Preserved formal predecessor index is required; screening results must not be reused as formal results")
        index = original
        for folder in ("predictions", "feature_selections"):
            shutil.copytree(baseline / folder, CAMPAIGN_ROOT / folder, dirs_exist_ok=True)
    groups = groups_by_id(catalog)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for target_id, target in index["targets"].items():
        # Remove the previous campaign's stale Test winners before publishing
        # the newly selected 127-combination winners.
        target["results"] = {
            key: value
            for key, value in target.get("results", {}).items()
            if not key.endswith("::test")
        }
        for combo_id, combo_groups in groups.items():
            target["groups"][combo_id]["campaign_role"] = (
                "semantic_retrained" if "E6" in combo_groups else "semantic_common_no_e6"
            )
            if "E6" not in combo_groups:
                for model in base.MODELS:
                    key = f"{combo_id}::{model}::validation"
                    existing = target["results"].get(key)
                    if existing:
                        existing["source"] = "semantic_common_no_e6"
                        existing["source_label"] = "E6 미포함 공통 결과"
                        existing["prediction_source"] = "baseline"
                continue
            if not is_completion_scope(combo_groups):
                continue
            for model in base.MODELS:
                summary_path = OUTPUT / "composite_summaries" / target_id / model / f"{combo_id}.json"
                summary = read_json(summary_path)
                if not summary:
                    raise FileNotFoundError(f"Composite summary missing: {summary_path}")
                seeded = runs.get((target_id, combo_id, model, "validation"), {})
                entry = result_entry(summary, seeded, "validation", "semantic_completion")
                if entry["status"] != "success":
                    raise RuntimeError(f"Incomplete final validation entry: {target_id}/{model}/{combo_id}")
                target["results"][f"{combo_id}::{model}::validation"] = entry

        target["frozen"] = {"overall": None, "winners": {}}
        for model in base.MODELS:
            winner = read_json(OUTPUT / "validation_winners" / target_id / f"{model}.json")
            selection = read_json(OUTPUT / "selections" / target_id / f"{model}.json")
            if not winner or not selection or selection.get("Status") != "success":
                raise RuntimeError(f"Final winner/Test record missing: {target_id}/{model}")
            combo_id = str(winner["Combination_ID"])
            target["frozen"]["winners"][model] = {"group": combo_id}
            seeded_test = runs.get((target_id, combo_id, model, "test"), {})
            test_entry = result_entry(winner, seeded_test, "test", "semantic_final_test")
            if test_entry["status"] != "success":
                raise RuntimeError(f"Final Test predictions incomplete: {target_id}/{model}")
            target["results"][f"{combo_id}::{model}::test"] = test_entry
            for run in seeded_test.values():
                cache_prediction(run)
            seeded_validation = runs.get((target_id, combo_id, model, "validation"), {})
            for run in seeded_validation.values():
                cache_prediction(run)
        target["status"] = "completed"

    validation_statuses = Counter(
        result.get("status", "unknown")
        for target in index["targets"].values()
        for key, result in target["results"].items()
        if key.endswith("::validation")
    )
    index["campaign"] = {
        "id": "semantic_control",
        "internal_campaign_id": "exhaustive_e6_semantic_v2+exhaustive_e6_semantic_completion_v2",
        "name": "장치 의미 통합 기준",
        "short_name": "장치 의미 통합",
        "description": "방향·층수 표현은 통합하고 장치 종류와 측정값 의미는 유지한 127개 조합 최종 결과",
        "why_separate": "시설마다 다른 방향·층수 코드로 같은 장치 의미가 지나치게 분산되는 문제를 줄여 시설 간 공통 패턴을 학습하기 위해 적용했습니다.",
        "scope_note": "E6 포함 64개 조합은 장치 의미 통합 피처를 사용하고, E6 미포함 63개 조합은 입력이 동일해 기존 완료 결과를 재사용했습니다. 127개 전체 Validation 순위를 확정한 뒤 타깃·모델별 48개 우승 조합만 Test 평가했습니다.",
        "status": "completed",
        "has_real_results": True,
        "partial_results": False,
        "snapshot_generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_note": "장치 의미 통합 기준 127개 조합의 최종 Validation 순위와 우승 조합 Test 결과입니다.",
        "total_combinations": 127,
        "semantic_retrained_combinations": 64,
        "reused_combinations": 63,
        "out_of_scope_combinations": 0,
        "total_targets": len(base.TARGET_NAMES_KR),
        "total_models": len(base.MODELS),
        "total_seeds": len(SEEDS),
        "seeds": SEEDS,
        "processed_combination_summaries": 6096,
        "expected_combination_summaries": 6096,
        "progress_percent": 100.0,
        "summary_status_counts": dict(sorted(validation_statuses.items())),
        "published_success_groups": validation_statuses.get("success", 0),
        "final_test_winners": 48,
        "final_test_seed_runs": 144,
        "predictions_path": "./cache/campaigns/semantic_control/predictions",
        "feature_selections_path": "./cache/campaigns/semantic_control/feature_selections",
    }
    write_json(INDEX_FILE, index, pretty=True)
    return index


def selection_audit(target_id: str, combo_id: str):
    directory = OUTPUT / "feature_selection" / target_id / combo_id
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        item = read_json(path)
        if item and item.get("identity", {}).get("stage") == "validation":
            return item
    return None


def update_feature_details(catalog: dict):
    semantic_lookup = semantic_helpers._semantic_feature_lookup()
    source_labels = feature_detail.load_source_labels()
    prepared = read_json(OUTPUT / "prepared_manifest.json", {"targets": {}})
    groups = groups_by_id(catalog)
    for target_id in base.TARGET_NAMES_KR:
        path = FEATURE_SELECTION_DIR / f"{target_id}.json"
        payload = read_json(path, {"combinations": {}})
        target_groups = prepared["targets"][target_id]["groups"]
        feature_group = {
            variable: group
            for group, variables in target_groups.items()
            for variable in variables
        }
        for combo in catalog["combinations"]:
            combo_id = str(combo["combination_id"])
            if not is_completion_scope(groups[combo_id]):
                existing = payload.get("combinations", {}).get(combo_id)
                if existing and "E6" not in groups[combo_id]:
                    existing["source_label"] = "E6 미포함 공통 변수 선택 기록"
                continue
            audit = selection_audit(target_id, combo_id)
            if not audit:
                raise RuntimeError(f"Validation feature selection missing: {target_id}/{combo_id}")
            numeric = set(audit.get("selected_numeric") or [])
            categorical = set(audit.get("selected_categorical") or [])
            selected = []
            for variable_id in sorted(numeric | categorical):
                if variable_id == "item_code":
                    continue
                group = "E6" if variable_id.startswith("E6S__") else feature_group.get(variable_id, "-")
                selected.append(
                    semantic_helpers._selected_item(
                        variable_id,
                        group,
                        "categorical" if variable_id in categorical else "numeric",
                        semantic_lookup,
                        source_labels,
                    )
                )
            excluded = []
            for row in audit.get("exclusions") or []:
                variable_id = str(row.get("Variable_ID", ""))
                if not variable_id or variable_id == "item_code":
                    continue
                group = "E6" if variable_id.startswith("E6S__") else feature_group.get(variable_id, "-")
                item = semantic_helpers._selected_item(
                    variable_id, group, "numeric", semantic_lookup, source_labels
                )
                related = row.get("Related_Variable")
                related_item = None
                if related:
                    related_text = str(related)
                    related_item = semantic_helpers._selected_item(
                        related_text,
                        "E6" if related_text.startswith("E6S__") else feature_group.get(related_text, "-"),
                        "numeric",
                        semantic_lookup,
                        source_labels,
                    )
                reason = str(row.get("Reason", ""))
                item.update(
                    reason=reason,
                    reason_label=feature_detail.REASON_LABELS.get(reason, reason),
                    missing_rate=finite(row.get("Missing_Rate")),
                    related_variable=str(related) if related else None,
                    related_variable_name=related_item.get("source_name") if related_item else None,
                    correlation=finite(row.get("Correlation")),
                    iteration=int(row["Iteration"]) if row.get("Iteration") is not None else None,
                    vif=finite(row.get("VIF")),
                    vif_is_infinite=bool(row.get("VIF_Is_Infinite", False)),
                )
                excluded.append(item)
            counts = Counter(item["reason"] for item in excluded)
            payload.setdefault("combinations", {})[combo_id] = {
                "combination_id": combo_id,
                "label": combo["label"],
                "groups": groups[combo_id],
                "status": "success",
                "reason": None,
                "stage": "validation",
                "fit_hash": audit.get("identity", {}).get("fit_hash"),
                "candidate_count": len(selected) + len(excluded),
                "selected_count": len(selected),
                "excluded_count": len(excluded),
                "selected": selected,
                "excluded": excluded,
                "exclusion_summary": [
                    {
                        "reason": reason,
                        "reason_label": feature_detail.REASON_LABELS.get(reason, reason),
                        "count": count,
                    }
                    for reason, count in counts.most_common()
                ],
                "source_label": "장치 의미 통합 기준 최종 Validation 변수 선택 기록",
            }
        payload["campaign"] = "semantic_control_final"
        write_json(path, payload)


def update_campaign_catalog():
    payload = {
        "default_campaign": "semantic_control",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaigns": [
            {
                "id": "semantic_control",
                "internal_campaign_id": "exhaustive_e6_semantic_v2+exhaustive_e6_semantic_completion_v2",
                "name": "장치 의미 통합 기준",
                "short_name": "장치 의미 통합",
                "description": "방향·층수 표현을 통합하고 장치 종류와 측정 의미를 유지한 127개 조합 최종 결과",
                "status": "completed",
                "progress_percent": 100.0,
                "completed_results": 6096,
                "total_results": 6096,
                "total_seeds": 3,
                "seeds": SEEDS,
                "data_path": "./cache/campaigns/semantic_control/experiment_index.json",
                "predictions_path": "./cache/campaigns/semantic_control/predictions",
                "feature_selections_path": "./cache/campaigns/semantic_control/feature_selections",
            }
        ],
    }
    write_json(CAMPAIGN_CATALOG, payload, pretty=True)


def update_kpi():
    bundle = read_json(KPI_BUNDLE)
    rankings = pd.read_csv(OUTPUT / "validation_rankings.csv")
    successful = rankings.loc[
        rankings.Status.eq("success") & rankings.Validation_RMSE.notna()
    ].copy()
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    for row in bundle["tables"]["kpi_planning_reference"]:
        candidates = successful.loc[successful.Target_ID.eq(row["target"])].sort_values(
            ["Validation_RMSE", "Validation_MAE", "Feature_Count", "Combination_ID"],
            kind="stable",
        )
        if candidates.empty:
            raise RuntimeError(f"No final KPI reference: {row['target']}")
        best = candidates.iloc[0]
        groups = str(best.Variable_Groups).split("|")
        row.update(
            reference_rmse=float(best.Validation_RMSE),
            illustrative_10pct_rmse=float(best.Validation_RMSE) * 0.9,
            status="completed semantic-control 127-combination validation best",
            reference_model=str(best.Model_Type),
            reference_groups=groups,
            reference_combination_id=str(best.Combination_ID),
            reference_is_provisional=False,
            reference_snapshot_label=generated,
            reference_campaign_progress={"processed": 6096, "total": 6096, "percent": 100.0},
            reference_method_label="의미 통합 127개 조합 Validation BEST",
        )
    write_json(KPI_BUNDLE, bundle)


def verify(index: dict):
    groups = 0
    validation_terminal = 0
    validation_statuses = Counter()
    tests = 0
    for target in index["targets"].values():
        if len(target.get("groups", {})) != 127:
            raise AssertionError("A target does not contain 127 combinations")
        groups += len(target["groups"])
        for key, value in target.get("results", {}).items():
            if key.endswith("::validation") and value.get("status") == "success":
                validation_terminal += 1
                validation_statuses["success"] += 1
            elif key.endswith("::validation") and value.get("status") in {"failed", "ineligible"}:
                validation_terminal += 1
                validation_statuses[value.get("status")] += 1
            if key.endswith("::test") and value.get("status") == "success":
                tests += 1
    if groups != 8 * 127 or validation_terminal != 6096 or tests != 48:
        raise AssertionError(
            f"Dashboard integrity failed: groups={groups}, validation={validation_terminal}, tests={tests}"
        )
    return {
        "groups": groups,
        "validation_terminal": validation_terminal,
        "validation_statuses": dict(validation_statuses),
        "tests": tests,
    }


def main():
    catalog = read_json(CATALOG_FILE)
    runs = load_runs()
    index = update_index(catalog, runs)
    update_feature_details(catalog)
    update_campaign_catalog()
    update_kpi()
    shutil.copy2(OUTPUT / "validation_rankings.csv", CAMPAIGN_ROOT / "validation_rankings.csv")
    shutil.copy2(OUTPUT / "validation_winners.csv", CAMPAIGN_ROOT / "validation_winners.csv")
    shutil.copy2(OUTPUT / "test_results.csv", CAMPAIGN_ROOT / "test_results.csv")
    archive = archive_granular_code_results.build()
    audit = verify(index)
    write_json(
        HERE / "semantic_final_dashboard_audit.json",
        {
            **audit,
            "campaign": index["campaign"]["internal_campaign_id"],
            "archive": str(archive.relative_to(HERE)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        pretty=True,
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()

