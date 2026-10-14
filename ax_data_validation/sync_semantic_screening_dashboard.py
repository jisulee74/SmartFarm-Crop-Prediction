#!/usr/bin/env python3
"""Publish the 127-combination single-seed semantic screening as a provisional view."""

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
    / "exhaustive_e6_semantic_screening_v1"
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
VALIDATION_SEEDS = [42]


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
    accepted = set()
    for selection_path in OUTPUT.glob("selections/*/*.json"):
        selection = read_json(selection_path, {})
        if selection.get("Status") == "success":
            accepted.update(selection.get("test_runs", []))
    for path in OUTPUT.glob("runs/*/result.json"):
        run = read_json(path)
        if not run or run.get("Status") != "success" or run.get("Run_ID") not in accepted:
            continue
        stage = str(run.get("Stage"))
        if stage != "test":
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


def summary_validation_runs(summary: dict):
    seeded = {}
    for item in summary.get("Runs") or []:
        if item.get("Status") != "success":
            continue
        seed = int(item["Seed"])
        seeded[seed] = {
            "Run_ID": None,
            "Seed": seed,
            "Params": summary.get("Params") or {},
            "Feature_Count": summary.get("Feature_Count", 0),
            "N_Eval": 0,
            "Metrics": {"raw": item.get("Raw") or {}, "bounded": item.get("Bounded") or {}},
        }
    return seeded


def result_entry(
    summary: dict,
    seeded: dict[int, dict],
    split: str,
    source: str,
    expected_seeds=None,
):
    expected_seeds = list(expected_seeds or SEEDS)
    runs = [seeded[seed] for seed in expected_seeds if seed in seeded]
    status = "success" if len(runs) == len(expected_seeds) else ("ineligible" if summary.get("Status") == "ineligible" else "failed")
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
            for seed in expected_seeds
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
    scheduler = read_json(OUTPUT / "screening_scheduler.json", {})
    if scheduler.get("Status") != "finished":
        raise RuntimeError("Semantic single-seed screening campaign has not finished")
    rankings = pd.read_csv(OUTPUT / "validation_rankings.csv")
    if len(rankings) != 127 * len(base.TARGET_NAMES_KR) * len(base.MODELS):
        raise ValueError(f"Final ranking is incomplete: {len(rankings)}/6096")
    index = read_json(INDEX_FILE)
    if not index:
        raise FileNotFoundError(f"Semantic dashboard index missing: {INDEX_FILE}")
    import pyarrow.parquet as pq
    validation_counts = {target: pq.ParquetFile(OUTPUT / "prepared" / target / "validation.parquet").metadata.num_rows for target in base.TARGET_NAMES_KR}
    groups = groups_by_id(catalog)
    if PREDICTIONS_DIR.exists():
        shutil.rmtree(PREDICTIONS_DIR)
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
            target["groups"][combo_id]["campaign_role"] = "semantic_single_seed_screening"
            for model in base.MODELS:
                summary_path = OUTPUT / "composite_summaries" / target_id / model / f"{combo_id}.json"
                summary = read_json(summary_path)
                if not summary:
                    raise FileNotFoundError(f"Composite summary missing: {summary_path}")
                seeded = summary_validation_runs(summary)
                entry = result_entry(
                    summary,
                    seeded,
                    "validation",
                    "semantic_single_seed_screening",
                    VALIDATION_SEEDS,
                )
                entry["source_label"] = "장치 의미 통합 1시드 잠정 선별"
                entry["provisional"] = True
                entry["predictions_available"] = False
                entry["prediction_note"] = "1시드 선별은 평가지표를 저장했습니다. 우승 조합의 Test에서 실제 관측값과 예측 시계열을 확인할 수 있습니다."
                entry["n_eval"] = validation_counts[target_id]
                if entry["status"] not in {"success", "failed", "ineligible"}:
                    raise RuntimeError(f"Non-terminal screening entry: {target_id}/{model}/{combo_id}")
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
            test_entry = result_entry(
                winner, seeded_test, "test", "semantic_screening_confirmed_test"
            )
            test_entry["predictions_available"] = True
            test_entry["source_label"] = "1시드 선별 우승 후보 3시드 Test 확인"
            if test_entry["status"] != "success":
                raise RuntimeError(f"Final Test predictions incomplete: {target_id}/{model}")
            target["results"][f"{combo_id}::{model}::test"] = test_entry
            for run in seeded_test.values():
                cache_prediction(run)
        target["status"] = "screening_complete"

    validation_statuses = Counter(
        result.get("status", "unknown")
        for target in index["targets"].values()
        for key, result in target["results"].items()
        if key.endswith("::validation")
    )
    index["campaign"] = {
        "id": "semantic_control",
        "internal_campaign_id": "exhaustive_e6_semantic_screening_v1",
        "name": "장치 의미 통합 기준",
        "short_name": "장치 의미 통합",
        "description": "방향·층수 표현은 통합하고 장치 종류와 측정값 의미는 유지한 127개 조합 1시드 잠정 선별 결과",
        "why_separate": "시설마다 다른 방향·층수 코드로 같은 장치 의미가 지나치게 분산되는 문제를 줄여 시설 간 공통 패턴을 학습하기 위해 적용했습니다.",
        "scope_note": "127개 조합 전체를 시드 42와 고정 하이퍼파라미터로 선별했습니다. 현재 표의 Validation 순위는 잠정 결과이며, 타깃·모델별 우승 후보의 Test 평가는 3개 시드로 수행했습니다. 병행 중인 조합별 3-fold CV·3시드 정식 실험이 끝나면 같은 화면이 최종 결과로 교체됩니다.",
        "status": "screening_complete",
        "has_real_results": True,
        "partial_results": True,
        "provisional": True,
        "snapshot_generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_note": "장치 의미 통합 기준 127개 조합의 1시드 잠정 Validation 순위와 우승 후보 3시드 Test 결과입니다.",
        "total_combinations": 127,
        "semantic_retrained_combinations": 127,
        "reused_combinations": 0,
        "out_of_scope_combinations": 0,
        "total_targets": len(base.TARGET_NAMES_KR),
        "total_models": len(base.MODELS),
        "total_seeds": len(VALIDATION_SEEDS),
        "seeds": VALIDATION_SEEDS,
        "test_seeds": SEEDS,
        "validation_method": "single_seed_fixed_configuration_screening",
        "replacement_plan": "per_combination_3fold_three_seed_campaign_on_completion",
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
    compact = OUTPUT / "feature_selection" / target_id / f"{combo_id}.json"
    item = read_json(compact)
    if item:
        return item
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
                "source_label": "장치 의미 통합 기준 1시드 잠정 Validation 변수 선택 기록",
            }
        payload["campaign"] = "semantic_control_single_seed_screening"
        write_json(path, payload)


def update_campaign_catalog():
    payload = {
        "default_campaign": "semantic_control",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaigns": [
            {
                "id": "semantic_control",
                "internal_campaign_id": "exhaustive_e6_semantic_screening_v1",
                "name": "장치 의미 통합 기준",
                "short_name": "장치 의미 통합",
                "description": "방향·층수 표현을 통합한 127개 조합 1시드 잠정 선별 결과",
                "status": "screening_complete",
                "progress_percent": 100.0,
                "completed_results": 6096,
                "total_results": 6096,
                "total_seeds": 1,
                "seeds": VALIDATION_SEEDS,
                "test_seeds": SEEDS,
                "provisional": True,
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
            status="provisional semantic-control single-seed validation best",
            reference_model=str(best.Model_Type),
            reference_groups=groups,
            reference_combination_id=str(best.Combination_ID),
            reference_is_provisional=True,
            reference_snapshot_label=generated,
            reference_campaign_progress={"processed": 6096, "total": 6096, "percent": 100.0},
            reference_method_label="의미 통합 127개 조합 1시드 잠정 Validation BEST",
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
        HERE / "semantic_screening_dashboard_audit.json",
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

