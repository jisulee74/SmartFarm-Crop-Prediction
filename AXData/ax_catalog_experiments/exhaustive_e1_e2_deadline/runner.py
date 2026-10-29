from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import digest, file_hash
from ..exhaustive.combinations import all_combinations
from ..exhaustive.io import atomic_json, frozen_json
from ..exhaustive.models import make_exhaustive_adapter
from ..stability.policy import constrain, score, upper_bound
from .config import (
    ACTIVE_SEEDS,
    CAMPAIGN_ID,
    CPU_MODELS,
    DEFAULT_OUTPUT,
    GPU_MODELS,
    MODELS,
    METHOD,
    SOURCE_ROOT,
    TARGETS,
    V1_ROOT,
    VERSION,
)
from .selection import CompactSelection, load_or_select


_DATA_CACHE: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]] = {}


def _prepared(target: str):
    if target not in _DATA_CACHE:
        directory = SOURCE_ROOT / "prepared" / target
        spec = json.loads((directory / "manifest.json").read_text())
        train = pd.read_parquet(directory / "train.parquet")
        validation = pd.read_parquet(directory / "validation.parquet")
        _DATA_CACHE[target] = spec, train, validation
    return _DATA_CACHE[target]


def _candidates(spec: dict, combination: dict):
    features = sorted({f for group in combination["Groups"] for f in spec["groups"].get(group, [])})
    categorical = [f for f in features if f in spec["categorical"]]
    numeric = [f for f in features if f not in categorical]
    return numeric, categorical


def _selection_path(root: Path, target: str, combination_id: str):
    return root / "feature_selection" / target / f"{combination_id}.json"


def _summary_path(root: Path, target: str, model: str, combination_id: str):
    return root / "summaries" / target / model / f"{combination_id}.json"


def _selection_task(args):
    root, target, combination = Path(args[0]), args[1], args[2]
    path = _selection_path(root, target, combination["Combination_ID"])
    if path.exists():
        return {"status": "cached", "target": target, "combination": combination["Combination_ID"]}
    spec, train, _ = _prepared(target)
    numeric, categorical = _candidates(spec, combination)
    result = load_or_select(root, target, combination["Combination_ID"], train, numeric, categorical)
    return {
        "status": "success",
        "target": target,
        "combination": combination["Combination_ID"],
        "selected": len(result.selected),
    }


def _load_selection(root: Path, target: str, combination_id: str):
    data = json.loads(_selection_path(root, target, combination_id).read_text())
    return CompactSelection(
        data["selected_numeric"],
        data["selected_categorical"],
        data["exclusions"],
        data["categorical_levels"],
        data["diagnostics"],
    )


def _fixed_configs():
    configs = {}
    for target in TARGETS:
        configs[target] = {}
        for model in MODELS:
            rows = []
            for path in (V1_ROOT / "summaries" / target / model).glob("*.json"):
                value = json.loads(path.read_text())
                if value.get("Status") == "success" and value.get("Validation_RMSE") is not None:
                    rows.append(value)
            if not rows:
                raise RuntimeError(f"No reusable v1 winner: {target}/{model}")
            winner = min(
                rows,
                key=lambda x: (
                    x["Validation_RMSE"],
                    x["Validation_MAE"],
                    x.get("Feature_Count", 10**9),
                    x["Combination_ID"],
                ),
            )
            configs[target][model] = {
                "params": winner["Params"],
                "fixed_epochs": winner.get("Best_Epoch"),
                "source_combination": winner["Combination_ID"],
                "source_validation_rmse": winner["Validation_RMSE"],
            }
    return configs


def _fit_task(args):
    root, target, model, combination, config = Path(args[0]), args[1], args[2], args[3], args[4]
    combination_id = combination["Combination_ID"]
    output = _summary_path(root, target, model, combination_id)
    if output.exists():
        return {"status": "cached", "target": target, "model": model, "combination": combination_id}
    spec, train, validation = _prepared(target)
    selection = _load_selection(root, target, combination_id)
    if not selection.selected:
        result = {
            "Campaign_ID": CAMPAIGN_ID,
            "Target_ID": target,
            "Combination_ID": combination_id,
            "Variable_Groups": combination["Groups"],
            "Model_Type": model,
            "Status": "ineligible",
            "Reason": "all_candidate_features_excluded",
            "Method": METHOD,
        }
        atomic_json(output, result)
        return {"status": "ineligible", "target": target, "model": model, "combination": combination_id}

    params = dict(config["params"])
    batch_override = os.getenv(f"AX_DEADLINE_BATCH_SIZE_{model.upper()}")
    if batch_override:
        params["batch_size"] = int(batch_override)
    if model == "random_forest":
        params["n_jobs"] = 1
    if model == "catboost":
        params["thread_count"] = 1
        # CatBoost otherwise reserves most of the B200 for every process.  A
        # bounded fraction allows four independent small tabular fits to share
        # the device without terminating the worker pool.
        params["gpu_ram_part"] = float(os.getenv("AX_CATBOOST_GPU_RAM_PART", "0.10"))
    run_rows = []
    started = time.monotonic()
    fixed_epochs = config.get("fixed_epochs")
    cap_value = os.getenv(f"AX_DEADLINE_EPOCH_CAP_{model.upper()}")
    if cap_value and fixed_epochs is not None:
        fixed_epochs = min(int(fixed_epochs), int(cap_value))
    for seed in ACTIVE_SEEDS:
        seed_start = time.monotonic()
        try:
            adapter = make_exhaustive_adapter(
                model,
                selection,
                params,
                seed,
                spec["entity"],
                checkpoint_dir=None,
            )
            adapter.fit(
                train,
                "target",
                validation=None,
                fixed_epochs=fixed_epochs,
            )
            raw = adapter.predict(validation)
            bounded, unstable = constrain(raw, upper_bound(train, spec["components"]))
            if unstable:
                raise FloatingPointError("Raw prediction exceeded stability multiplier")
            run_rows.append({
                "Seed": seed,
                "Status": "success",
                "Duration_Seconds": time.monotonic() - seed_start,
                "Raw": score(validation.target, raw),
                "Bounded": score(validation.target, bounded),
            })
        except Exception as exc:
            run_rows.append({
                "Seed": seed,
                "Status": "failed",
                "Duration_Seconds": time.monotonic() - seed_start,
                "Error": f"{type(exc).__name__}: {exc}",
            })
    successful = [row for row in run_rows if row["Status"] == "success"]
    result = {
        "Campaign_ID": CAMPAIGN_ID,
        "Target_ID": target,
        "Combination_ID": combination_id,
        "Variable_Groups": combination["Groups"],
        "Model_Type": model,
        "Status": "success" if len(successful) == len(ACTIVE_SEEDS) else "failed",
        "Method": METHOD,
        "Comparability": "within_same_campaign_only",
        "Params": params,
        "Source_Params": config["params"],
        "Fixed_Epochs": fixed_epochs,
        "Source_Fixed_Epochs": config.get("fixed_epochs"),
        "Config_Source_Combination": config["source_combination"],
        "Seed_Count": len(successful),
        "Feature_Count": len(selection.selected),
        "Selected_Features": selection.selected,
        "Runs": run_rows,
        "Duration_Seconds": time.monotonic() - started,
    }
    if successful:
        for metric in ("rmse", "mae", "r2", "ccc"):
            values = [float(row["Bounded"][metric]) for row in successful]
            result[f"Validation_{metric.upper()}_Mean"] = float(np.mean(values))
            result[f"Validation_{metric.upper()}_Std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if len(successful) != len(ACTIVE_SEEDS):
        result["Reason"] = "incomplete_validation_seeds"
    atomic_json(output, result)
    return {"status": result["Status"], "target": target, "model": model, "combination": combination_id}


def _write_progress(root: Path, phase: str, complete: int, total: int, statuses=None):
    path = root / ("progress.json" if phase == "finished" else f"progress_{phase}.json")
    atomic_json(path, {
        "Campaign_ID": CAMPAIGN_ID,
        "Phase": phase,
        "Completed": complete,
        "Total": total,
        "Percent": round(complete / total * 100, 2) if total else 100.0,
        "Statuses": statuses or {},
        "Updated_At": time.time(),
    })


def initialize(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    configs = _fixed_configs()
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "version": VERSION,
        "method": METHOD,
        "purpose": "Complete all 127 combinations on the registered 8 targets and 6 models within the deadline.",
        "not_equivalent_to": "per-combination 3-fold hyperparameter tuning campaign",
        "combination_count": 127,
        "target_count": len(TARGETS),
        "model_count": len(MODELS),
        "seeds": list(ACTIVE_SEEDS),
        "source_prepared_manifest_sha256": file_hash(SOURCE_ROOT / "prepared_manifest.json"),
        "v1_manifest_sha256": file_hash(V1_ROOT / "campaign_manifest.json"),
        "fixed_configs": configs,
    }
    epoch_caps = {
        model: int(os.environ[f"AX_DEADLINE_EPOCH_CAP_{model.upper()}"])
        for model in MODELS
        if os.getenv(f"AX_DEADLINE_EPOCH_CAP_{model.upper()}")
    }
    if epoch_caps:
        manifest["epoch_caps"] = epoch_caps
    parameter_overrides = {}
    for model in MODELS:
        batch_size = os.getenv(f"AX_DEADLINE_BATCH_SIZE_{model.upper()}")
        if batch_size:
            parameter_overrides.setdefault(model, {})["batch_size"] = int(batch_size)
    if parameter_overrides:
        manifest["parameter_overrides"] = parameter_overrides
    frozen_json(root / "campaign_manifest.json", manifest)
    pd.DataFrame([{
        "Combination_ID": row["Combination_ID"],
        "Variable_Groups": "|".join(row["Groups"]),
        "Group_Count": row["Group_Count"],
        "Label": row["Label"],
    } for row in all_combinations()]).to_csv(root / "combination_catalog.csv", index=False, encoding="utf-8-sig")
    return manifest


def _run_pool(tasks, fn, workers, root, phase):
    statuses, completed = {}, 0
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(fn, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                value = future.result()
                state = value.get("status", "unknown")
            except Exception as exc:
                state = "worker_error"
                value = {"error": f"{type(exc).__name__}: {exc}"}
            statuses[state] = statuses.get(state, 0) + 1
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                _write_progress(root, phase, completed, len(tasks), statuses)
    return statuses


def run(root: Path = DEFAULT_OUTPUT):
    root = Path(root)
    manifest = initialize(root)
    combinations = all_combinations()
    selection_tasks = [(str(root), target, combo) for target in TARGETS for combo in combinations]
    selection_workers = int(os.getenv("AX_DEADLINE_SELECTION_WORKERS", "12"))
    _run_pool(selection_tasks, _selection_task, selection_workers, root, "feature_selection")

    configs = manifest["fixed_configs"]
    cpu_tasks = [(str(root), target, model, combo, configs[target][model]) for target in TARGETS for model in CPU_MODELS for combo in combinations]
    gpu_tasks = [(str(root), target, model, combo, configs[target][model]) for target in TARGETS for model in GPU_MODELS for combo in combinations]
    cpu_workers = int(os.getenv("AX_DEADLINE_CPU_WORKERS", "16"))
    gpu_workers = int(os.getenv("AX_DEADLINE_GPU_WORKERS", "16"))
    _write_progress(root, "model_screening", 0, len(cpu_tasks) + len(gpu_tasks), {})
    status = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as coordinator:
        jobs = [
            coordinator.submit(_run_pool, cpu_tasks, _fit_task, cpu_workers, root, "cpu_model_screening"),
            coordinator.submit(_run_pool, gpu_tasks, _fit_task, gpu_workers, root, "gpu_model_screening"),
        ]
        for job in jobs:
            for key, value in job.result().items():
                status[key] = status.get(key, 0) + value
    summaries = list((root / "summaries").glob("*/*/*.json"))
    _write_progress(root, "finished", len(summaries), 127 * len(TARGETS) * len(MODELS), status)
    return status


def run_model_phase(root: Path = DEFAULT_OUTPUT, phase: str = "gpu"):
    """Resume only one model queue after a worker-pool interruption."""
    root = Path(root)
    manifest = initialize(root)
    combinations = all_combinations()
    models = GPU_MODELS if phase == "gpu" else CPU_MODELS
    requested = [
        value.strip()
        for value in os.getenv("AX_DEADLINE_MODELS", "").split(",")
        if value.strip()
    ]
    if requested:
        unknown = sorted(set(requested) - set(models))
        if unknown:
            raise ValueError(
                f"Models are not part of the {phase} queue: {', '.join(unknown)}"
            )
        models = tuple(model for model in models if model in requested)
    if not models:
        raise ValueError(f"No models selected for the {phase} queue")
    targets = TARGETS
    requested_targets = [
        value.strip()
        for value in os.getenv("AX_DEADLINE_TARGETS", "").split(",")
        if value.strip()
    ]
    if requested_targets:
        unknown_targets = sorted(set(requested_targets) - set(TARGETS))
        if unknown_targets:
            raise ValueError(
                f"Unknown targets: {', '.join(unknown_targets)}"
            )
        targets = tuple(target for target in TARGETS if target in requested_targets)
    if not targets:
        raise ValueError("No targets selected for the model queue")
    tasks = [
        (str(root), target, model, combo, manifest["fixed_configs"][target][model])
        for target in targets
        for model in models
        for combo in combinations
    ]
    workers = int(os.getenv(
        "AX_DEADLINE_GPU_WORKERS" if phase == "gpu" else "AX_DEADLINE_CPU_WORKERS",
        "4" if phase == "gpu" else "16",
    ))
    model_label = "_".join(models)
    target_label = "_".join(targets) if requested_targets else "all_targets"
    return _run_pool(
        tasks,
        _fit_task,
        workers,
        root,
        f"{phase}_{model_label}_{target_label}_model_screening",
    )


def status(root: Path = DEFAULT_OUTPUT):
    root = Path(root)
    summaries = list((root / "summaries").glob("*/*/*.json"))
    values = []
    for path in summaries:
        try:
            values.append(json.loads(path.read_text()))
        except Exception:
            pass
    states = {}
    for value in values:
        state = value.get("Status", "unknown")
        states[state] = states.get(state, 0) + 1
    total = 127 * len(TARGETS) * len(MODELS)
    return {
        "Campaign_ID": CAMPAIGN_ID,
        "Completed": len(values),
        "Total": total,
        "Percent": round(len(values) / total * 100, 2),
        "States": states,
        "Progress": json.loads((root / "progress.json").read_text()) if (root / "progress.json").exists() else None,
    }


def finalize(root: Path = DEFAULT_OUTPUT, require_complete: bool = True):
    """Build compact review files after every target/model/combination has a result."""
    root = Path(root)
    paths = sorted((root / "summaries").glob("*/*/*.json"))
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    campaign_method = manifest["method"]
    values = []
    for path in paths:
        value = json.loads(path.read_text())
        if value.get("Method") != campaign_method:
            value["Method"] = campaign_method
            atomic_json(path, value)
        values.append(value)
    expected = 127 * len(TARGETS) * len(MODELS)
    if require_complete and len(values) != expected:
        raise RuntimeError(f"Cannot finalize an incomplete campaign: {len(values)}/{expected}")

    rows = []
    for value in values:
        rows.append({
            "Campaign_ID": value.get("Campaign_ID"),
            "Target_ID": value.get("Target_ID"),
            "Model_Type": value.get("Model_Type"),
            "Combination_ID": value.get("Combination_ID"),
            "Variable_Groups": "|".join(value.get("Variable_Groups", [])),
            "Status": value.get("Status"),
            "Reason": value.get("Reason"),
            "Seed_Count": value.get("Seed_Count", 0),
            "Feature_Count": value.get("Feature_Count", 0),
            "Validation_RMSE_Mean": value.get("Validation_RMSE_Mean"),
            "Validation_RMSE_Std": value.get("Validation_RMSE_Std"),
            "Validation_MAE_Mean": value.get("Validation_MAE_Mean"),
            "Validation_MAE_Std": value.get("Validation_MAE_Std"),
            "Validation_R2_Mean": value.get("Validation_R2_Mean"),
            "Validation_R2_Std": value.get("Validation_R2_Std"),
            "Validation_CCC_Mean": value.get("Validation_CCC_Mean"),
            "Validation_CCC_Std": value.get("Validation_CCC_Std"),
            "Duration_Seconds": value.get("Duration_Seconds"),
            "Method": value.get("Method"),
            "Comparability": value.get("Comparability"),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "metrics.csv", index=False, encoding="utf-8-sig")

    successful = metrics.loc[metrics.Status.eq("success")].copy()
    successful = successful.sort_values(
        [
            "Target_ID",
            "Model_Type",
            "Validation_RMSE_Mean",
            "Validation_MAE_Mean",
            "Feature_Count",
            "Combination_ID",
        ],
        kind="stable",
    )
    successful["Validation_Rank"] = (
        successful.groupby(["Target_ID", "Model_Type"]).cumcount() + 1
    )
    successful.to_csv(root / "rankings.csv", index=False, encoding="utf-8-sig")

    status_rows = [
        {"Status": key, "Count": value}
        for key, value in sorted(Counter(metrics.Status.fillna("unknown")).items())
    ]
    pd.DataFrame(status_rows).to_csv(
        root / "status_summary.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "Campaign_ID": CAMPAIGN_ID,
        "Method": METHOD,
        "Completed": len(values),
        "Total": expected,
        "Percent": round(len(values) / expected * 100, 2),
        "Statuses": dict(Counter(metrics.Status.fillna("unknown"))),
        "Generated_At": time.time(),
        "Files": ["metrics.csv", "rankings.csv", "status_summary.csv"],
    }
    atomic_json(root / "completion_report.json", report)
    _write_progress(root, "finished", len(values), expected, report["Statuses"])
    lines = [
        f"# {CAMPAIGN_ID} 완료 보고",
        "",
        f"- 완료: {len(values):,}/{expected:,} ({report['Percent']:.2f}%)",
        f"- 방법: {report['Method']}",
        "- 비교 범위: 이 긴급 고정 설정 캠페인 내부",
        "",
        "## 상태",
        "",
    ]
    lines.extend(f"- {row['Status']}: {row['Count']:,}" for row in status_rows)
    (root / "Progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
