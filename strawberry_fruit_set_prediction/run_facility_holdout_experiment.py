#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
AXDATA = ROOT.parent
SOURCE_ROOT = ROOT
sys.path[:0] = [str(AXDATA), str(SOURCE_ROOT)]

from common_regression.contracts import SplitBundle
from common_regression.evaluation import write_json
from common_regression.metrics import regression_metrics
from common_regression.models import make_adapter
from common_regression.runner import CommonRegressionRunner, Stage
from common_regression.temporal_cv import build_expanding_folds
from prepare_splits import COHORT_USERS, TRUSSES
from run_fixed_13_common_history_temperature import FEATURES
from run_fixed_13_with_tft import (
    TFT_GRID,
    choose_learning_model,
    predict_tft,
    train_tft,
)
from run_fixed_17_features import load_fixed_bundle


ACTIVE_COUNTS = {
    "first_fruits_num": (10, 3, 3),
    "second_fruits_num": (13, 3, 3),
    "third_fruits_num": (13, 3, 3),
}


def select_facility_partition(frame: pd.DataFrame, truss: str) -> dict[str, list[str]]:
    counts = frame.groupby("user_id").size().sort_index()
    active = list(counts.index)
    expected = ACTIVE_COUNTS[truss]
    if len(active) != sum(expected):
        raise RuntimeError(f"{truss}: expected {sum(expected)} active users, got {len(active)}")
    total = len(frame)
    best = None
    indices = range(len(active))
    for val_idx in itertools.combinations(indices, expected[1]):
        val_rows = int(counts.iloc[list(val_idx)].sum())
        remaining = [i for i in indices if i not in val_idx]
        for test_idx in itertools.combinations(remaining, expected[2]):
            test_rows = int(counts.iloc[list(test_idx)].sum())
            split_rows = (total - val_rows - test_rows, val_rows, test_rows)
            ratios = tuple(value / total for value in split_rows)
            error = (
                max(abs(ratios[i] - (0.70, 0.15, 0.15)[i]) for i in range(3)),
                sum((ratios[i] - (0.70, 0.15, 0.15)[i]) ** 2 for i in range(3)),
                tuple(active[i] for i in val_idx),
                tuple(active[i] for i in test_idx),
            )
            if best is None or error < best[0]:
                best = (error, val_idx, test_idx)
    _, val_idx, test_idx = best
    validation = [active[i] for i in val_idx]
    test = [active[i] for i in test_idx]
    train = [user for user in active if user not in set(validation + test)]
    return {"train": train, "validation": validation, "test": test}


def prepare_bundle(truss: str, output: Path) -> tuple[SplitBundle, dict]:
    original, _ = load_fixed_bundle(truss)
    complete = pd.concat([original.train, original.validation, original.load_test()], ignore_index=True)
    complete = complete.sort_values(["target_date", "user_id", "sample_num"], kind="stable").reset_index(drop=True)
    partition = select_facility_partition(complete, truss)
    frames = {
        split: complete[complete.user_id.isin(users)].copy().reset_index(drop=True)
        for split, users in partition.items()
    }
    data_dir = output / "processed_data" / truss
    data_dir.mkdir(parents=True, exist_ok=False)
    for split, frame in frames.items():
        frame.to_parquet(data_dir / f"{split}.parquet", index=False)
    summary = {
        "truss": truss,
        "active_user_count": int(complete.user_id.nunique()),
        "cohort_user_count": len(COHORT_USERS),
        "cohort_users_without_valid_rows": sorted(set(COHORT_USERS) - set(complete.user_id.unique())),
        "splits": {},
    }
    for split, frame in frames.items():
        summary["splits"][split] = {
            "users": partition[split], "user_count": len(partition[split]), "rows": len(frame),
            "row_ratio": len(frame) / len(complete),
            "target_date_min": str(pd.to_datetime(frame.target_date).min().date()),
            "target_date_max": str(pd.to_datetime(frame.target_date).max().date()),
        }
    write_json(data_dir / "split_manifest.json", summary)
    bundle = SplitBundle(
        train=frames["train"], validation=frames["validation"],
        test_path=data_dir / "test.parquet", spec=original.spec,
    )
    return bundle, summary


def tune_tft(train: pd.DataFrame, target: str, date_column: str, config: dict, out: Path):
    trials = []
    folds = build_expanding_folds(train, date_column, n_folds=3)
    for params in TFT_GRID:
        params_json = json.dumps(params, sort_keys=True)
        for seed in config["feature_selection_seeds"]:
            for fold_no, (fit_idx, val_idx) in enumerate(folds, 1):
                fit, validation = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()
                start = time.perf_counter()
                model, dataset, medians, epoch = train_tft(fit, validation, fit, target, params, seed)
                prediction = predict_tft(model, dataset, fit, validation, target, medians, params["batch_size"])
                trials.append({
                    "model": "tft", "params_json": params_json, "seed": seed, "fold": fold_no,
                    "best_epoch": epoch, "training_seconds": time.perf_counter() - start,
                    **regression_metrics(validation[target], prediction),
                })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    table = pd.DataFrame(trials)
    table.to_csv(out / "tft_internal_cv_trials.csv", index=False)
    grouped = table.groupby("params_json", as_index=False).agg(
        mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), std_rmse=("rmse", "std")
    ).sort_values(["mean_rmse", "mean_mae", "params_json"], kind="stable")
    grouped.to_csv(out / "tft_tuning_summary.csv", index=False)
    params_json = str(grouped.iloc[0].params_json)
    params = json.loads(params_json)
    epochs = max(1, int(statistics.median(table[table.params_json.eq(params_json)].best_epoch)))
    write_json(out / "tft_selected_params.json", {"params": params, "fixed_epochs": epochs})
    return params, epochs


def validate_tft(bundle: SplitBundle, params: dict, epochs: int, config: dict):
    predictions, latencies, sizes = [], [], []
    for seed in config["validation_seeds"]:
        model, dataset, medians, _ = train_tft(
            bundle.train, None, pd.DataFrame(columns=bundle.train.columns), bundle.spec.target,
            params, seed, fixed_epochs=epochs,
        )
        start = time.perf_counter()
        prediction = predict_tft(
            model, dataset, pd.DataFrame(columns=bundle.train.columns), bundle.validation,
            bundle.spec.target, medians, params["batch_size"],
        )
        latencies.append((time.perf_counter() - start) * 1000)
        predictions.append(prediction)
        sizes.append(sum(p.numel() * p.element_size() for p in model.parameters()))
    prediction = np.mean(predictions, axis=0)
    return {
        "model": "tft", "is_learning_model": True, "seed_count": len(predictions),
        "inference_latency_ms": float(statistics.median(latencies)),
        "model_size_bytes": int(sum(sizes)), "artifact_count": len(predictions),
        "runtime_dependency_count": 4,
        **regression_metrics(bundle.validation[bundle.spec.target], prediction),
    }


def fit_predict_selected(bundle: SplitBundle, winner: str, runner: CommonRegressionRunner,
                         tft_params: dict, tft_epochs: int, config: dict):
    combined = pd.concat([bundle.train, bundle.validation], ignore_index=True)
    test = bundle.load_test()
    if winner == "tft":
        predictions = []
        for seed in config["validation_seeds"]:
            model, dataset, medians, _ = train_tft(
                combined, None, pd.DataFrame(columns=combined.columns), bundle.spec.target,
                tft_params, seed, fixed_epochs=tft_epochs,
            )
            predictions.append(predict_tft(
                model, dataset, pd.DataFrame(columns=combined.columns), test,
                bundle.spec.target, medians, tft_params["batch_size"],
            ))
        prediction = np.mean(predictions, axis=0)
    else:
        seeds = config["validation_seeds"] if winner in {"mlp", "tabm"} else [config["validation_seeds"][0]]
        predictions = []
        for seed in seeds:
            adapter = make_adapter(winner, FEATURES, [], runner.params[winner], seed)
            adapter.fit(combined, bundle.spec.target, fixed_epochs=runner.epochs[winner])
            predictions.append(adapter.predict(test))
        prediction = np.mean(predictions, axis=0)
    result = regression_metrics(test[bundle.spec.target], prediction)
    persistence = regression_metrics(test[bundle.spec.target], test[bundle.spec.current_value])
    prediction_table = test[list(bundle.spec.row_key) + [bundle.spec.target, bundle.spec.current_value]].copy()
    prediction_table["prediction"] = prediction
    return result, persistence, prediction_table


def run_truss(truss: str, output: Path, config: dict):
    bundle, split_summary = prepare_bundle(truss, output)
    out = output / "artifacts" / truss
    out.mkdir(parents=True, exist_ok=False)
    runner = CommonRegressionRunner(bundle, {"fixed_13": FEATURES}, [], config, out / "five_model_pipeline")
    runner.feature_sets = {"fixed_13": FEATURES}
    runner.selected_feature_set = "fixed_13"
    runner.features = FEATURES
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION
    runner._write_manifest(selected_feature_set="fixed_13")
    runner.tune()
    runner.validate()

    tft_params, tft_epochs = tune_tft(bundle.train, bundle.spec.target, bundle.spec.target_date, config, out)
    tft_metric = validate_tft(bundle, tft_params, tft_epochs, config)
    validation = pd.concat([runner.validation_table, pd.DataFrame([tft_metric])], ignore_index=True)
    validation.to_csv(out / "validation_model_metrics.csv", index=False)
    winner = choose_learning_model(validation)
    write_json(out / "selected_configuration.json", {
        "selected_learning_model": winner, "selection_metric": "validation_rmse",
        "features": FEATURES, "facility_split": split_summary,
        "tft_params": tft_params, "tft_fixed_epochs": tft_epochs,
    })
    final, persistence, prediction = fit_predict_selected(bundle, winner, runner, tft_params, tft_epochs, config)
    prediction.to_parquet(out / "final_test_predictions.parquet", index=False)
    result = {"truss": truss, "selected_learning_model": winner, "final": final, "persistence": persistence}
    write_json(out / "result.json", result)
    write_json(out / "test_access_log.json", {"access_count": 1, "reason": "selected configuration frozen"})
    return result, validation, split_summary


def build_report(output: Path, results: list[dict], validations: list[pd.DataFrame], splits: list[dict]):
    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    validation_parts = []
    for result, table in zip(results, validations):
        table = table.copy(); table.insert(0, "truss", result["truss"]); validation_parts.append(table)
    pd.concat(validation_parts, ignore_index=True).to_csv(summary / "validation_all_model_metrics.csv", index=False)
    rows = []
    for result in results:
        rows.extend([
            {"truss": result["truss"], "model": result["selected_learning_model"], **result["final"]},
            {"truss": result["truss"], "model": "persistence", **result["persistence"]},
        ])
    pd.DataFrame(rows).to_csv(summary / "final_test_metrics.csv", index=False)
    split_rows = []
    for item in splits:
        for split, values in item["splits"].items():
            split_rows.append({"truss": item["truss"], "split": split, **values})
    pd.DataFrame(split_rows).to_csv(summary / "facility_split_summary.csv", index=False)
    lines = [
        "# 착과수 시설 ID holdout 실험", "",
        "- 입력변수: 기존 fixed-13과 동일", "- 모델 후보: Poisson, Random Forest, CatBoost, MLP, TabM, TFT", "- Persistence: 별도 baseline",
        "- 분할: 시설 ID 완전 분리; 행 비율이 70:15:15에 가장 가까운 조합", "- 1화방 10/3/3개 활성 시설, 2·3화방 13/3/3개 시설", "",
        "## 최종 Test", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), "",
        "Test는 Validation으로 최종 모델을 고정한 후 한 번만 접근했다.",
    ]
    (summary / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="facility_holdout_fixed13_with_tft_v1")
    args = parser.parse_args()
    output = ROOT / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((SOURCE_ROOT / "config/model_suite.yaml").read_text())
    write_json(output / "experiment_manifest.json", {
        "split_unit": "user_id", "target_row_ratio": [0.70, 0.15, 0.15],
        "active_facility_counts": ACTIVE_COUNTS, "features": FEATURES,
        "learning_models": ["poisson", "random_forest", "catboost", "mlp", "tabm", "tft"],
        "baseline": "persistence", "test_policy": "access once after Validation selection",
    })
    results, validations, splits = [], [], []
    for truss in TRUSSES:
        print(f"START {truss}", flush=True)
        result, validation, split = run_truss(truss, output, config)
        results.append(result); validations.append(validation); splits.append(split)
        print(f"COMPLETE {truss}", flush=True)
    build_report(output, results, validations, splits)
    print(output)


if __name__ == "__main__":
    main()
