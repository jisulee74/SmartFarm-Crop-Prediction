from __future__ import annotations

import itertools
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_blocks import choose_final_feature_set, summarize_feature_sets
from .metrics import regression_metrics
from .models import make_adapter
from .temporal_cv import build_expanding_folds

LEARNING_MODELS = ("poisson", "random_forest", "catboost", "mlp", "tabm")
DL_MODELS = {"mlp", "tabm"}
RUNTIME_DEPENDENCIES = {"poisson": 1, "random_forest": 1, "catboost": 2, "mlp": 2, "tabm": 3}


def parameter_grid(values: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(values)
    return [dict(zip(keys, choice)) for choice in itertools.product(*(values[k] for k in keys))]


def _model_seeds(model: str, seeds: list[int]) -> list[int]:
    return seeds if model in DL_MODELS else [seeds[0]]


def cross_validate_model(
    train: pd.DataFrame, target: str, date_column: str, feature_set: str,
    features: list[str], categorical: list[str], model: str, params: dict[str, Any],
    seeds: list[int], n_folds: int = 3,
) -> pd.DataFrame:
    rows = []
    folds = build_expanding_folds(train, date_column, n_folds=n_folds)
    for seed in _model_seeds(model, seeds):
        for fold, (fit_idx, val_idx) in enumerate(folds, 1):
            fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
            adapter = make_adapter(model, features, categorical, params, seed)
            adapter.fit(fit, target, validation=validation)
            metric = regression_metrics(validation[target], adapter.predict(validation))
            rows.append({"stage": "internal_cv", "feature_set": feature_set, "model": model,
                         "params_json": json.dumps(params, sort_keys=True), "seed": seed,
                         "fold": fold, "feature_count": len(features), "best_epoch": adapter.best_epoch,
                         **metric})
    return pd.DataFrame(rows)


def select_feature_set_with_fixed_configs(
    train: pd.DataFrame, target: str, date_column: str,
    feature_sets: dict[str, list[str]], categorical: list[str],
    baseline_params: dict[str, dict[str, Any]], seeds: list[int],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    if set(baseline_params) != set(LEARNING_MODELS):
        raise ValueError("Feature selection requires exactly one fixed config per learning model")
    parts = []
    for feature_set, features in feature_sets.items():
        for model in LEARNING_MODELS:
            parts.append(cross_validate_model(train, target, date_column, feature_set, features,
                                              categorical, model, baseline_params[model], seeds))
    trials = pd.concat(parts, ignore_index=True)
    seed_averaged = trials.groupby(
        ["feature_set", "model", "seed", "feature_count"], as_index=False
    ).agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
    model_averaged = seed_averaged.groupby(
        ["feature_set", "model", "feature_count"], as_index=False
    ).agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
    summary = summarize_feature_sets(model_averaged)
    selected = str(choose_final_feature_set(summary).feature_set)
    return selected, trials, summary


def tune_models_on_selected_features(
    train: pd.DataFrame, target: str, date_column: str, selected_feature_set: str,
    features: list[str], categorical: list[str], grids: dict[str, dict[str, list[Any]]],
    seeds: list[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, int | None], pd.DataFrame]:
    if set(grids) != set(LEARNING_MODELS):
        raise ValueError("Tuning grids must cover exactly the learning models")
    parts = []
    for model in LEARNING_MODELS:
        for params in parameter_grid(grids[model]):
            parts.append(cross_validate_model(train, target, date_column, selected_feature_set,
                                              features, categorical, model, params, seeds))
    trials = pd.concat(parts, ignore_index=True)
    grouped = trials.groupby(["model", "params_json"], as_index=False).agg(
        mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), std_rmse=("rmse", "std")
    ).sort_values(["model", "mean_rmse", "mean_mae", "params_json"], kind="stable")
    selected_params, epochs = {}, {}
    for model, rows in grouped.groupby("model", sort=False):
        winner = rows.iloc[0]
        selected_params[model] = json.loads(winner.params_json)
        chosen_epochs = trials[(trials.model.eq(model)) & (trials.params_json.eq(winner.params_json))].best_epoch.dropna()
        epochs[model] = int(statistics.median(chosen_epochs)) if len(chosen_epochs) else None
    return selected_params, epochs, trials


def _measure_inference(adapter, validation: pd.DataFrame) -> float:
    for _ in range(5):
        adapter.predict(validation)
    times = []
    for _ in range(30):
        start = time.perf_counter(); adapter.predict(validation); times.append(time.perf_counter() - start)
    return float(statistics.median(times) * 1000)


def validate_learning_models(
    train: pd.DataFrame, validation: pd.DataFrame, target: str, features: list[str],
    categorical: list[str], params: dict[str, dict[str, Any]], epochs: dict[str, int | None],
    seeds: list[int], current_value: str,
) -> tuple[str, pd.DataFrame, dict[str, list[Any]]]:
    rows, fitted = [], {}
    for model in LEARNING_MODELS:
        predictions, adapters = [], []
        latencies, sizes = [], []
        for seed in _model_seeds(model, seeds):
            adapter = make_adapter(model, features, categorical, params[model], seed)
            adapter.fit(train, target, fixed_epochs=epochs.get(model))
            predictions.append(adapter.predict(validation)); adapters.append(adapter)
            latencies.append(_measure_inference(adapter, validation))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("model.cbm" if model == "catboost" else "model.bin")
                adapter.save(path); sizes.append(path.stat().st_size)
        prediction = np.mean(predictions, axis=0)
        rows.append({"model": model, "is_learning_model": True,
                     "seed_count": len(predictions), "inference_latency_ms": sum(latencies),
                     "model_size_bytes": sum(sizes), "artifact_count": len(adapters),
                     "runtime_dependency_count": RUNTIME_DEPENDENCIES[model],
                     **regression_metrics(validation[target], prediction)})
        fitted[model] = adapters
    persistence = validation[current_value].to_numpy(float)
    rows.append({"model": "persistence", "is_learning_model": False, "seed_count": 0,
                 "inference_latency_ms": 0.0, "model_size_bytes": 0,
                 "artifact_count": 0, "runtime_dependency_count": 0,
                 **regression_metrics(validation[target], persistence)})
    table = pd.DataFrame(rows)
    learning = table[table.is_learning_model].copy()
    best_rmse = learning.rmse.min(); tied = learning[learning.rmse.le(best_rmse * 1.01)]
    best_mae = tied.mae.min(); tied = tied[tied.mae.le(best_mae * 1.01)]
    winner = tied.sort_values(["inference_latency_ms", "model_size_bytes", "runtime_dependency_count",
                               "artifact_count", "model"], kind="stable").iloc[0]
    return str(winner.model), table, fitted
