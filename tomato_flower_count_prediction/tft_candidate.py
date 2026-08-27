"""Official PyTorch-Forecasting TFT candidate for the tomato experiment."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "strawberry_fruit_set_prediction"
sys.path.insert(0, str(SOURCE))

import run_fixed_13_with_tft as backend

from common_regression.metrics import regression_metrics
from common_regression.temporal_cv import build_expanding_folds

TFT_GRID = backend.TFT_GRID


def configure(features: list[str], entity: list[str], current_value: str) -> None:
    backend.FEATURES = features
    backend.ENTITY = entity

    def make_long_sequences(context: pd.DataFrame, evaluation: pd.DataFrame, target: str) -> pd.DataFrame:
        context = context.copy().assign(_is_eval=False, _eval_order=-1)
        evaluation = evaluation.copy().assign(_is_eval=True, _eval_order=np.arange(len(evaluation)))
        combined = pd.concat([context, evaluation], ignore_index=True)
        combined["feature_date"] = pd.to_datetime(combined["feature_date"])
        combined["target_date"] = pd.to_datetime(combined["target_date"])
        combined = combined.sort_values([*entity, "feature_date", "target_date", "_is_eval"], kind="stable")
        rows = []
        for _, group in combined.groupby(entity, sort=False, dropna=False):
            group = group.reset_index(drop=True)
            for position in np.flatnonzero(group["_is_eval"].to_numpy()):
                history = group.iloc[max(0, position - backend.ENCODER_LENGTH):position]
                current = group.iloc[[position]]
                encoder = history.tail(backend.ENCODER_LENGTH)
                if encoder.empty:
                    encoder = pd.concat([current] * backend.ENCODER_LENGTH, ignore_index=True)
                    encoder[target] = encoder[current_value].to_numpy()
                elif len(encoder) < backend.ENCODER_LENGTH:
                    padding = pd.concat([encoder.iloc[[0]]] * (backend.ENCODER_LENGTH - len(encoder)), ignore_index=True)
                    encoder = pd.concat([padding, encoder], ignore_index=True)
                sequence = pd.concat([encoder, current], ignore_index=True)
                sequence_id = f"seq_{int(current['_eval_order'].iloc[0])}"
                for time_idx, (_, source) in enumerate(sequence.iterrows()):
                    item = {"sequence_id": sequence_id, "time_idx": time_idx, target: float(source[target])}
                    item.update({feature: float(source[feature]) for feature in features})
                    rows.append(item)
        return pd.DataFrame(rows)

    backend.make_long_sequences = make_long_sequences


def tune(train: pd.DataFrame, target: str, date_column: str, features: list[str], entity: list[str],
         current_value: str, seeds: list[int], output: Path):
    configure(features, entity, current_value)
    trials = []
    folds = build_expanding_folds(train, date_column, n_folds=3)
    for params in TFT_GRID:
        params_json = json.dumps(params, sort_keys=True)
        for seed in seeds:
            for fold_no, (fit_idx, val_idx) in enumerate(folds, 1):
                fit, validation = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()
                start = time.perf_counter()
                model, dataset, medians, epoch = backend.train_tft(fit, validation, fit, target, params, seed)
                prediction = backend.predict_tft(model, dataset, fit, validation, target, medians, params["batch_size"])
                trials.append({"model": "tft", "params_json": params_json, "seed": seed, "fold": fold_no,
                               "best_epoch": epoch, "training_seconds": time.perf_counter() - start,
                               **regression_metrics(validation[target], prediction)})
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    table = pd.DataFrame(trials)
    table.to_csv(output / "tft_internal_cv_trials.csv", index=False)
    summary = table.groupby("params_json", as_index=False).agg(
        mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), std_rmse=("rmse", "std")
    ).sort_values(["mean_rmse", "mean_mae", "params_json"], kind="stable")
    summary.to_csv(output / "tft_tuning_summary.csv", index=False)
    params_json = str(summary.iloc[0].params_json)
    params = json.loads(params_json)
    epochs = max(1, int(statistics.median(table.loc[table.params_json.eq(params_json), "best_epoch"])))
    return params, epochs


def validate(train, validation, target, features, entity, current_value, params, epochs, seeds):
    configure(features, entity, current_value)
    predictions, latencies, sizes = [], [], []
    empty = pd.DataFrame(columns=train.columns)
    for seed in seeds:
        model, dataset, medians, _ = backend.train_tft(train, None, empty, target, params, seed, fixed_epochs=epochs)
        start = time.perf_counter()
        pred = backend.predict_tft(model, dataset, empty, validation, target, medians, params["batch_size"])
        latencies.append((time.perf_counter() - start) * 1000); predictions.append(pred)
        sizes.append(sum(p.numel() * p.element_size() for p in model.parameters()))
    prediction = np.mean(predictions, axis=0)
    return {"model": "tft", "is_learning_model": True, "seed_count": len(seeds),
            "inference_latency_ms": float(statistics.median(latencies)), "model_size_bytes": int(sum(sizes)),
            "artifact_count": len(seeds), "runtime_dependency_count": 4,
            **regression_metrics(validation[target], prediction)}


def fit_predict(train, validation, test, target, features, entity, current_value, params, epochs, seeds):
    configure(features, entity, current_value)
    combined = pd.concat([train, validation], ignore_index=True)
    empty = pd.DataFrame(columns=combined.columns)
    predictions = []
    for seed in seeds:
        model, dataset, medians, _ = backend.train_tft(combined, None, empty, target, params, seed, fixed_epochs=epochs)
        predictions.append(backend.predict_tft(model, dataset, empty, test, target, medians, params["batch_size"]))
    return np.mean(predictions, axis=0)
