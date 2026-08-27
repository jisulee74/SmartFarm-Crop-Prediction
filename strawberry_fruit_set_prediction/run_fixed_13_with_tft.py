#!/usr/bin/env python3
"""Add an official PyTorch-Forecasting TFT candidate to the fixed-13 experiment.

The five existing learning-model validation results are immutable inputs. Only
TFT is tuned on the same Train expanding-window folds and evaluated on the same
external Validation rows. Test is accessed only if TFT becomes the selected
learning model; otherwise the already-recorded original final result is reused.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
BASE_RUN = ROOT / "artifacts/fixed_13_common_history_temperature_v1"
sys.path[:0] = [str(ROOT.parent), str(ROOT)]

from common_regression.evaluation import write_json
from common_regression.metrics import regression_metrics
from common_regression.temporal_cv import build_expanding_folds
from prepare_splits import TRUSSES
from run_fixed_13_common_history_temperature import FEATURES
from run_fixed_17_features import load_fixed_bundle


ENTITY = ["user_id", "sample_num", "crop_sn"]
ENCODER_LENGTH = 4
TFT_GRID = [
    {"hidden_size": 8, "attention_head_size": 1, "hidden_continuous_size": 8,
     "dropout": 0.1, "learning_rate": 0.01, "batch_size": 64, "max_epochs": 80},
    {"hidden_size": 16, "attention_head_size": 2, "hidden_continuous_size": 8,
     "dropout": 0.1, "learning_rate": 0.003, "batch_size": 64, "max_epochs": 80},
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_imputation(frame: pd.DataFrame) -> dict[str, float]:
    medians = {}
    for feature in FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        value = float(values.median())
        if not np.isfinite(value):
            raise ValueError(f"Train median is unavailable for {feature}")
        medians[feature] = value
    return medians


def apply_imputation(frame: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    result = frame.copy()
    for feature, median in medians.items():
        result[feature] = pd.to_numeric(result[feature], errors="coerce").fillna(median)
    return result


def make_long_sequences(context: pd.DataFrame, evaluation: pd.DataFrame, target: str) -> pd.DataFrame:
    """Create one fixed-length pseudo-series per evaluation row without crossing entity boundaries."""
    context = context.copy().assign(_is_eval=False, _eval_order=-1)
    evaluation = evaluation.copy().assign(_is_eval=True, _eval_order=np.arange(len(evaluation)))
    combined = pd.concat([context, evaluation], ignore_index=True)
    combined["feature_date"] = pd.to_datetime(combined["feature_date"])
    combined["target_date"] = pd.to_datetime(combined["target_date"])
    combined = combined.sort_values([*ENTITY, "feature_date", "target_date", "_is_eval"], kind="stable")
    rows = []
    for _, group in combined.groupby(ENTITY, sort=False, dropna=False):
        group = group.reset_index(drop=True)
        for position in np.flatnonzero(group["_is_eval"].to_numpy()):
            history = group.iloc[max(0, position - ENCODER_LENGTH):position]
            current = group.iloc[[position]]
            encoder = history.tail(ENCODER_LENGTH)
            if encoder.empty:
                # Current state is known at prediction time. Repeat it only as
                # left padding, and use current_fruit_count as a historical target.
                encoder = pd.concat([current] * ENCODER_LENGTH, ignore_index=True)
                encoder[target] = encoder["current_fruit_count"].to_numpy()
            elif len(encoder) < ENCODER_LENGTH:
                padding = pd.concat([encoder.iloc[[0]]] * (ENCODER_LENGTH - len(encoder)), ignore_index=True)
                encoder = pd.concat([padding, encoder], ignore_index=True)
            sequence = pd.concat([encoder, current], ignore_index=True)
            sequence_id = f"seq_{int(current['_eval_order'].iloc[0])}"
            for time_idx, (_, source) in enumerate(sequence.iterrows()):
                item = {"sequence_id": sequence_id, "time_idx": time_idx, target: float(source[target])}
                item.update({feature: float(source[feature]) for feature in FEATURES})
                rows.append(item)
    return pd.DataFrame(rows)


def make_datasets(train_long: pd.DataFrame, validation_long: pd.DataFrame, target: str):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder
    from pytorch_forecasting.data.encoders import TorchNormalizer

    training = TimeSeriesDataSet(
        train_long,
        time_idx="time_idx",
        target=target,
        group_ids=["sequence_id"],
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_known_reals=FEATURES,
        time_varying_unknown_reals=[target],
        target_normalizer=TorchNormalizer(method="standard"),
        categorical_encoders={"sequence_id": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, validation_long, predict=False, stop_randomization=True
    )
    return training, validation


def train_tft(
    fit: pd.DataFrame,
    validation: pd.DataFrame | None,
    context: pd.DataFrame,
    target: str,
    params: dict,
    seed: int,
    fixed_epochs: int | None = None,
):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import RMSE

    seed_everything(seed)
    medians = fit_imputation(fit)
    fit_i = apply_imputation(fit, medians)
    validation_i = apply_imputation(validation, medians) if validation is not None else None
    context_i = apply_imputation(context, medians)
    train_long = make_long_sequences(pd.DataFrame(columns=fit_i.columns), fit_i, target)
    if validation_i is None:
        # A structural validation set is required to instantiate from the same
        # dataset parameters; reuse the final training rows but do not early-stop.
        validation_i = fit_i.tail(min(max(8, len(fit_i) // 10), len(fit_i))).copy()
        context_i = fit_i.iloc[:-len(validation_i)].copy()
    val_long = make_long_sequences(context_i, validation_i, target)
    train_ds, val_ds = make_datasets(train_long, val_long, target)
    batch_size = min(int(params["batch_size"]), len(train_ds))
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=min(batch_size, len(val_ds)), num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=float(params["learning_rate"]),
        hidden_size=int(params["hidden_size"]),
        attention_head_size=int(params["attention_head_size"]),
        hidden_continuous_size=int(params["hidden_continuous_size"]),
        dropout=float(params["dropout"]),
        output_size=1,
        loss=RMSE(),
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )
    max_epochs = int(fixed_epochs or params["max_epochs"])
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = ModelCheckpoint(dirpath=directory, monitor="val_loss", mode="min", save_top_k=1)
        callbacks = [checkpoint]
        if fixed_epochs is None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min", min_delta=1e-5))
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=callbacks,
            deterministic="warn",
        )
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        best_epoch = int(trainer.current_epoch + 1)
        if fixed_epochs is None and checkpoint.best_model_path:
            model = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    model.to("cpu")
    return model, train_ds, medians, best_epoch


def predict_tft(model, training_dataset, context, evaluation, target, medians, batch_size=64):
    evaluation_i = apply_imputation(evaluation, medians)
    context_i = apply_imputation(context, medians)
    long = make_long_sequences(context_i, evaluation_i, target)
    from pytorch_forecasting import TimeSeriesDataSet
    dataset = TimeSeriesDataSet.from_dataset(training_dataset, long, predict=False, stop_randomization=True)
    loader = dataset.to_dataloader(train=False, batch_size=min(batch_size, len(dataset)), num_workers=0)
    prediction = model.predict(loader, mode="prediction", return_index=True, trainer_kwargs={"logger": False, "enable_progress_bar": False})
    values = prediction.output.detach().cpu().numpy().reshape(-1)
    index = prediction.index.copy()
    index["prediction"] = values
    index["_eval_order"] = index["sequence_id"].str.replace("seq_", "", regex=False).astype(int)
    return index.sort_values("_eval_order")["prediction"].to_numpy(float)


def choose_learning_model(table: pd.DataFrame) -> str:
    learning = table[table.is_learning_model].copy()
    best_rmse = learning.rmse.min()
    tied = learning[learning.rmse.le(best_rmse * 1.01)]
    best_mae = tied.mae.min()
    tied = tied[tied.mae.le(best_mae * 1.01)]
    return str(tied.sort_values(["inference_latency_ms", "model_size_bytes", "model"], kind="stable").iloc[0].model)


def run_truss(truss: str, output: Path, config: dict) -> dict:
    bundle, _ = load_fixed_bundle(truss)
    target = bundle.spec.target
    train, validation = bundle.train.copy(), bundle.validation.copy()
    out = output / truss
    out.mkdir(parents=True, exist_ok=False)
    trials = []
    folds = build_expanding_folds(train, bundle.spec.target_date, n_folds=3)
    for params in TFT_GRID:
        params_json = json.dumps(params, sort_keys=True)
        for seed in config["feature_selection_seeds"]:
            for fold_no, (fit_idx, val_idx) in enumerate(folds, 1):
                fit, val = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()
                start = time.perf_counter()
                model, dataset, medians, epoch = train_tft(
                    fit, val, fit, target, params, seed
                )
                pred = predict_tft(model, dataset, fit, val, target, medians, params["batch_size"])
                trials.append({
                    "model": "tft", "params_json": params_json, "seed": seed, "fold": fold_no,
                    "best_epoch": epoch, "training_seconds": time.perf_counter() - start,
                    **regression_metrics(val[target], pred),
                })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(out / "tft_internal_cv_trials.csv", index=False)
    grouped = trials_df.groupby("params_json", as_index=False).agg(
        mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), std_rmse=("rmse", "std")
    ).sort_values(["mean_rmse", "mean_mae", "params_json"], kind="stable")
    selected_json = str(grouped.iloc[0].params_json)
    selected_params = json.loads(selected_json)
    epochs = trials_df[trials_df.params_json.eq(selected_json)].best_epoch
    fixed_epochs = max(1, int(statistics.median(epochs)))
    grouped.to_csv(out / "tft_tuning_summary.csv", index=False)
    write_json(out / "tft_selected_params.json", {"params": selected_params, "fixed_epochs": fixed_epochs})

    predictions, latencies, models = [], [], []
    for seed in config["validation_seeds"]:
        model, dataset, medians, _ = train_tft(
            train, None, pd.DataFrame(columns=train.columns), target,
            selected_params, seed, fixed_epochs=fixed_epochs,
        )
        start = time.perf_counter()
        pred = predict_tft(model, dataset, train, validation, target, medians, selected_params["batch_size"])
        latencies.append((time.perf_counter() - start) * 1000)
        predictions.append(pred)
        models.append((model, dataset, medians))
    val_prediction = np.mean(predictions, axis=0)
    tft_metric = {
        "model": "tft", "is_learning_model": True, "seed_count": len(predictions),
        "inference_latency_ms": float(statistics.median(latencies)),
        "model_size_bytes": int(sum(sum(p.numel() * p.element_size() for p in m.parameters()) for m, _, _ in models)),
        "artifact_count": len(models), "runtime_dependency_count": 4,
        **regression_metrics(validation[target], val_prediction),
    }
    original = pd.read_csv(BASE_RUN / truss / "validation/model_metrics.csv")
    combined = pd.concat([original, pd.DataFrame([tft_metric])], ignore_index=True)
    combined.to_csv(out / "validation_model_metrics_with_tft.csv", index=False)
    winner = choose_learning_model(combined)
    result = {"truss": truss, "selected_learning_model": winner, "tft_validation": tft_metric}

    # Test protection: only access Test when the newly added TFT wins Validation.
    if winner == "tft":
        combined_fit = pd.concat([train, validation], ignore_index=True)
        test = bundle.load_test()
        final_predictions = []
        for seed in config["validation_seeds"]:
            model, dataset, medians, _ = train_tft(
                combined_fit, None, pd.DataFrame(columns=combined_fit.columns), target,
                selected_params, seed, fixed_epochs=fixed_epochs,
            )
            final_predictions.append(predict_tft(
                model, dataset, combined_fit, test, target, medians, selected_params["batch_size"]
            ))
        prediction = np.mean(final_predictions, axis=0)
        result["test"] = regression_metrics(test[target], prediction)
        result["persistence"] = regression_metrics(test[target], test[bundle.spec.current_value])
        result["test_access"] = "TFT selected on Validation; Test accessed once"
    else:
        original_test = pd.read_csv(BASE_RUN / "summary/final_test_metrics.csv")
        selected = original_test[(original_test.truss.eq(truss)) & (original_test.model.eq(winner))]
        baseline = original_test[(original_test.truss.eq(truss)) & (original_test.model.eq("persistence"))]
        result["test"] = selected.iloc[0][["rmse", "mae", "r2", "ccc"]].to_dict()
        result["persistence"] = baseline.iloc[0][["rmse", "mae", "r2", "ccc"]].to_dict()
        result["test_access"] = "TFT not selected; reused original recorded Test result without loading Test"
    write_json(out / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="tft_candidate_v1")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/model_suite.yaml").read_text())
    output = BASE_RUN / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment_manifest.json", {
        "base_experiment": str(BASE_RUN), "features": FEATURES,
        "new_candidate": "pytorch_forecasting.TemporalFusionTransformer",
        "pytorch_forecasting_version": "1.8.0", "encoder_length": ENCODER_LENGTH,
        "prediction_length": 1, "entity_boundary": ENTITY,
        "row_policy": "left-pad short histories to preserve identical evaluation rows",
        "tft_grid": TFT_GRID, "test_policy": "access only if TFT wins external Validation",
    })
    results = []
    for truss in TRUSSES:
        print(f"START {truss}", flush=True)
        results.append(run_truss(truss, output, config))
        print(f"COMPLETE {truss}", flush=True)
    rows = []
    for result in results:
        rows.extend([
            {"truss": result["truss"], "model": result["selected_learning_model"], **result["test"]},
            {"truss": result["truss"], "model": "persistence", **result["persistence"]},
        ])
    pd.DataFrame(rows).to_csv(output / "final_test_metrics.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()
