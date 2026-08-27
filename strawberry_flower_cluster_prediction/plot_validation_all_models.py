#!/usr/bin/env python3
"""Reproduce and plot all original direct-regression Validation predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from common_regression.metrics import regression_metrics
from common_regression.models import make_adapter
from dataset import load_dataset

RUN = ROOT / "artifacts/common_model_selection_v1_3/20260819_full_v1_3_retry1"
MODEL_LABELS = {
    "poisson": "Poisson", "random_forest": "Random Forest", "catboost": "CatBoost",
    "mlp": "MLP", "tabm": "TabM", "persistence": "Persistence",
}
COLORS = {
    "poisson": "#9467BD", "random_forest": "#2CA02C", "catboost": "#2878B5",
    "mlp": "#D62728", "tabm": "#17BECF", "persistence": "#E68613",
}


def reproduce_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle, _, _ = load_dataset()
    selected = json.loads((RUN / "selected_configuration.json").read_text())
    tuned = json.loads((RUN / "internal_cv/hyperparameter_tuning/selected_params.json").read_text())
    config = yaml.safe_load((ROOT / "config/model_suite_v1_3.yaml").read_text())
    features = selected["features"]
    output = bundle.validation[[*bundle.spec.row_key, bundle.spec.target]].copy().rename(
        columns={bundle.spec.target: "actual"})
    output["persistence"] = bundle.validation[bundle.spec.current_value].to_numpy(float)
    for model in ("poisson", "random_forest", "catboost", "mlp", "tabm"):
        seeds = config["validation_seeds"] if model in {"mlp", "tabm"} else [config["validation_seeds"][0]]
        predictions = []
        for seed in seeds:
            adapter = make_adapter(model, features, [], tuned["params"][model], seed)
            adapter.fit(bundle.train, bundle.spec.target, fixed_epochs=tuned["fixed_epochs"][model])
            predictions.append(adapter.predict(bundle.validation))
        output[model] = np.mean(predictions, axis=0)
    metrics = []
    for model in ("poisson", "random_forest", "catboost", "mlp", "tabm", "persistence"):
        metrics.append({"model": model, **regression_metrics(output.actual, output[model])})
    return output, pd.DataFrame(metrics)


def plot(prediction: pd.DataFrame, output_path: Path) -> None:
    raw = pd.concat([pd.read_parquet(ROOT / "data" / f"{split}.parquet") for split in ("train", "val", "test")], ignore_index=True)
    raw["meas_date"] = pd.to_datetime(raw.meas_date); prediction["target_date"] = pd.to_datetime(prediction.target_date)
    cycles = raw[["user_id", "crop_cycle_id", "crop_start_date"]].drop_duplicates().sort_values(["user_id", "crop_start_date"])
    fig, axes = plt.subplots(len(cycles), 4, figsize=(25, 16), squeeze=False)
    styles = {
        "persistence": {"ls": "--", "marker": "s"}, "poisson": {"ls": "-", "marker": "x"},
        "random_forest": {"ls": "-", "marker": "D"}, "catboost": {"ls": "-", "marker": "^"},
        "mlp": {"ls": ":", "marker": "v"}, "tabm": {"ls": "-.", "marker": "P"},
    }
    for row_index, cycle in enumerate(cycles.itertuples(index=False)):
        for col_index, sample in enumerate(("1", "2", "3", "4")):
            axis = axes[row_index, col_index]
            actual = raw[(raw.user_id == cycle.user_id) & (raw.crop_cycle_id == cycle.crop_cycle_id) & (raw.sample_num.astype(str) == sample)].sort_values("meas_date")
            pred = prediction[(prediction.user_id == cycle.user_id) & (prediction.crop_cycle_id == cycle.crop_cycle_id) & (prediction.sample_num.astype(str) == sample)].sort_values("target_date")
            axis.plot(actual.meas_date, actual.fruit_cluster_num, color="#222222", lw=2.5, marker="o", ms=3.5, label="Actual", zorder=8)
            for model in ("persistence", "poisson", "random_forest", "catboost", "mlp", "tabm"):
                axis.plot(pred.target_date, pred[model], color=COLORS[model], lw=1.65, ms=4,
                          alpha=.88, label=MODEL_LABELS[model], **styles[model])
            if not pred.empty: axis.axvline(pred.target_date.min(), color="#777777", lw=1, ls=":")
            axis.set_title(f"{cycle.user_id} · Sample {sample} ({pd.Timestamp(cycle.crop_start_date).year})", fontsize=15)
            axis.grid(alpha=.2); axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            axis.tick_params(axis="x", labelsize=11); axis.tick_params(axis="y", labelsize=11)
            if col_index == 0: axis.set_ylabel("Fruit-cluster count", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=15, bbox_to_anchor=(.5, .968))
    fig.suptitle("Direct-regression Models: External Validation Predictions", fontsize=25, y=.995)
    fig.text(.5, .012, "Model lines show only the external Validation period; black lines show observed counts over each crop cycle.", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, .04, 1, .93), h_pad=.8, w_pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    prediction, metrics = reproduce_predictions()
    expected = pd.read_csv(RUN / "validation/model_metrics.csv")
    merged = metrics.merge(expected[["model", "rmse", "mae", "r2", "ccc"]], on="model", suffixes=("_new", "_expected"), validate="one_to_one")
    differences = {metric: float(np.max(np.abs(merged[f"{metric}_new"] - merged[f"{metric}_expected"]))) for metric in ("rmse", "mae", "r2", "ccc")}
    if max(differences.values()) > 1e-6: raise RuntimeError(f"Reproduced metrics differ: {differences}")
    output = RUN / "exploration"; output.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output / "validation_all_model_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "validation_all_model_metrics_reproduced.csv", index=False)
    plot(prediction, output / "validation_all_model_timeseries.png")
    print({"rows": len(prediction), "metric_max_abs_differences": differences,
           "figure": str(output / "validation_all_model_timeseries.png")})


if __name__ == "__main__": main()
