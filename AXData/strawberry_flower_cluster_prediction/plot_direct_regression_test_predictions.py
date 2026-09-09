#!/usr/bin/env python3
"""Plot original direct-regression CatBoost Test predictions against Persistence."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "artifacts/common_model_selection_v1_3/20260819_full_v1_3_retry1"


def main() -> None:
    raw = pd.concat([pd.read_parquet(ROOT / "data" / f"{split}.parquet") for split in ("train", "val", "test")], ignore_index=True)
    raw["meas_date"] = pd.to_datetime(raw.meas_date)
    prediction = pd.read_parquet(RUN / "final_test_predictions.parquet").rename(
        columns={"actual": "target_actual", "prediction": "catboost_prediction"})
    prediction["target_date"] = pd.to_datetime(prediction.target_date)
    prediction["persistence_prediction"] = prediction.groupby(
        ["user_id", "crop_cycle_id", "sample_num"], sort=False)["target_actual"].shift(1)
    # The first Test target's current value comes from the raw observation immediately before it.
    lookup = raw.set_index(["user_id", "crop_cycle_id", "sample_num", "meas_date"])["fruit_cluster_num"]
    source = pd.read_parquet(ROOT / "processed_data/test_candidates.parquet")
    keys = ["user_id", "crop_cycle_id", "sample_num", "target_date"]
    current = source[keys + ["fruit_cluster_num"]].rename(columns={"fruit_cluster_num": "persistence_from_source"})
    prediction = prediction.merge(current, on=keys, how="left", validate="one_to_one")
    prediction["persistence_prediction"] = prediction.persistence_from_source

    cycles = raw[["user_id", "crop_cycle_id", "crop_start_date"]].drop_duplicates().sort_values(["user_id", "crop_start_date"])
    fig, axes = plt.subplots(len(cycles), 4, figsize=(24, 15), squeeze=False)
    for row_index, cycle in enumerate(cycles.itertuples(index=False)):
        for col_index, sample in enumerate(("1", "2", "3", "4")):
            axis = axes[row_index, col_index]
            actual = raw[(raw.user_id == cycle.user_id) & (raw.crop_cycle_id == cycle.crop_cycle_id) & (raw.sample_num.astype(str) == sample)].sort_values("meas_date")
            pred = prediction[(prediction.user_id == cycle.user_id) & (prediction.crop_cycle_id == cycle.crop_cycle_id) & (prediction.sample_num.astype(str) == sample)].sort_values("target_date")
            axis.plot(actual.meas_date, actual.fruit_cluster_num, color="#222222", marker="o", ms=3.5, lw=2.2, label="Actual")
            if not pred.empty:
                axis.plot(pred.target_date, pred.persistence_prediction, color="#E68613", ls="--", marker="s", ms=4, lw=2, label="Persistence")
                axis.plot(pred.target_date, pred.catboost_prediction, color="#2878B5", marker="^", markerfacecolor="white", ms=5, lw=2, label="Direct CatBoost")
                axis.axvline(pred.target_date.min(), color="#777777", ls=":", lw=1.2)
            axis.set_title(f"{cycle.user_id} · Sample {sample} ({pd.Timestamp(cycle.crop_start_date).year})", fontsize=15)
            axis.grid(alpha=.22); axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            axis.tick_params(axis="x", labelsize=11); axis.tick_params(axis="y", labelsize=12)
            if col_index == 0: axis.set_ylabel("Fruit-cluster count", fontsize=14)
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    if not handles: handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=17, bbox_to_anchor=(.5, .965))
    fig.suptitle("Original Direct-regression CatBoost: Historical Test Predictions", fontsize=25, y=.995)
    fig.text(.5, .012, "Dotted lines mark the beginning of each historical Test period. Only Test predictions are shown.", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, .04, 1, .935), h_pad=.8, w_pad=1.0)
    output = RUN / "exploration"; output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "direct_catboost_test_timeseries.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    prediction.to_csv(output / "direct_catboost_test_timeseries_data.csv", index=False, encoding="utf-8-sig")
    print({"figure": str(output / "direct_catboost_test_timeseries.png"), "rows": len(prediction)})


if __name__ == "__main__": main()
