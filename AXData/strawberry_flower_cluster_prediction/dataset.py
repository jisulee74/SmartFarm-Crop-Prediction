from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from common_regression.contracts import SplitBundle, TargetSpec

ROOT = Path(__file__).resolve().parent


def _semantic_group(feature: str) -> str:
    for suffix in ("_daily_change", "_roll4_mean", "_roll4_min", "_roll4_max", "_roll4_std",
                   "_lag1", "_change", "_roll4_slope_per_day"):
        if feature.endswith(suffix):
            return feature[:-len(suffix)]
    return feature


def load_dataset() -> tuple[SplitBundle, dict[str, list[str]], pd.DataFrame]:
    frames = {split: pd.read_parquet(ROOT / "processed_data" / f"{split}_candidates.parquet")
              for split in ("train", "val")}
    manifest = json.loads((ROOT / "processed_data" / "manifest.json").read_text())
    features = list(dict.fromkeys([*manifest["extended_features"], "fruit_cluster_num_lag1",
                                   "fruit_cluster_num_change", "fruit_cluster_num_roll4_slope_per_day"]))
    history_names = {
        "fruit_cluster_num", "fruit_cluster_num_lag1", "fruit_cluster_num_change",
        "fruit_cluster_num_daily_change", "fruit_cluster_num_roll4_mean",
        "fruit_cluster_num_roll4_std", "fruit_cluster_num_roll4_slope_per_day",
        "days_since_previous", "days_since_crop_start", "week_of_year_sin", "week_of_year_cos",
    }
    history = [f for f in features if f in history_names]
    blocks = {"individual_history": history, "growth": [f for f in features if f not in history]}
    exclusions = ("next_fruit_cluster_num", "target_delta", "target_date", "target_split",
                  "input_raw_split", "raw_split", "sn", "user_id", "crop_cycle_id",
                  "sample_num", "meas_date", "crop_start_date", "cropping_serl_no")
    spec = TargetSpec("flower_cluster", "next_fruit_cluster_num", "fruit_cluster_num", "meas_date",
                      "target_date", ("user_id", "crop_cycle_id", "sample_num"),
                      ("user_id", "crop_cycle_id", "sample_num", "target_date"), exclusions, 0.90,
                      {k: tuple(v) for k, v in blocks.items()},
                      {name: "source_unavailable_for_2016_2019_period" for name in
                       ("nutrient", "indoor", "outdoor", "control")})
    catalog = pd.DataFrame([{
        "feature": f, "block": "individual_history" if f in history else "growth",
        "source_variable": _semantic_group(f), "semantic_group": _semantic_group(f),
        "is_quality_indicator": False, "inference_available": True,
        "is_direct": f == _semantic_group(f), "derivation_complexity": 0 if f == _semantic_group(f) else 1,
        "semantic_equivalent": True,
    } for f in features])
    return SplitBundle(frames["train"], frames["val"],
                       ROOT / "processed_data/test_candidates.parquet", spec), blocks, catalog
