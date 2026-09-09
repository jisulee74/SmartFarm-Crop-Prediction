#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from common_regression.contracts import SplitBundle
from common_regression.evaluation import write_json
from common_regression.runner import CommonRegressionRunner, Stage
from dataset import source_module
from prepare_splits import TRUSSES

SENSOR_VARIABLES = ["EI__TI", "EI__HI", "EI__CI", "EI__IS", "EI__CR"]
FEATURES = [
    "current_fruit_count",
    "previous_fruit_count",
    "days_since_previous",
    "days_since_first_positive",
    "sensor_EI__TI_7d_mean",
    "sensor_EI__HI_7d_mean",
    "sensor_EI__CI_7d_mean",
    "sensor_EI__IS_1d_last",
    "sensor_EI__CR_1d_last",
    "fruit_cluster_num",
    "previous_fruit_cluster_num",
    "first_flower_num",
    "second_flower_num",
    "third_flower_num",
    "previous_first_flower_num",
    "previous_second_flower_num",
    "previous_third_flower_num",
]
SHIFT_COLUMNS = {
    "fruit_cluster_num": "previous_fruit_cluster_num",
    "first_flower_num": "previous_first_flower_num",
    "second_flower_num": "previous_second_flower_num",
    "third_flower_num": "previous_third_flower_num",
}


def load_fixed_bundle(truss: str) -> tuple[SplitBundle, pd.DataFrame]:
    from dataset import load_dataset

    base_bundle, _, _, _ = load_dataset(truss)
    parts = []
    for split in ("train", "val", "test"):
        frame = pd.read_parquet(ROOT / "processed_data" / truss / f"{split}_candidates.parquet")
        parts.append(frame.assign(_dataset_split=split))
    combined = pd.concat(parts, ignore_index=True)
    combined["feature_date"] = pd.to_datetime(combined["feature_date"])
    group = ["user_id", "sample_num", "crop_sn"]
    combined = combined.sort_values([*group, "feature_date", "target_date"], kind="stable")
    for current, previous in SHIFT_COLUMNS.items():
        combined[previous] = combined.groupby(group, sort=False)[current].shift(1)

    src = source_module()
    sensor = pd.read_parquet(ROOT / "data/sensor_category_cache/sensor_collapsed.parquet")
    sensor["meas_date"] = pd.to_datetime(sensor["meas_date"])
    combined = src._add_sensor_features(combined, sensor, SENSOR_VARIABLES)
    frames = {
        split: combined[combined._dataset_split.eq(split)].drop(columns="_dataset_split").reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    missing = pd.concat([
        pd.DataFrame({
            "truss": truss,
            "split": split,
            "feature": FEATURES,
            "missing_rate": [frames[split][feature].isna().mean() for feature in FEATURES],
        })
        for split in frames
    ], ignore_index=True)
    bundle = SplitBundle(
        frames["train"], frames["val"],
        ROOT / "processed_data" / truss / "test_candidates.parquet",
        base_bundle.spec,
        test_loader=lambda: frames["test"],
    )
    bundle.spec.validate_features(FEATURES)
    return bundle, missing


def run_one(truss: str, run_root: Path, config: dict) -> None:
    bundle, missing = load_fixed_bundle(truss)
    out = run_root / truss
    out.mkdir(parents=True, exist_ok=False)
    missing.to_csv(out / "feature_missing_rates.csv", index=False)
    write_json(out / "fixed_feature_manifest.json", {
        "feature_set": "fixed_17",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "sensor_window_definition": {
            "indoor_temperature": "EI__TI previous 7-day mean",
            "indoor_humidity": "EI__HI previous 7-day mean",
            "indoor_co2": "EI__CI previous 7-day mean",
            "instant_radiation": "EI__IS latest value in previous 1 day",
            "cumulative_radiation": "EI__CR latest value in previous 1 day",
        },
        "previous_growth_definition": "shift(1) within user_id/sample_num/crop_sn ordered by feature_date",
        "imputation": "adapter preprocessing fitted on Train only; numeric median applied to Validation/Test",
    })
    runner = CommonRegressionRunner(bundle, {"fixed_17": FEATURES}, [], config, out)
    runner.feature_sets = {"fixed_17": FEATURES}
    runner.selected_feature_set = "fixed_17"
    runner.features = FEATURES
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION
    runner._write_manifest(selected_feature_set="fixed_17")
    runner.tune()
    runner.validate()
    runner.freeze()
    runner.fit_final()
    runner.test_once()


def build_summary(run_root: Path) -> None:
    validation = []
    tests = []
    for truss in TRUSSES:
        table = pd.read_csv(run_root / truss / "validation/model_metrics.csv")
        table.insert(0, "truss", truss)
        validation.append(table)
        manifest = json.loads((run_root / truss / "run_manifest.json").read_text())
        selected = manifest["selected_learning_model"]
        tests.extend([
            {"truss": truss, "model": selected, **manifest["final"]},
            {"truss": truss, "model": "persistence", **manifest["persistence"]},
        ])
    summary = run_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.concat(validation, ignore_index=True).to_csv(summary / "validation_six_model_metrics.csv", index=False)
    pd.DataFrame(tests).to_csv(summary / "final_test_metrics.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(summary / "fixed_17_features.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/model_suite.yaml").read_text())
    run_root = ROOT / "artifacts" / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    for truss in TRUSSES:
        print(f"START {truss}", flush=True)
        run_one(truss, run_root, config)
        print(f"COMPLETE {truss}", flush=True)
    build_summary(run_root)
    print(run_root)


if __name__ == "__main__":
    main()
