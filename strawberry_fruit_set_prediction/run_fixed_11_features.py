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

from common_regression.evaluation import write_json
from common_regression.runner import CommonRegressionRunner, Stage
from prepare_splits import TRUSSES
from run_fixed_17_features import FEATURES as FEATURES_17, load_fixed_bundle

EXCLUDED = [
    "sensor_EI__IS_1d_last",
    "sensor_EI__CR_1d_last",
    "previous_fruit_cluster_num",
    "previous_first_flower_num",
    "previous_second_flower_num",
    "previous_third_flower_num",
]
FEATURES = [feature for feature in FEATURES_17 if feature not in EXCLUDED]


def run_one(truss: str, run_root: Path, config: dict) -> None:
    bundle, missing = load_fixed_bundle(truss)
    out = run_root / truss
    out.mkdir(parents=True, exist_ok=False)
    missing[missing.feature.isin(FEATURES)].to_csv(out / "feature_missing_rates.csv", index=False)
    write_json(out / "fixed_feature_manifest.json", {
        "feature_set": "fixed_11_without_solar_and_previous_growth",
        "source_feature_count": len(FEATURES_17),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "excluded_features": EXCLUDED,
        "sensor_window_definition": {
            "indoor_temperature": "EI__TI previous 7-day mean",
            "indoor_humidity": "EI__HI previous 7-day mean",
            "indoor_co2": "EI__CI previous 7-day mean",
        },
        "imputation": "adapter preprocessing fitted on Train only; numeric median applied to Validation/Test",
    })
    runner = CommonRegressionRunner(bundle, {"fixed_11": FEATURES}, [], config, out)
    runner.feature_sets = {"fixed_11": FEATURES}
    runner.selected_feature_set = "fixed_11"
    runner.features = FEATURES
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION
    runner._write_manifest(selected_feature_set="fixed_11")
    runner.tune()
    runner.validate()
    runner.freeze()
    runner.fit_final()
    runner.test_once()


def build_summary(run_root: Path) -> None:
    validation, tests = [], []
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
    pd.DataFrame({"feature": FEATURES}).to_csv(summary / "fixed_11_features.csv", index=False)
    pd.DataFrame({"excluded_feature": EXCLUDED}).to_csv(summary / "excluded_6_features.csv", index=False)


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
