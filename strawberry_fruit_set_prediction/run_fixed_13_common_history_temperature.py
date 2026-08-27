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
from run_fixed_12_common_history import FEATURES as HISTORY_FEATURES
from run_fixed_17_features import load_fixed_bundle


TEMPERATURE_FEATURE = "sensor_EI__TI_7d_mean"
FEATURES = [*HISTORY_FEATURES, TEMPERATURE_FEATURE]


def run_one(truss: str, run_root: Path, config: dict) -> None:
    bundle, missing = load_fixed_bundle(truss)
    bundle.spec.validate_features(FEATURES)
    out = run_root / truss
    out.mkdir(parents=True, exist_ok=False)

    missing[missing.feature.isin(FEATURES)].to_csv(out / "feature_missing_rates.csv", index=False)
    write_json(out / "fixed_feature_manifest.json", {
        "feature_set": "fixed_13_common_history_temperature",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "same_columns_for_all_trusses": True,
        "temperature_definition": "EI__TI previous 7-day mean before feature_date",
        "selection_protocol": "features fixed before tuning; tune on Train CV; select model on Validation RMSE; evaluate locked historical holdout once",
        "test_role": "locked_historical_holdout_previously_used_by_other_experiments",
        "imputation": "numeric median fitted on Train only; no missing-indicator column added",
    })

    runner = CommonRegressionRunner(bundle, {"fixed_13_common_history_temperature": FEATURES}, [], config, out)
    runner.feature_sets = {"fixed_13_common_history_temperature": FEATURES}
    runner.selected_feature_set = "fixed_13_common_history_temperature"
    runner.features = FEATURES
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION
    runner._write_manifest(selected_feature_set="fixed_13_common_history_temperature")
    runner.tune()
    runner.validate()
    runner.freeze()
    runner.fit_final()
    runner.test_once()


def build_summary(run_root: Path) -> None:
    validation, tests, comparisons = [], [], []
    for truss in TRUSSES:
        table = pd.read_csv(run_root / truss / "validation/model_metrics.csv")
        table.insert(0, "truss", truss)
        validation.append(table)
        manifest = json.loads((run_root / truss / "run_manifest.json").read_text())
        selected = manifest["selected_learning_model"]
        final = {"truss": truss, "model": selected, **manifest["final"]}
        baseline = {"truss": truss, "model": "persistence", **manifest["persistence"]}
        tests.extend([final, baseline])
        improved = {
            "rmse_improved": final["rmse"] < baseline["rmse"],
            "mae_improved": final["mae"] < baseline["mae"],
            "r2_improved": final["r2"] > baseline["r2"],
            "ccc_improved": final["ccc"] > baseline["ccc"],
        }
        comparisons.append({
            "truss": truss,
            "selected_model": selected,
            **improved,
            "improved_metric_count": sum(improved.values()),
            "all_four_improved": all(improved.values()),
        })

    summary = run_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.concat(validation, ignore_index=True).to_csv(summary / "validation_six_model_metrics.csv", index=False)
    pd.DataFrame(tests).to_csv(summary / "final_test_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(summary / "persistence_improvement_check.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(summary / "fixed_13_features.csv", index=False)

    base_path = ROOT / "artifacts/fixed_12_common_history_v1/summary/final_test_metrics.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        current = pd.DataFrame(tests)
        selected_current = current[current.model.ne("persistence")]
        selected_base = base[base.model.ne("persistence")]
        merged = selected_current.merge(selected_base, on="truss", suffixes=("_13", "_12"))
        for metric in ("rmse", "mae", "r2", "ccc"):
            merged[f"{metric}_delta_13_minus_12"] = merged[f"{metric}_13"] - merged[f"{metric}_12"]
        merged.to_csv(summary / "comparison_with_fixed_12.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="fixed_13_common_history_temperature_v1")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/model_suite.yaml").read_text())
    run_root = ROOT / "artifacts" / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "precommitted_protocol.json", {
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "same_columns_for_all_trusses": True,
        "added_to_fixed_12": TEMPERATURE_FEATURE,
        "model_candidates": ["poisson", "random_forest", "catboost", "mlp", "tabm"],
        "baseline": "persistence",
        "primary_selection_metric": "validation_rmse",
        "test_role": "locked_historical_holdout_previously_used_by_other_experiments",
    })
    for truss in TRUSSES:
        print(f"START {truss}", flush=True)
        run_one(truss, run_root, config)
        print(f"COMPLETE {truss}", flush=True)
    build_summary(run_root)
    print(run_root)


if __name__ == "__main__":
    main()
