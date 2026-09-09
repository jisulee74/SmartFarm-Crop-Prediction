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
from dataset import load_dataset
from prepare_splits import TRUSSES


# Precommitted common columns: identical names and order for all three trusses.
FEATURES = [
    "current_fruit_count",
    "fruit_count_delta",
    "fruit_count_roll2_std",
    "fruit_count_roll2_min",
    "fruit_count_roll3_min",
    "fruit_count_roll4_std",
    "days_since_previous",
    "days_since_first_measurement",
    "days_since_first_positive",
    "history_count",
    "week_of_year_sin",
    "week_of_year_cos",
]


def run_one(truss: str, run_root: Path, config: dict) -> None:
    bundle, _, _, _ = load_dataset(truss)
    bundle.spec.validate_features(FEATURES)
    out = run_root / truss
    out.mkdir(parents=True, exist_ok=False)

    missing = pd.concat([
        pd.DataFrame({
            "truss": truss,
            "split": split,
            "feature": FEATURES,
            "missing_rate": [frame[feature].isna().mean() for feature in FEATURES],
        })
        for split, frame in (("train", bundle.train), ("validation", bundle.validation))
    ], ignore_index=True)
    missing.to_csv(out / "feature_missing_rates_pretest.csv", index=False)
    write_json(out / "fixed_feature_manifest.json", {
        "feature_set": "fixed_12_common_history",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "same_columns_for_all_trusses": True,
        "selection_protocol": "features fixed before tuning; tune on Train CV; select model on Validation; evaluate locked historical holdout once",
        "test_role": "locked_historical_holdout_previously_used_by_other_experiments",
        "imputation": "model adapter preprocessing fitted on Train only and applied to Validation/Test",
    })

    runner = CommonRegressionRunner(bundle, {"fixed_12_common_history": FEATURES}, [], config, out)
    runner.feature_sets = {"fixed_12_common_history": FEATURES}
    runner.selected_feature_set = "fixed_12_common_history"
    runner.features = FEATURES
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION
    runner._write_manifest(selected_feature_set="fixed_12_common_history")
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
        final = {"truss": truss, "model": selected, **manifest["final"]}
        persistence = {"truss": truss, "model": "persistence", **manifest["persistence"]}
        tests.extend([final, persistence])

    summary = run_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.concat(validation, ignore_index=True).to_csv(summary / "validation_six_model_metrics.csv", index=False)
    test_table = pd.DataFrame(tests)
    test_table.to_csv(summary / "final_test_metrics.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(summary / "fixed_12_common_history_features.csv", index=False)

    comparisons = []
    for truss in TRUSSES:
        pair = test_table[test_table.truss.eq(truss)].set_index("model")
        selected = next(model for model in pair.index if model != "persistence")
        model_row, base = pair.loc[selected], pair.loc["persistence"]
        improvements = {
            "rmse_improved": model_row.rmse < base.rmse,
            "mae_improved": model_row.mae < base.mae,
            "r2_improved": model_row.r2 > base.r2,
            "ccc_improved": model_row.ccc > base.ccc,
        }
        comparisons.append({
            "truss": truss,
            "selected_model": selected,
            **improvements,
            "improved_metric_count": sum(improvements.values()),
            "all_four_improved": all(improvements.values()),
        })
    pd.DataFrame(comparisons).to_csv(summary / "persistence_improvement_check.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="fixed_12_common_history_v1")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/model_suite.yaml").read_text())
    run_root = ROOT / "artifacts" / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "precommitted_protocol.json", {
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "same_columns_for_all_trusses": True,
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
