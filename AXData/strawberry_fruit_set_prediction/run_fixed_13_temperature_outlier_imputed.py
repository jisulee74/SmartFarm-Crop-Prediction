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
from run_fixed_13_common_history_temperature import (
    FEATURES,
    TEMPERATURE_FEATURE,
    build_summary,
)
from run_fixed_17_features import load_fixed_bundle


# Greenhouse indoor-temperature validity rule fixed before model fitting.
# Zero is treated as a missing/default sensor value; values above 50 C are
# physically implausible for the target crop and include the observed
# 1,000--4,800 scale sensor errors.
TEMPERATURE_LOWER_EXCLUSIVE = 0.0
TEMPERATURE_UPPER_INCLUSIVE = 50.0


def clean_temperature_outliers(bundle):
    frames = {
        "train": bundle.train.copy(),
        "validation": bundle.validation.copy(),
        "test": bundle.load_test().copy(),
    }
    train_temperature = pd.to_numeric(frames["train"][TEMPERATURE_FEATURE], errors="coerce")
    train_outlier = train_temperature.notna() & ~train_temperature.gt(
        TEMPERATURE_LOWER_EXCLUSIVE
    ) | train_temperature.gt(TEMPERATURE_UPPER_INCLUSIVE)
    valid_train = train_temperature[~train_outlier & train_temperature.notna()]
    if valid_train.empty:
        raise RuntimeError("No valid Train temperature remains after outlier filtering")
    train_median = float(valid_train.median())

    audit_rows = []
    for split, frame in frames.items():
        values = pd.to_numeric(frame[TEMPERATURE_FEATURE], errors="coerce")
        outlier = values.notna() & (
            values.le(TEMPERATURE_LOWER_EXCLUSIVE)
            | values.gt(TEMPERATURE_UPPER_INCLUSIVE)
        )
        original_missing = values.isna()
        frame.loc[outlier, TEMPERATURE_FEATURE] = train_median
        audit_rows.append({
            "split": split,
            "row_count": len(frame),
            "original_missing_count": int(original_missing.sum()),
            "original_missing_rate": float(original_missing.mean()),
            "temperature_outlier_count": int(outlier.sum()),
            "temperature_outlier_rate": float(outlier.mean()),
            "train_median_replacement": train_median,
            "minimum_outlier_value": float(values[outlier].min()) if outlier.any() else None,
            "maximum_outlier_value": float(values[outlier].max()) if outlier.any() else None,
        })

    # Existing missing values deliberately remain missing so the original
    # model-specific missing-value paths remain unchanged. Only newly detected
    # temperature outliers are replaced here.
    bundle.train = frames["train"]
    bundle.validation = frames["validation"]
    bundle.test_loader = lambda: frames["test"].copy()
    return bundle, pd.DataFrame(audit_rows), train_median


def run_one(truss: str, run_root: Path, config: dict) -> None:
    bundle, missing = load_fixed_bundle(truss)
    bundle.spec.validate_features(FEATURES)
    bundle, audit, train_median = clean_temperature_outliers(bundle)

    out = run_root / truss
    out.mkdir(parents=True, exist_ok=False)
    missing[missing.feature.isin(FEATURES)].to_csv(out / "feature_missing_rates_before_outlier_treatment.csv", index=False)
    audit.to_csv(out / "temperature_outlier_imputation_audit.csv", index=False)
    write_json(out / "fixed_feature_manifest.json", {
        "feature_set": "fixed_13_common_history_temperature_outlier_imputed",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "same_columns_for_all_trusses": True,
        "temperature_definition": "EI__TI previous 7-day mean before feature_date",
        "temperature_validity_rule": {
            "lower_bound_exclusive_celsius": TEMPERATURE_LOWER_EXCLUSIVE,
            "upper_bound_inclusive_celsius": TEMPERATURE_UPPER_INCLUSIVE,
            "outlier_replacement": "median of valid Train temperature for the same truss",
            "train_median": train_median,
        },
        "selection_protocol": "same as fixed_13_common_history_temperature_v1",
        "test_role": "locked_historical_holdout_previously_used_by_other_experiments",
        "missing_imputation": "unchanged from original experiment; only temperature outliers are pre-replaced",
    })

    runner = CommonRegressionRunner(bundle, {"fixed_13_temperature_outlier_imputed": FEATURES}, [], config, out)
    runner.feature_sets = {"fixed_13_temperature_outlier_imputed": FEATURES}
    runner.selected_feature_set = "fixed_13_temperature_outlier_imputed"
    runner.features = FEATURES
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION
    runner._write_manifest(selected_feature_set=runner.selected_feature_set)
    runner.tune()
    runner.validate()
    runner.freeze()
    runner.fit_final()
    runner.test_once()


def add_comparison(run_root: Path) -> None:
    current = pd.read_csv(run_root / "summary/final_test_metrics.csv")
    base = pd.read_csv(ROOT / "artifacts/fixed_13_common_history_temperature_v1/summary/final_test_metrics.csv")
    merged = current.merge(base, on=["truss", "model"], suffixes=("_outlier_treated", "_original"))
    for metric in ("rmse", "mae", "r2", "ccc"):
        merged[f"{metric}_delta_treated_minus_original"] = (
            merged[f"{metric}_outlier_treated"] - merged[f"{metric}_original"]
        )
    merged.to_csv(run_root / "summary/comparison_with_original_fixed_13.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="fixed_13_temperature_outlier_imputed_v1")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/model_suite.yaml").read_text())
    run_root = ROOT / "artifacts" / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "precommitted_protocol.json", {
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "temperature_validity_rule": f"{TEMPERATURE_LOWER_EXCLUSIVE} < temperature <= {TEMPERATURE_UPPER_INCLUSIVE} Celsius",
        "replacement": "per-truss valid-Train median",
        "everything_else": "identical to fixed_13_common_history_temperature_v1",
        "primary_selection_metric": "validation_rmse",
        "test_role": "locked_historical_holdout_previously_used_by_other_experiments",
    })
    for truss in TRUSSES:
        print(f"START {truss}", flush=True)
        run_one(truss, run_root, config)
        print(f"COMPLETE {truss}", flush=True)
    build_summary(run_root)
    add_comparison(run_root)
    print(run_root)


if __name__ == "__main__":
    main()
