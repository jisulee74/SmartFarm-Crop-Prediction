#!/usr/bin/env python3
"""Requested crop-window experiment with facility holdout and TFT."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from common_regression.contracts import SplitBundle
from common_regression.evaluation import write_json
from common_regression.metrics import regression_metrics
from common_regression.runner import CommonRegressionRunner, Stage
from common_regression.selection import select_feature_set_with_fixed_configs
from pipeline import add_simple_sensor_features, prepare_growth, split_and_filter_features
from run_pipeline import (_grouped_metrics, _save_audits, _split_summary, _target_spec,
                          _write_report, _eligible_sensor_facilities)
from tft_candidate import fit_predict as fit_predict_tft
from tft_candidate import tune as tune_tft
from tft_candidate import validate as validate_tft


def crop_policy(raw: pd.DataFrame):
    data = raw.copy()
    data["examin_date"] = pd.to_datetime(data.examin_date)
    facility = data.facility_id.astype(str).eq("PF_0025108_01")
    remove_205 = facility & data.crop_sn.eq(205) & data.examin_date.gt("2025-05-31")
    remove_207 = facility & data.crop_sn.eq(207)
    audit = {
        "facility_id": "PF_0025108_01",
        "crop_205_rule": "examin_date <= 2025-05-31",
        "crop_206_rule": "all rows retained",
        "crop_207_rule": "all rows excluded",
        "removed_raw_rows_crop_205": int(remove_205.sum()),
        "removed_raw_rows_crop_207": int(remove_207.sum()),
    }
    return data.loc[~(remove_205 | remove_207)].copy(), audit


def choose_model(table: pd.DataFrame) -> str:
    learning = table[table.is_learning_model].copy()
    tied = learning[learning.rmse.le(learning.rmse.min() * 1.01)]
    tied = tied[tied.mae.le(tied.mae.min() * 1.01)]
    return str(tied.sort_values(["inference_latency_ms", "model_size_bytes", "model"], kind="stable").iloc[0].model)


def save_tft_test(output: Path, bundle: SplitBundle, prediction: np.ndarray):
    test = bundle.load_test(); spec = bundle.spec; keys = list(spec.row_key)
    final = regression_metrics(test[spec.target], prediction)
    persistence = regression_metrics(test[spec.target], test[spec.current_value])
    final_table = test[keys + [spec.target, spec.current_value]].copy(); final_table["prediction"] = prediction
    base_table = test[keys + [spec.target, spec.current_value]].copy(); base_table["prediction"] = test[spec.current_value]
    final_table.to_parquet(output / "model_selection/final_test_predictions.parquet", index=False)
    base_table.to_parquet(output / "model_selection/persistence_test_predictions.parquet", index=False)
    write_json(output / "model_selection/final_test_metrics.json", final)
    write_json(output / "model_selection/persistence_test_metrics.json", persistence)
    write_json(output / "model_selection/test_access_log.json", {"access_count": 1, "reason": "configuration frozen"})
    return {"final": final, "persistence": persistence, "baseline_beaten_on_test": final["rmse"] < persistence["rmse"]}


def run(run_id: str = "total_flower_count_v1_sample_key_retry3") -> Path:
    output = ROOT / "artifacts" / run_id
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    raw, policy = crop_policy(pd.read_parquet(ROOT / "data/raw/growth.parquet"))
    candidates, audits = prepare_growth(raw)
    enriched, quality = add_simple_sensor_features(candidates, pd.read_parquet(ROOT / "data/raw/environment.parquet"))
    _save_audits(output, audits, quality, candidates)
    _, sensor_audit = _eligible_sensor_facilities(enriched)
    sensor_audit.to_csv(output / "data_audit/sensor_eligibility_by_facility.csv", index=False)

    # Sensor availability is audited, not used as a population exclusion, so PF_0025108 is retained.
    frames, feature_sets, prep = split_and_filter_features(enriched)
    processed = output / "processed_data"; processed.mkdir()
    for name, frame in frames.items(): frame.to_parquet(processed / f"{name}.parquet", index=False)
    _split_summary(frames, prep["partition"]).to_csv(output / "facility_split_summary.csv", index=False)
    write_json(output / "preprocessing_manifest.json", {
        **prep, "feature_sets": feature_sets, "crop_policy": policy,
        "sensor_population_policy": "quality_audit_only; missing values use Train medians",
    })

    spec = _target_spec(feature_sets)
    bundle = SplitBundle(frames["train"], frames["validation"], processed / "test.parquet", spec)
    config = yaml.safe_load((ROOT / "config/model_suite.yaml").read_text())
    aliases = {("individual_history" if n == "history" else n): f for n, f in feature_sets.items()}
    selected, trials, summary = select_feature_set_with_fixed_configs(
        bundle.train, spec.target, spec.target_date, aliases, [],
        config["feature_selection_baseline_params"], config["feature_selection_seeds"])
    selected = "history" if selected == "individual_history" else selected
    trials["feature_set"] = trials.feature_set.replace({"individual_history": "history"})
    summary["feature_set"] = summary.feature_set.replace({"individual_history": "history"})
    fs_dir = output / "feature_selection"; fs_dir.mkdir()
    trials.to_csv(fs_dir / "internal_cv_trials.csv", index=False)
    summary.to_csv(fs_dir / "feature_set_summary.csv", index=False)
    write_json(fs_dir / "feature_sets.json", {"sets": feature_sets, "selected": selected})

    model_dir = output / "model_selection"
    runner = CommonRegressionRunner(bundle, {"fixed": feature_sets[selected]}, [], config, model_dir)
    runner.feature_sets = feature_sets; runner.selected_feature_set = selected; runner.features = feature_sets[selected]
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION; runner._write_manifest(selected_feature_set=selected)
    runner.tune(); runner.validate()
    params, epochs = tune_tft(bundle.train, spec.target, spec.target_date, feature_sets[selected],
                              list(spec.entity_key), spec.current_value,
                              config["feature_selection_seeds"], model_dir)
    tft_metric = validate_tft(bundle.train, bundle.validation, spec.target, feature_sets[selected],
                              list(spec.entity_key), spec.current_value, params, epochs,
                              config["validation_seeds"])
    runner.validation_table = pd.concat([runner.validation_table, pd.DataFrame([tft_metric])], ignore_index=True)
    runner.validation_table.to_csv(model_dir / "validation/model_metrics.csv", index=False)
    winner = choose_model(runner.validation_table)
    if winner == "tft":
        runner.selected_model = winner
        prediction = fit_predict_tft(bundle.train, bundle.validation, bundle.load_test(), spec.target,
                                     feature_sets[selected], list(spec.entity_key), spec.current_value,
                                     params, epochs, config["validation_seeds"])
        result = save_tft_test(output, bundle, prediction)
    else:
        runner.freeze(); runner.fit_final(); result = runner.test_once()
    _grouped_metrics(output, bundle.load_test())
    write_json(output / "final_run_manifest.json", {
        "selected_feature_set": selected, "selected_features": feature_sets[selected],
        "selected_learning_model": runner.selected_model, "candidate_models":
        ["poisson", "random_forest", "catboost", "mlp", "tabm", "tft"],
        "crop_policy": policy, "sensor_population_policy": "quality_audit_only",
        "tft_params": params, "tft_fixed_epochs": epochs, "test": result, "test_access_count": 1,
    })
    _write_report(output, runner, selected, result, frames)
    return output


if __name__ == "__main__":
    print(run())
