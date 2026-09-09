#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
AXDATA = ROOT.parent
sys.path.insert(0, str(AXDATA))

from common_regression.contracts import SplitBundle, TargetSpec
from common_regression.evaluation import write_json
from common_regression.metrics import regression_metrics
from common_regression.runner import CommonRegressionRunner, Stage
from common_regression.selection import select_feature_set_with_fixed_configs
from extract import rebuild_cache
from pipeline import (CONTROL_CODES, ENV_SPECS, HISTORY_FEATURES, add_simple_sensor_features,
                      prepare_growth, split_and_filter_features)


def _save_audits(output: Path, audits: dict[str, pd.DataFrame], quality: pd.DataFrame,
                 candidates: pd.DataFrame) -> None:
    audit_dir = output / "data_audit"; audit_dir.mkdir(parents=True)
    for name, frame in audits.items():
        frame.to_csv(audit_dir / f"{name}.csv", index=False)
    quality.to_csv(audit_dir / "sensor_coverage_by_row.csv", index=False)
    rows = []
    raw_truss = pd.read_parquet(ROOT / "data/raw/growth.parquet")
    raw_truss["growth_measure_code"] = pd.to_numeric(raw_truss.growth_measure_code, errors="coerce")
    raw_truss["truss_number"] = raw_truss.growth_measure_code - 10000233
    for truss, frame in raw_truss.groupby("truss_number", dropna=False):
        rows.append({"truss_number": truss, "rows": len(frame),
                     "facilities": frame.facility_id.nunique(), "crops": frame.crop_sn.nunique(),
                     "flower_mean": pd.to_numeric(frame.flower_count, errors="coerce").mean(),
                     "flower_median": pd.to_numeric(frame.flower_count, errors="coerce").median()})
    pd.DataFrame(rows).to_csv(audit_dir / "truss_distribution.csv", index=False)
    raw_env = pd.read_parquet(ROOT / "data/raw/environment.parquet")
    dictionary = raw_env.groupby(["sect_code", "fatr_code"], dropna=False).agg(
        rows=("facility_id", "size"), facilities=("facility_id", "nunique")
    ).reset_index()
    dictionary["planned_feature"] = dictionary.apply(
        lambda r: next((name for name, (sect, code, *_rest) in ENV_SPECS.items()
                       if r.sect_code == sect and r.fatr_code == code),
                      next((name for name, predicate in CONTROL_CODES.items()
                            if r.sect_code == "CR" and predicate(str(r.fatr_code))), "excluded")), axis=1)
    dictionary.to_csv(audit_dir / "sensor_control_code_dictionary.csv", index=False)
    write_json(audit_dir / "audit_summary.json", {
        "candidate_rows_before_sensor_eligibility": len(candidates),
        "audits": {name: len(frame) for name, frame in audits.items()},
    })


def _eligible_sensor_facilities(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = list(ENV_SPECS)
    complete = data[env].notna().all(axis=1)
    audit = data.assign(_complete=complete).groupby(["facility_id", "crop_sn"]).agg(
        rows=("row_id", "size"), complete_environment_rows=("_complete", "sum"),
        complete_environment_ratio=("_complete", "mean"),
    ).reset_index()
    audit["eligible"] = audit.complete_environment_ratio.ge(.50)
    selected = pd.MultiIndex.from_frame(audit.loc[audit.eligible, ["facility_id", "crop_sn"]])
    index = pd.MultiIndex.from_frame(data[["facility_id", "crop_sn"]])
    return data[index.isin(selected)].copy().reset_index(drop=True), audit


def _target_spec(feature_sets: dict[str, list[str]]) -> TargetSpec:
    blocks = {
        "history": tuple(HISTORY_FEATURES), "environment": tuple(ENV_SPECS),
        "control": tuple(CONTROL_CODES),
    }
    exclusions = ("next_total_flower_count", "target_date", "row_id", "facility_id", "crop_sn", "sample_num",
                  "cropping_date", "cropping_end_date", "examin_date", "feature_date")
    return TargetSpec(
        "tomato_total_flower_count", "next_total_flower_count", "current_total_flower_count",
        "feature_date", "target_date", tuple(["facility_id", "crop_sn", "sample_num"]),
        tuple(["facility_id", "crop_sn", "sample_num", "target_date"]), exclusions, .95, blocks)


def _split_summary(frames: dict[str, pd.DataFrame], partition: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for split, frame in frames.items():
        rows.append({"split": split, "facility_count": frame.facility_id.nunique(),
                     "facilities": ",".join(partition[split]), "rows": len(frame),
                     "target_date_min": frame.target_date.min(), "target_date_max": frame.target_date.max(),
                     "crop_count": frame.crop_sn.nunique()})
    return pd.DataFrame(rows)


def _grouped_metrics(output: Path, test: pd.DataFrame) -> None:
    final = pd.read_parquet(output / "model_selection/final_test_predictions.parquet")
    persistence = pd.read_parquet(output / "model_selection/persistence_test_predictions.parquet")
    keys = ["facility_id", "crop_sn", "sample_num", "target_date"]
    merged = test.merge(final[keys + ["prediction"]], on=keys, validate="one_to_one")
    merged = merged.merge(persistence[keys + ["prediction"]].rename(columns={"prediction": "persistence"}),
                          on=keys, validate="one_to_one")
    merged = merged.rename(columns={"prediction": "model_prediction"})
    merged["max_truss_band"] = pd.cut(merged.current_max_truss_number, [0, 10, 20, 34],
                                      labels=["1-10", "11-20", "21-34"], include_lowest=True)
    rows = []
    for group_name, columns in (("facility", ["facility_id"]), ("crop", ["facility_id", "crop_sn"]),
                                ("max_truss_band", ["max_truss_band"])):
        grouper = columns[0] if len(columns) == 1 else columns
        for key, frame in merged.groupby(grouper, observed=True):
            key = (key,) if not isinstance(key, tuple) else key
            for model, prediction in (("final", "model_prediction"), ("persistence", "persistence")):
                rows.append({"group_type": group_name, "group": "|".join(map(str, key)), "model": model,
                             "rows": len(frame), **regression_metrics(frame.next_total_flower_count, frame[prediction])})
    pd.DataFrame(rows).to_csv(output / "grouped_test_metrics.csv", index=False)
    merged.to_parquet(output / "test_predictions_with_context.parquet", index=False)
    _plot_timeseries(merged, output / "test_timeseries.png")


def _plot_timeseries(data: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    counts = data.groupby("facility_id").size().sort_values(ascending=False)
    selected = list(counts.head(min(3, len(counts))).index)
    fig, axes = plt.subplots(len(selected), 1, figsize=(16, 4.5 * len(selected)), squeeze=False)
    for ax, facility in zip(axes[:, 0], selected):
        frame = data[data.facility_id.eq(facility)].sort_values("target_date")
        ax.plot(frame.target_date, frame.next_total_flower_count, "o-", label="Actual", color="black")
        ax.plot(frame.target_date, frame.model_prediction, "o-", label="Final model")
        ax.plot(frame.target_date, frame.persistence, "o--", label="Persistence")
        ax.set_title(facility, fontsize=14); ax.set_ylabel("Total flower count", fontsize=12)
        ax.tick_params(labelsize=11); ax.grid(alpha=.2); ax.legend(fontsize=11)
    axes[-1, 0].set_xlabel("Target survey date", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _write_report(output: Path, runner: CommonRegressionRunner, selected: str, result: dict, frames: dict[str, pd.DataFrame]) -> None:
    validation = runner.validation_table[["model", "rmse", "mae", "r2", "ccc"]].copy()
    test = pd.DataFrame([
        {"model": runner.selected_model, **result["final"]},
        {"model": "persistence", **result["persistence"]},
    ])
    lines = [
        "# 토마토 다음 조사 전체 꽃수 예측 결과", "",
        f"- 선택 변수군: `{selected}`",
        f"- 선택 학습모델: `{runner.selected_model}`",
        f"- 시설 분할 행 수: Train {len(frames['train'])}, Validation {len(frames['validation'])}, Test {len(frames['test'])}",
        "", "## Validation", "", validation.to_markdown(index=False, floatfmt=".4f"),
        "", "## 최종 Test", "", test.to_markdown(index=False, floatfmt=".4f"), "",
        f"- Persistence 대비 Test RMSE 개선 여부: {result['baseline_beaten_on_test']}",
    ]
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_id: str, rebuild: bool, value_column: str | None) -> Path:
    raw_dir = ROOT / "data/raw"
    extraction = rebuild_cache(raw_dir, value_column) if rebuild else {"cache_reused": True}
    growth_path, env_path = raw_dir / "growth.parquet", raw_dir / "environment.parquet"
    if not growth_path.exists() or not env_path.exists():
        raise FileNotFoundError("Raw cache is absent. Configure FARMSTOM_DB_* and use --rebuild-cache")
    output = ROOT / "artifacts" / run_id
    output.mkdir(parents=True, exist_ok=False)
    candidates, audits = prepare_growth(pd.read_parquet(growth_path))
    enriched, quality = add_simple_sensor_features(candidates, pd.read_parquet(env_path))
    _save_audits(output, audits, quality, candidates)
    enriched, facility_audit = _eligible_sensor_facilities(enriched)
    facility_audit.to_csv(output / "data_audit/sensor_eligibility_by_facility.csv", index=False)
    frames, feature_sets, prep = split_and_filter_features(enriched)
    processed = output / "processed_data"; processed.mkdir()
    for split, frame in frames.items(): frame.to_parquet(processed / f"{split}.parquet", index=False)
    _split_summary(frames, prep["partition"]).to_csv(output / "facility_split_summary.csv", index=False)
    write_json(output / "preprocessing_manifest.json", {**extraction, **prep, "feature_sets": feature_sets})

    spec = _target_spec(feature_sets)
    bundle = SplitBundle(frames["train"], frames["validation"], processed / "test.parquet", spec)
    config = yaml.safe_load((ROOT / "config/model_suite.yaml").read_text())
    selection_sets = {("individual_history" if name == "history" else name): features
                      for name, features in feature_sets.items()}
    selected, trials, summary = select_feature_set_with_fixed_configs(
        bundle.train, spec.target, spec.target_date, selection_sets, [],
        config["feature_selection_baseline_params"], config["feature_selection_seeds"])
    selected = "history" if selected == "individual_history" else selected
    trials["feature_set"] = trials.feature_set.replace({"individual_history": "history"})
    summary["feature_set"] = summary.feature_set.replace({"individual_history": "history"})
    selection_dir = output / "feature_selection"; selection_dir.mkdir()
    trials.to_csv(selection_dir / "internal_cv_trials.csv", index=False)
    summary.to_csv(selection_dir / "feature_set_summary.csv", index=False)
    write_json(selection_dir / "feature_sets.json", {"sets": feature_sets, "selected": selected})

    runner = CommonRegressionRunner(bundle, {"fixed": feature_sets[selected]}, [], config, output / "model_selection")
    runner.feature_sets = feature_sets; runner.selected_feature_set = selected; runner.features = feature_sets[selected]
    runner.stage = Stage.INTERNAL_FEATURE_SELECTION
    runner._write_manifest(selected_feature_set=selected)
    runner.tune(); runner.validate(); runner.freeze(); runner.fit_final(); result = runner.test_once()
    _grouped_metrics(output, bundle.load_test())
    write_json(output / "final_run_manifest.json", {
        "selected_feature_set": selected, "selected_features": feature_sets[selected],
        "selected_learning_model": runner.selected_model, "test": result,
        "environment_eligibility_rule": "facility complete TI/HI/CI rows >= 50%",
        "test_access_count": 1,
    })
    _write_report(output, runner, selected, result, frames)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="total_flower_count_v1")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--growth-value-column")
    args = parser.parse_args()
    print(run(args.run_id, args.rebuild_cache, args.growth_value_column))

