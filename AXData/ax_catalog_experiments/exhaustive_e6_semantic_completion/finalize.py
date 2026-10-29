from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..exhaustive.combinations import all_combinations
from ..exhaustive.config import MODELS, SEEDS, TARGETS
from ..exhaustive.io import atomic_json
from ..exhaustive.selection import cached_selection
from .combinations import (
    e1_e2_without_e6_combinations,
    missing_semantic_combinations,
    original_v1_combinations,
    semantic_v2_combinations,
)
from .config import DEADLINE_ROOT, SEMANTIC_V2_ROOT, V1_ROOT
from .engine import CompletionTargetCampaign


def _source_for(combination: dict, root: Path):
    groups = set(combination["Groups"])
    if "E6" in groups and ({"E1", "E2"} & groups):
        return Path(root), "semantic_e6_completion_v1"
    if "E6" in groups:
        return SEMANTIC_V2_ROOT, "semantic_e6_v2"
    if {"E1", "E2"} & groups:
        return DEADLINE_ROOT, "e1_e2_fixed_configuration_v1"
    return V1_ROOT, "original_v1"


def _normalise(value: dict, source: str):
    row = dict(value)
    row["Source"] = source
    aliases = {
        "Validation_RMSE": "Validation_RMSE_Mean",
        "Validation_MAE": "Validation_MAE_Mean",
        "Validation_R2": "Validation_R2_Mean",
        "Validation_CCC": "Validation_CCC_Mean",
    }
    for destination, source_key in aliases.items():
        if row.get(destination) is None and row.get(source_key) is not None:
            row[destination] = row[source_key]
    if row.get("Best_Epoch") is None and row.get("Fixed_Epochs") is not None:
        row["Best_Epoch"] = row["Fixed_Epochs"]
    return row


def build_rankings(root: Path):
    root = Path(root)
    expected_sets = [
        original_v1_combinations(),
        e1_e2_without_e6_combinations(),
        semantic_v2_combinations(),
        missing_semantic_combinations(),
    ]
    if sum(map(len, expected_sets)) != 127:
        raise AssertionError("Composite semantic ranking must cover 127 combinations")
    all_rows = []
    winners = []
    for target in TARGETS:
        for model in MODELS:
            values = []
            for combination in all_combinations():
                source_root, source = _source_for(combination, root)
                path = (
                    source_root
                    / "summaries"
                    / target
                    / model
                    / f"{combination['Combination_ID']}.json"
                )
                if not path.exists():
                    raise FileNotFoundError(f"Required validation summary missing: {path}")
                value = _normalise(json.loads(path.read_text()), source)
                value["Variable_Groups"] = combination["Groups"]
                values.append(value)
            successes = [
                value
                for value in values
                if value.get("Status") == "success"
                and value.get("Validation_RMSE") is not None
                and value.get("Validation_MAE") is not None
            ]
            successes.sort(
                key=lambda value: (
                    float(value["Validation_RMSE"]),
                    float(value["Validation_MAE"]),
                    int(value.get("Feature_Count", 10**9)),
                    value["Combination_ID"],
                )
            )
            rank_by_id = {
                value["Combination_ID"]: rank
                for rank, value in enumerate(successes, 1)
            }
            for value in values:
                value["Validation_Rank"] = rank_by_id.get(value["Combination_ID"])
                destination = root / "composite_summaries" / target / model
                atomic_json(destination / f"{value['Combination_ID']}.json", value)
                all_rows.append(
                    {
                        "Target_ID": target,
                        "Model_Type": model,
                        "Combination_ID": value["Combination_ID"],
                        "Variable_Groups": "|".join(value["Variable_Groups"]),
                        "Status": value.get("Status"),
                        "Reason": value.get("Reason"),
                        "Source": value["Source"],
                        "Validation_Rank": value["Validation_Rank"],
                        "Validation_RMSE": value.get("Validation_RMSE"),
                        "Validation_MAE": value.get("Validation_MAE"),
                        "Validation_R2": value.get("Validation_R2"),
                        "Validation_CCC": value.get("Validation_CCC"),
                        "Feature_Count": value.get("Feature_Count"),
                        "Seed_Count": value.get("Seed_Count"),
                    }
                )
            if not successes:
                raise RuntimeError(f"No successful validation result: {target}/{model}")
            winner = dict(successes[0])
            atomic_json(root / "validation_winners" / target / f"{model}.json", winner)
            winners.append(winner)
    rankings = pd.DataFrame(all_rows)
    rankings.to_csv(root / "validation_rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "Target_ID": winner["Target_ID"],
                "Model_Type": winner["Model_Type"],
                "Combination_ID": winner["Combination_ID"],
                "Variable_Groups": "|".join(winner["Variable_Groups"]),
                "Source": winner["Source"],
                "Validation_RMSE": winner["Validation_RMSE"],
                "Validation_MAE": winner["Validation_MAE"],
                "Validation_R2": winner.get("Validation_R2"),
                "Validation_CCC": winner.get("Validation_CCC"),
                "Feature_Count": winner.get("Feature_Count"),
            }
            for winner in winners
        ]
    ).to_csv(root / "validation_winners.csv", index=False, encoding="utf-8-sig")
    return {"rows": len(all_rows), "winners": len(winners)}


def evaluate_winner(root: Path, target: str, model: str):
    root = Path(root)
    winner_path = root / "validation_winners" / target / f"{model}.json"
    winner = json.loads(winner_path.read_text())
    combination = next(
        value
        for value in all_combinations()
        if value["Combination_ID"] == winner["Combination_ID"]
    )
    campaign = CompletionTargetCampaign(root, target)
    numeric, categorical = campaign.candidates(combination)
    frozen = cached_selection(
        root,
        target,
        combination["Combination_ID"],
        campaign.train,
        numeric,
        categorical,
        "validation",
        None,
    )
    # Test is opened only after the validation winner and train-only feature
    # selection have both been frozen.
    test = campaign.load("test")
    combined = pd.concat([campaign.train, campaign.validation], ignore_index=True)
    runs = [
        campaign.evaluate(
            combination,
            model,
            winner["Params"],
            seed,
            "test",
            combined,
            test,
            epochs=winner.get("Best_Epoch"),
            frozen_selection=frozen,
        )
        for seed in SEEDS
    ]
    successful = [run for run in runs if run.get("Status") == "success"]
    record = {
        "Target_ID": target,
        "Model_Type": model,
        "winner": winner,
        "winner_source": winner["Source"],
        "candidate_count": 127,
        "test_runs": [run["Run_ID"] for run in runs],
        "test_accessed_after_selection": True,
        "frozen_features": frozen.selected,
        "Status": "success" if len(successful) == len(SEEDS) else "failed",
    }
    if successful:
        for metric in ("rmse", "mae", "r2", "ccc"):
            values = [float(run["Metrics"]["bounded"][metric]) for run in successful]
            record[f"Test_{metric.upper()}_Mean"] = float(np.mean(values))
            record[f"Test_{metric.upper()}_Std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
    atomic_json(root / "selections" / target / f"{model}.json", record)
    return record


def consolidate_test_outputs(root: Path):
    root = Path(root)
    records = []
    prediction_frames = []
    for target in TARGETS:
        for model in MODELS:
            path = root / "selections" / target / f"{model}.json"
            if not path.exists():
                raise FileNotFoundError(f"Final test selection missing: {path}")
            value = json.loads(path.read_text())
            records.append(
                {
                    "Target_ID": target,
                    "Model_Type": model,
                    "Combination_ID": value["winner"]["Combination_ID"],
                    "Variable_Groups": "|".join(value["winner"]["Variable_Groups"]),
                    "Winner_Source": value["winner_source"],
                    "Status": value["Status"],
                    "Test_RMSE_Mean": value.get("Test_RMSE_Mean"),
                    "Test_RMSE_Std": value.get("Test_RMSE_Std"),
                    "Test_MAE_Mean": value.get("Test_MAE_Mean"),
                    "Test_MAE_Std": value.get("Test_MAE_Std"),
                    "Test_R2_Mean": value.get("Test_R2_Mean"),
                    "Test_R2_Std": value.get("Test_R2_Std"),
                    "Test_CCC_Mean": value.get("Test_CCC_Mean"),
                    "Test_CCC_Std": value.get("Test_CCC_Std"),
                    "Feature_Count": len(value["frozen_features"]),
                }
            )
            for run_id in value["test_runs"]:
                result_path = root / "runs" / run_id / "result.json"
                prediction_path = root / "runs" / run_id / "predictions.parquet"
                if not result_path.exists() or not prediction_path.exists():
                    continue
                result = json.loads(result_path.read_text())
                if result.get("Status") != "success":
                    continue
                frame = pd.read_parquet(prediction_path)
                frame["Target_ID"] = target
                frame["Model_Type"] = model
                frame["Combination_ID"] = value["winner"]["Combination_ID"]
                frame["Seed"] = result["Seed"]
                prediction_frames.append(frame)
    pd.DataFrame(records).to_csv(
        root / "test_results.csv", index=False, encoding="utf-8-sig"
    )
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_parquet(root / "final_test_predictions.parquet", index=False)
        keys = [
            "Target_ID",
            "Model_Type",
            "Combination_ID",
            "row_id",
            "facility_id",
            "crop_sn",
            "sample_num",
            "feature_date",
            "target_date",
            "target",
        ]
        averaged = (
            predictions.groupby(keys, dropna=False, as_index=False)
            .agg(
                prediction_raw=("prediction_raw", "mean"),
                prediction_bounded=("prediction_bounded", "mean"),
                prediction_bounded_std=("prediction_bounded", "std"),
                seed_count=("Seed", "nunique"),
            )
        )
        averaged.to_parquet(root / "winner_predictions.parquet", index=False)
    return {"test_results": len(records), "prediction_runs": len(prediction_frames)}
