from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np

from ..exhaustive.config import MODELS, TARGETS, SEEDS
from ..exhaustive.combinations import all_combinations
from ..exhaustive_e1_e2_deadline.runner import _load_selection
from ..exhaustive_e6_semantic_completion.engine import CompletionTargetCampaign
from ..exhaustive_e6_semantic_completion import runner as completion_runner
from ..exhaustive.io import atomic_json
from ..exhaustive_e6_semantic_completion.finalize import consolidate_test_outputs



SOURCE = "semantic_single_seed_screening"


def _normalise(value: dict) -> dict:
    row = dict(value)
    row["Source"] = SOURCE
    aliases = {
        "Validation_RMSE": "Validation_RMSE_Mean",
        "Validation_MAE": "Validation_MAE_Mean",
        "Validation_R2": "Validation_R2_Mean",
        "Validation_CCC": "Validation_CCC_Mean",
    }
    for destination, source in aliases.items():
        if row.get(destination) is None:
            row[destination] = row.get(source)
    if row.get("Best_Epoch") is None:
        row["Best_Epoch"] = row.get("Fixed_Epochs")
    return row


def build_rankings(root: Path) -> dict:
    root = Path(root)
    paths = sorted((root / "summaries").glob("*/*/*.json"))
    expected = 127 * len(TARGETS) * len(MODELS)
    if len(paths) != expected:
        raise RuntimeError(f"Screening summaries incomplete: {len(paths)}/{expected}")

    values: dict[tuple[str, str], list[dict]] = {}
    for path in paths:
        value = _normalise(json.loads(path.read_text(encoding="utf-8")))
        values.setdefault((value["Target_ID"], value["Model_Type"]), []).append(value)

    ranking_rows = []
    winner_rows = []
    for target in TARGETS:
        for model in MODELS:
            candidates = values.get((target, model), [])
            if len(candidates) != 127:
                raise RuntimeError(f"Screening group count mismatch: {target}/{model}={len(candidates)}")
            successes = [
                value
                for value in candidates
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
            if not successes:
                raise RuntimeError(f"No successful screening result: {target}/{model}")
            ranks = {
                value["Combination_ID"]: rank
                for rank, value in enumerate(successes, start=1)
            }
            for value in candidates:
                value["Validation_Rank"] = ranks.get(value["Combination_ID"])
                atomic_json(
                    root
                    / "composite_summaries"
                    / target
                    / model
                    / f"{value['Combination_ID']}.json",
                    value,
                )
                ranking_rows.append(
                    {
                        "Target_ID": target,
                        "Model_Type": model,
                        "Combination_ID": value["Combination_ID"],
                        "Variable_Groups": "|".join(value.get("Variable_Groups", [])),
                        "Status": value.get("Status"),
                        "Reason": value.get("Reason"),
                        "Source": SOURCE,
                        "Validation_Rank": value["Validation_Rank"],
                        "Validation_RMSE": value.get("Validation_RMSE"),
                        "Validation_MAE": value.get("Validation_MAE"),
                        "Validation_R2": value.get("Validation_R2"),
                        "Validation_CCC": value.get("Validation_CCC"),
                        "Feature_Count": value.get("Feature_Count"),
                        "Seed_Count": value.get("Seed_Count"),
                    }
                )
            winner = dict(successes[0])
            atomic_json(root / "validation_winners" / target / f"{model}.json", winner)
            winner_rows.append(
                {
                    "Target_ID": target,
                    "Model_Type": model,
                    "Combination_ID": winner["Combination_ID"],
                    "Variable_Groups": "|".join(winner.get("Variable_Groups", [])),
                    "Source": SOURCE,
                    "Validation_RMSE": winner["Validation_RMSE"],
                    "Validation_MAE": winner["Validation_MAE"],
                    "Validation_R2": winner.get("Validation_R2"),
                    "Validation_CCC": winner.get("Validation_CCC"),
                    "Feature_Count": winner.get("Feature_Count"),
                }
            )

    pd.DataFrame(ranking_rows).to_csv(
        root / "validation_rankings.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(winner_rows).to_csv(
        root / "validation_winners.csv", index=False, encoding="utf-8-sig"
    )
    return {"rows": len(ranking_rows), "winners": len(winner_rows)}


def evaluate_screening_winner(root: Path, target: str, model: str):
    """Freeze the actual screening features; never rerun selection for Test."""
    root = Path(root)
    winner = json.loads((root / "validation_winners" / target / f"{model}.json").read_text())
    combination = next(c for c in all_combinations() if c["Combination_ID"] == winner["Combination_ID"])
    frozen = _load_selection(root, target, combination["Combination_ID"])
    if frozen.selected != winner["Selected_Features"]:
        raise ValueError(f"Screening feature freeze mismatch: {target}/{model}")
    campaign = CompletionTargetCampaign(root, target)
    atomic_json(root / "test_feature_freeze" / target / f"{model}.json", {
        "target": target, "model": model, "combination": combination["Combination_ID"],
        "selected": frozen.selected, "source": "screening_train_only_selection",
        "frozen_before_test_load": True,
    })
    test = campaign.load("test")
    combined = pd.concat([campaign.train, campaign.validation], ignore_index=True)
    runs = [campaign.evaluate(combination, model, winner["Params"], seed, "test",
                              combined, test, epochs=winner.get("Best_Epoch"),
                              frozen_selection=frozen) for seed in SEEDS]
    successful = [r for r in runs if r.get("Status") == "success"]
    record = {
        "Target_ID": target, "Model_Type": model, "winner": winner,
        "winner_source": SOURCE, "candidate_count": 127,
        "test_runs": [r["Run_ID"] for r in runs],
        "test_accessed_after_selection": True, "frozen_features": frozen.selected,
        "feature_selection_source": "screening_train_only_selection",
        "Status": "success" if len(successful) == len(SEEDS) else "failed",
    }
    if successful:
        for metric in ("rmse", "mae", "r2", "ccc"):
            values = [float(r["Metrics"]["bounded"][metric]) for r in successful]
            record[f"Test_{metric.upper()}_Mean"] = float(np.mean(values))
            record[f"Test_{metric.upper()}_Std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    atomic_json(root / "selections" / target / f"{model}.json", record)
    return record


def finalize(root: Path) -> dict:
    root = Path(root)
    ranking = build_rankings(root)
    resources = {
        "cpu_workers": max(1, int(os.getenv("AX_SCREENING_TEST_CPU_WORKERS", "4"))),
        "gpu_workers": max(1, int(os.getenv("AX_SCREENING_TEST_GPU_WORKERS", "2"))),
    }
    atomic_json(
        root / "screening_scheduler.json",
        {
            "Status": "test_running",
            "Method": "single_seed_screening_then_three_seed_test",
            "Ranking": ranking,
            "Resources": resources,
            "Updated_At": time.time(),
        },
    )
    # The submitted callable is pickled by its screening-module name under spawn.
    # This leaves the separately running formal campaign untouched.
    completion_runner.evaluate_winner = evaluate_screening_winner
    tests = completion_runner._run_tests(root, {"resources": resources})
    if len(tests) != len(TARGETS) * len(MODELS) or any(r.get("Status") != "success" for r in tests):
        raise RuntimeError("Screening Test incomplete or failed; publishing is blocked")
    consolidated = consolidate_test_outputs(root)
    result = {
        "Status": "finished",
        "Method": "single_seed_screening_then_three_seed_test",
        "Validation_Seeds": [42],
        "Test_Seeds": [42, 52, 62],
        "Ranking": ranking,
        "Test_Count": len(tests),
        "Consolidated": consolidated,
        "Updated_At": time.time(),
    }
    atomic_json(root / "screening_scheduler.json", result)
    return result
