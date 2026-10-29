from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..exhaustive.config import SEEDS
from ..exhaustive.engine import TargetCampaign
from ..exhaustive.io import atomic_json
from ..exhaustive.selection import cached_selection
from .combinations import composite_combinations, reused_v1_combinations, semantic_combinations
from .config import V1_ROOT


class SemanticTargetCampaign(TargetCampaign):
    pass


def _summary_path(root: Path, target: str, model: str, combination_id: str):
    return Path(root) / "summaries" / target / model / f"{combination_id}.json"


def run_lane(root: Path, target: str, model: str):
    root = Path(root)
    eligibility = json.loads((root / "e6_target_eligibility.json").read_text())[target]
    campaign = SemanticTargetCampaign(root, target)
    results = []
    for combination in semantic_combinations():
        if (root / "STOP").exists():
            break
        if not eligibility["eligible"]:
            result = {
                "Target_ID": target,
                "Combination_ID": combination["Combination_ID"],
                "Variable_Groups": combination["Groups"],
                "Model_Type": model,
                "Status": "ineligible",
                "Reason": "semantic_e6_no_feature_survived_all_cv_fit_folds",
                "Source": "semantic_e6_v2",
            }
            atomic_json(_summary_path(root, target, model, combination["Combination_ID"]), result)
        else:
            result = campaign.tune_and_validate(combination, model)
            result["Source"] = "semantic_e6_v2"
            atomic_json(_summary_path(root, target, model, combination["Combination_ID"]), result)
        results.append(result)
    return {"target": target, "model": model, "completed": len(results)}


def build_composite_summaries(root: Path, target: str, model: str, v1_root: Path = V1_ROOT):
    root, v1_root = Path(root), Path(v1_root)
    rows = []
    for combination in reused_v1_combinations():
        path = _summary_path(v1_root, target, model, combination["Combination_ID"])
        if not path.exists():
            raise FileNotFoundError(f"Required reusable v1 summary missing: {path}")
        value = json.loads(path.read_text())
        value["Source"] = "reused_from_v1"
        rows.append(value)
    for combination in semantic_combinations():
        path = _summary_path(root, target, model, combination["Combination_ID"])
        if not path.exists():
            raise FileNotFoundError(f"Semantic v2 summary missing: {path}")
        value = json.loads(path.read_text())
        value["Source"] = "semantic_e6_v2"
        rows.append(value)
    if len(rows) != 31:
        raise AssertionError("Composite summary must contain exactly 31 rows")
    destination = root / "composite_summaries" / target / model
    destination.mkdir(parents=True, exist_ok=True)
    for value in rows:
        atomic_json(destination / f"{value['Combination_ID']}.json", value)
    return rows


def finalize_composite_model(root: Path, target: str, model: str):
    root = Path(root)
    rows = build_composite_summaries(root, target, model)
    successes = [r for r in rows if r.get("Status") == "success"]
    if not successes:
        return {"Target_ID": target, "Model_Type": model, "Status": "failed", "Reason": "no_eligible_combination"}
    winner = min(
        successes,
        key=lambda x: (
            x["Validation_RMSE"], x["Validation_MAE"], x["Feature_Count"], x["Combination_ID"]
        ),
    )
    combination = next(c for c in composite_combinations() if c["Combination_ID"] == winner["Combination_ID"])
    campaign = SemanticTargetCampaign(root, target)
    numeric, categorical = campaign.candidates(combination)
    frozen = cached_selection(
        root, target, combination["Combination_ID"], campaign.train,
        numeric, categorical, "validation", None,
    )
    # Test is intentionally opened only after the composite validation winner is frozen.
    test = campaign.load("test")
    combined = pd.concat([campaign.train, campaign.validation], ignore_index=True)
    runs = [
        campaign.evaluate(
            combination, model, winner["Params"], seed, "test", combined, test,
            epochs=winner.get("Best_Epoch"), frozen_selection=frozen,
        )
        for seed in SEEDS
    ]
    record = {
        "winner": winner,
        "winner_source": winner["Source"],
        "candidate_count": 31,
        "test_runs": [r["Run_ID"] for r in runs],
        "test_accessed_after_selection": True,
        "frozen_features": frozen.selected,
        "Status": "success" if all(r["Status"] == "success" for r in runs) else "failed",
    }
    atomic_json(root / "selections" / target / f"{model}.json", record)
    return record
