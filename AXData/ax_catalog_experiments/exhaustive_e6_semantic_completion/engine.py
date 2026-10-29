from __future__ import annotations

import json
import os
from pathlib import Path

from ..exhaustive.engine import TargetCampaign
from ..exhaustive.io import atomic_json
from .combinations import missing_semantic_combinations


class CompletionTargetCampaign(TargetCampaign):
    def evaluate(self, combination, model, params, *args, **kwargs):
        bounded_params = dict(params)
        if model == "catboost":
            # CatBoost otherwise reserves most currently free B200 memory for
            # each process.  Four small tabular fits can safely share the GPU
            # with the separately running OCR workload under this cap.
            bounded_params.setdefault(
                "gpu_ram_part", float(os.getenv("AX_CATBOOST_GPU_RAM_PART", "0.08"))
            )
        return super().evaluate(
            combination, model, bounded_params, *args, **kwargs
        )


def _summary_path(root: Path, target: str, model: str, combination_id: str):
    return Path(root) / "summaries" / target / model / f"{combination_id}.json"


def run_lane(root: Path, target: str, model: str):
    root = Path(root)
    eligibility = json.loads((root / "e6_target_eligibility.json").read_text())[target]
    campaign = CompletionTargetCampaign(root, target)
    results = []
    for combination in missing_semantic_combinations():
        if (root / "STOP").exists():
            break
        path = _summary_path(root, target, model, combination["Combination_ID"])
        if not eligibility["eligible"]:
            result = {
                "Target_ID": target,
                "Combination_ID": combination["Combination_ID"],
                "Variable_Groups": combination["Groups"],
                "Model_Type": model,
                "Status": "ineligible",
                "Reason": "semantic_e6_no_feature_survived_all_cv_fit_folds",
                "Source": "semantic_e6_completion_v1",
            }
        else:
            result = campaign.tune_and_validate(combination, model)
            result["Source"] = "semantic_e6_completion_v1"
        atomic_json(path, result)
        results.append(result)
    return {"target": target, "model": model, "completed": len(results)}
