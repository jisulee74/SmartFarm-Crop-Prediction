from __future__ import annotations

import json
from pathlib import Path

from ..exhaustive.engine import TargetCampaign
from ..exhaustive.io import atomic_json
from .combinations import backfill_combinations, reused_v1_combinations
from .config import V1_ROOT


def _summary_path(root: Path, target: str, model: str, combination_id: str):
    return Path(root) / "summaries" / target / model / f"{combination_id}.json"


def run_lane(root: Path, target: str, model: str):
    root = Path(root)
    campaign = TargetCampaign(root, target)
    results = []
    for combination in backfill_combinations():
        if (root / "STOP").exists():
            break
        result = campaign.tune_and_validate(combination, model)
        result["Source"] = "e1_e2_backfill"
        atomic_json(_summary_path(root, target, model, combination["Combination_ID"]), result)
        results.append(result)
    return {"target": target, "model": model, "completed": len(results)}


def attach_reused_v1_summaries(root: Path, v1_root: Path = V1_ROOT):
    root, v1_root = Path(root), Path(v1_root)
    copied = 0
    for target_dir in sorted((v1_root / "summaries").glob("*")):
        if not target_dir.is_dir():
            continue
        for model_dir in sorted(target_dir.glob("*")):
            if not model_dir.is_dir():
                continue
            for combination in reused_v1_combinations():
                source = model_dir / f"{combination['Combination_ID']}.json"
                if not source.exists():
                    raise FileNotFoundError(f"Reusable v1 summary missing: {source}")
                value = json.loads(source.read_text())
                value["Source"] = "reused_from_v1"
                atomic_json(_summary_path(root, target_dir.name, model_dir.name, combination["Combination_ID"]), value)
                copied += 1
    if copied != 31 * 8 * 6:
        raise AssertionError(f"Expected 1488 reused summaries, got {copied}")
    return copied
