from __future__ import annotations

import concurrent.futures
import fcntl
import json
import multiprocessing
import signal
import time
from pathlib import Path

import pandas as pd

from ..core import digest, file_hash
from ..exhaustive.config import CPU_MODELS, GPU_MODELS, MODELS, SEEDS, SELECTION_POLICY, TARGETS
from ..exhaustive.engine import grid
from ..exhaustive.io import atomic_json, frozen_json
from ..exhaustive.resources import cpu_workers, gpu_preflight
from .combinations import composite_combinations, semantic_combinations
from .config import CAMPAIGN_ID, PACKAGE_ROOT, VERSION, V1_ROOT
from .engine import finalize_composite_model, run_lane
from .prepare import prepare


def _implementation_hash():
    ax_package = PACKAGE_ROOT.parent
    runtime_files = (
        sorted(PACKAGE_ROOT.glob("*.py"))
        + sorted((ax_package / "exhaustive").glob("*.py"))
        + sorted((ax_package / "stability").glob("*.py"))
        + sorted((ax_package.parent / "common_regression").glob("*.py"))
        + [ax_package / "core.py", PACKAGE_ROOT / "sfkr_col_info_cr.csv"]
    )
    return digest({str(path.relative_to(ax_package.parent)): file_hash(path) for path in runtime_files})


def _catalog(path: Path, rows, source: str):
    pd.DataFrame(
        [{"Combination_ID": r["Combination_ID"], "Variable_Groups": "|".join(r["Groups"]), "Group_Count": r["Group_Count"], "Label": r["Label"], "Source": source} for r in rows]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _worker_init():
    # The parent records STOP and lets each lane finish its current persisted
    # combination. Children ignore terminal SIGINT and stop at that boundary.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def initialize(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prepared = prepare(root, V1_ROOT)
    semantic = semantic_combinations()
    composite = composite_combinations()
    _catalog(root / "combination_catalog.csv", semantic, "semantic_e6_v2")
    _catalog(root / "composite_combination_catalog.csv", composite, "mixed")
    gpu = gpu_preflight()
    resources = {"gpu": gpu, "cpu_workers": cpu_workers(gpu["gpu_workers"])}
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "version": VERSION,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "new_combination_count": len(semantic),
        "composite_combination_count": len(composite),
        "selection_policy": SELECTION_POLICY,
        "grids": {m: grid(m) for m in MODELS},
        "implementation_hash": _implementation_hash(),
        "v1_root": str(V1_ROOT.resolve()),
        "v1_snapshot_manifest_hash": prepared["v1_snapshot_manifest_hash"],
        "v1_prepared_row_hashes": {t: s["row_hashes"] for t, s in prepared["targets"].items()},
        "v2_prepared_dataset_hashes": {t: s["dataset_hashes"] for t, s in prepared["targets"].items()},
        "v2_prepared_manifest_hash": file_hash(root / "prepared_manifest.json"),
        "sfkr_col_info_reference_hash": prepared["sfkr_col_info_reference_hash"],
        "mapping_hash": prepared["mapping_hash"],
        "semantic_feature_catalog_hash": prepared["semantic_feature_catalog_hash"],
    }
    frozen_json(root / "campaign_manifest.json", manifest)
    atomic_json(root / "resource_plan.json", resources)
    return {**manifest, "resources": resources}


def run(root: Path):
    root = Path(root)
    manifest = initialize(root)
    stop = root / "STOP"
    stop.unlink(missing_ok=True)
    with (root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def request_stop(*_):
            stop.touch()
            atomic_json(root / "scheduler.json", {"Status": "stopping", "Updated_At": time.time()})

        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        cpu_lanes = [(t, m) for t in TARGETS for m in MODELS if m in CPU_MODELS]
        gpu_lanes = [(t, m) for t in TARGETS for m in MODELS if m in GPU_MODELS]
        futures, completed = {}, []
        try:
            with (
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=manifest["resources"]["cpu_workers"], initializer=_worker_init
                ) as cp,
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=manifest["resources"]["gpu"]["gpu_workers"],
                    mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1,
                    initializer=_worker_init,
                ) as gp,
            ):
                for target, model in cpu_lanes:
                    futures[cp.submit(run_lane, root, target, model)] = (target, model, "cpu")
                for target, model in gpu_lanes:
                    futures[gp.submit(run_lane, root, target, model)] = (target, model, "gpu")
                for future in concurrent.futures.as_completed(futures):
                    target, model, lane = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"target": target, "model": model, "lane": lane, "error": f"{type(exc).__name__}: {exc}"}
                    completed.append(result)
                    atomic_json(root / "scheduler.json", {"Status": "stopping" if stop.exists() else "running", "Completed_Lanes": completed, "Total_Lanes": len(futures), "Updated_At": time.time()})
            if stop.exists():
                atomic_json(root / "scheduler.json", {"Status": "paused", "Completed_Lanes": completed, "Updated_At": time.time()})
                return "paused"
            finals = [
                {"target": target, "model": model, **finalize_composite_model(root, target, model)}
                for target in TARGETS for model in MODELS
            ]
            atomic_json(root / "scheduler.json", {"Status": "finished", "Completed_Lanes": completed, "Finals": finals, "Updated_At": time.time()})
            return "finished"
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def status(root: Path):
    root = Path(root)
    summaries = list((root / "summaries").glob("*/*/*.json"))
    runs = list((root / "runs").glob("*/result.json"))
    states = {}
    for path in runs:
        value = json.loads(path.read_text())
        states[value.get("Status", "unknown")] = states.get(value.get("Status", "unknown"), 0) + 1
    return {
        "Campaign_ID": CAMPAIGN_ID,
        "New_Combination_Count": 16,
        "Composite_Combination_Count": 31,
        "Target_Model_Combination_Count": 16 * 8 * 6,
        "Completed_Target_Model_Combinations": len(summaries),
        "Persisted_Runs": len(runs),
        "Run_Status": states,
        "Scheduler": json.loads((root / "scheduler.json").read_text()) if (root / "scheduler.json").exists() else None,
    }
