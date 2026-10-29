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
from ..exhaustive.combinations import all_combinations
from ..exhaustive.config import CPU_MODELS, GPU_MODELS
from ..exhaustive.engine import TargetCampaign, grid
from ..exhaustive.io import atomic_json, frozen_json
from ..exhaustive.report import report
from ..exhaustive.resources import cpu_workers, gpu_preflight
from .audit import run_audit
from .combinations import backfill_combinations
from .config import CAMPAIGN_ID, MODELS, PACKAGE_ROOT, SEEDS, SELECTION_POLICY, TARGETS, VERSION, V1_ROOT
from .engine import attach_reused_v1_summaries, run_lane
from .prepare import prepare


def _implementation_hash():
    files = sorted(PACKAGE_ROOT.glob("*.py"))
    return digest({path.name: file_hash(path) for path in files})


def _write_catalogs(root: Path):
    def frame(rows, source):
        return pd.DataFrame([{
            "Combination_ID": row["Combination_ID"],
            "Variable_Groups": "|".join(row["Groups"]),
            "Group_Count": row["Group_Count"],
            "Label": row["Label"],
            "Source": source,
        } for row in rows])
    frame(backfill_combinations(), "e1_e2_backfill").to_csv(root / "backfill_combination_catalog.csv", index=False, encoding="utf-8-sig")
    all_rows = []
    for row in all_combinations():
        source = "e1_e2_backfill" if ({"E1", "E2"} & set(row["Groups"])) else "reused_from_v1"
        all_rows.append((row, source))
    pd.concat([frame([row], source) for row, source in all_rows], ignore_index=True).to_csv(root / "combination_catalog.csv", index=False, encoding="utf-8-sig")


def initialize(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prepared = prepare(root, V1_ROOT)
    audit = run_audit(root)
    _write_catalogs(root)
    gpu = gpu_preflight()
    resources = {"gpu": gpu, "cpu_workers": cpu_workers(gpu["gpu_workers"])}
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "version": VERSION,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "backfill_combination_count": 96,
        "reused_v1_combination_count": 31,
        "composite_combination_count": 127,
        "selection_policy": SELECTION_POLICY,
        "grids": {model: grid(model) for model in MODELS},
        "implementation_hash": _implementation_hash(),
        "v1_root": str(V1_ROOT.resolve()),
        "v1_snapshot_manifest_hash": prepared["v1_snapshot_manifest_hash"],
        "v1_prepared_manifest_hash": prepared["v1_prepared_manifest_hash"],
        "prepared_manifest_hash": file_hash(root / "prepared_manifest.json"),
        "audit_hash": file_hash(root / "e1_e2_preflight_audit.json"),
        "audit_status": audit["Status"],
    }
    frozen_json(root / "campaign_manifest.json", manifest)
    atomic_json(root / "resource_plan.json", resources)
    return {**manifest, "resources": resources}


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


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
        cpu_lanes = [(target, model) for target in TARGETS for model in MODELS if model in CPU_MODELS]
        gpu_lanes = [(target, model) for target in TARGETS for model in MODELS if model in GPU_MODELS]
        futures, completed = {}, []
        try:
            with (
                concurrent.futures.ProcessPoolExecutor(max_workers=manifest["resources"]["cpu_workers"], initializer=_worker_init) as cpu_pool,
                concurrent.futures.ProcessPoolExecutor(max_workers=manifest["resources"]["gpu"]["gpu_workers"], mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1, initializer=_worker_init) as gpu_pool,
            ):
                for target, model in cpu_lanes:
                    futures[cpu_pool.submit(run_lane, root, target, model)] = (target, model, "cpu")
                for target, model in gpu_lanes:
                    futures[gpu_pool.submit(run_lane, root, target, model)] = (target, model, "gpu")
                for future in concurrent.futures.as_completed(futures):
                    target, model, lane = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"target": target, "model": model, "lane": lane, "error": f"{type(exc).__name__}: {exc}"}
                    completed.append(result)
                    atomic_json(root / "scheduler.json", {"Status": "stopping" if stop.exists() else "running", "Completed_Lanes": completed, "Total_Lanes": len(futures), "Updated_At": time.time()})
            if stop.exists():
                atomic_json(root / "scheduler.json", {"Status": "paused", "Completed_Lanes": completed, "Total_Lanes": len(futures), "Updated_At": time.time()})
                return "paused"
            completed_backfill = list((root / "summaries").glob("*/*/*.json"))
            backfill_ids = {row["Combination_ID"] for row in backfill_combinations()}
            completed_backfill = [path for path in completed_backfill if path.stem in backfill_ids]
            expected_backfill = 96 * len(TARGETS) * len(MODELS)
            if len(completed_backfill) != expected_backfill:
                atomic_json(root / "scheduler.json", {
                    "Status": "incomplete",
                    "Completed_Lanes": completed,
                    "Total_Lanes": len(futures),
                    "Completed_Backfill_Summaries": len(completed_backfill),
                    "Expected_Backfill_Summaries": expected_backfill,
                    "Updated_At": time.time(),
                })
                raise RuntimeError(
                    f"Backfill summaries incomplete: {len(completed_backfill)}/{expected_backfill}"
                )
            attach_reused_v1_summaries(root, V1_ROOT)
            finals = []
            for target in TARGETS:
                campaign = TargetCampaign(root, target)
                for model in MODELS:
                    finals.append({"target": target, "model": model, **campaign.finalize_model(model)})
            atomic_json(root / "scheduler.json", {"Status": "finished", "Completed_Lanes": completed, "Total_Lanes": len(futures), "Finals": finals, "Updated_At": time.time()})
            report(root)
            return "finished"
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def status(root: Path):
    root = Path(root)
    summaries = list((root / "summaries").glob("*/*/*.json"))
    backfill_ids = {row["Combination_ID"] for row in backfill_combinations()}
    backfill_summaries = [path for path in summaries if path.stem in backfill_ids]
    runs = list((root / "runs").glob("*/result.json"))
    states = {}
    for path in runs:
        value = json.loads(path.read_text())
        states[value.get("Status", "unknown")] = states.get(value.get("Status", "unknown"), 0) + 1
    return {
        "Campaign_ID": CAMPAIGN_ID,
        "Backfill_Combination_Count": 96,
        "Target_Model_Combination_Count": 96 * 8 * 6,
        "Completed_Target_Model_Combinations": len(backfill_summaries),
        "Completion_Percent": round(len(backfill_summaries) / (96 * 8 * 6) * 100, 2),
        "Persisted_Runs": len(runs),
        "Run_Status": states,
        "Audit": json.loads((root / "e1_e2_preflight_audit.json").read_text()) if (root / "e1_e2_preflight_audit.json").exists() else None,
        "Scheduler": json.loads((root / "scheduler.json").read_text()) if (root / "scheduler.json").exists() else None,
    }
