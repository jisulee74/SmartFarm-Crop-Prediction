from __future__ import annotations
import concurrent.futures
import fcntl
import json
import multiprocessing
import signal
import time
from pathlib import Path
from ..core import digest, file_hash
from .combinations import write_catalog
from .config import (
    CPU_MODELS,
    GPU_MODELS,
    MODELS,
    SEEDS,
    SELECTION_POLICY,
    TARGETS,
    VERSION,
)
from .engine import TargetCampaign, grid, run_lane
from .io import atomic_json, frozen_json
from .prepare import prepare
import pandas as pd
from .resources import cpu_workers, gpu_preflight
from .snapshot import preflight_snapshot


def _implementation_hash():
    directory = Path(__file__).parent
    files = sorted(directory.glob("*.py")) + [
        directory / "variable_group_definition.csv"
    ]
    return digest({p.name: file_hash(p) for p in files})


def initialize(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    definition = Path(__file__).parent / "variable_group_definition.csv"
    mapping = pd.read_csv(definition)
    expected = {f"E{i}" for i in range(1, 8)}
    if set(mapping["변수군 코드"]) != expected:
        raise ValueError("Frozen variable-group definition must contain E1-E7 exactly")
    if not mapping["수집 구분"].eq("기존 DB 확인").all():
        raise ValueError("E1-E7 definition contains non-existing/new variables")
    gpu = gpu_preflight()
    cpu = cpu_workers(gpu["gpu_workers"])
    resources = {"gpu": gpu, "cpu_workers": cpu}
    preflight_snapshot(root)
    prepare(root)
    combinations = write_catalog(root / "combination_catalog.csv")
    manifest = {
        "version": VERSION,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "combination_count": len(combinations),
        "selection_policy": SELECTION_POLICY,
        "grids": {m: grid(m) for m in MODELS},
        "implementation_hash": _implementation_hash(),
        "snapshot_manifest_hash": file_hash(root / "data_snapshot/manifest.json"),
        "prepared_manifest_hash": file_hash(root / "prepared_manifest.json"),
        "variable_group_definition_hash": file_hash(
            Path(__file__).parent / "variable_group_definition.csv"
        ),
    }
    frozen_json(root / "campaign_manifest.json", manifest)
    atomic_json(root / "resource_plan.json", resources)
    return {**manifest, "resources": resources}


def run(root: Path):
    root = Path(root)
    manifest = initialize(root)
    stop = root / "STOP"
    stop.unlink(missing_ok=True)
    lock_path = root / "campaign.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def request_stop(*_):
            stop.touch()
            atomic_json(
                root / "scheduler.json",
                {"Status": "stopping", "Updated_At": time.time()},
            )

        previous = {
            sig: signal.signal(sig, request_stop)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        cpu_lanes = [(t, m) for t in TARGETS for m in MODELS if m in CPU_MODELS]
        gpu_lanes = [(t, m) for t in TARGETS for m in MODELS if m in GPU_MODELS]
        futures = {}
        completed = []
        try:
            with (
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=manifest["resources"]["cpu_workers"]
                ) as cp,
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=manifest["resources"]["gpu"]["gpu_workers"],
                    mp_context=multiprocessing.get_context("spawn"),
                    max_tasks_per_child=1,
                ) as gp,
            ):
                for t, m in cpu_lanes:
                    futures[cp.submit(run_lane, root, t, m)] = (t, m, "cpu")
                for t, m in gpu_lanes:
                    futures[gp.submit(run_lane, root, t, m)] = (t, m, "gpu")
                for future in concurrent.futures.as_completed(futures):
                    t, m, lane = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "target": t,
                            "model": m,
                            "lane": lane,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    completed.append(result)
                    atomic_json(
                        root / "scheduler.json",
                        {
                            "Status": "stopping" if stop.exists() else "running",
                            "Completed_Lanes": completed,
                            "Total_Lanes": len(futures),
                            "Updated_At": time.time(),
                        },
                    )
            if stop.exists():
                atomic_json(
                    root / "scheduler.json",
                    {
                        "Status": "paused",
                        "Completed_Lanes": completed,
                        "Updated_At": time.time(),
                    },
                )
                return "paused"
            finals = []
            for target in TARGETS:
                campaign = TargetCampaign(root, target)
                for model in MODELS:
                    finals.append(
                        {
                            "target": target,
                            "model": model,
                            **campaign.finalize_model(model),
                        }
                    )
            atomic_json(
                root / "scheduler.json",
                {
                    "Status": "finished",
                    "Completed_Lanes": completed,
                    "Finals": finals,
                    "Updated_At": time.time(),
                },
            )
            return "finished"
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def status(root):
    root = Path(root)
    summaries = list((root / "summaries").glob("*/*/*.json"))
    runs = list((root / "runs").glob("*/result.json"))
    states = {}
    for p in runs:
        value = json.loads(p.read_text())
        states[value.get("Status", "unknown")] = (
            states.get(value.get("Status", "unknown"), 0) + 1
        )
    return {
        "Combination_Count": 127,
        "Target_Model_Combination_Count": 127 * 8 * 6,
        "Completed_Target_Model_Combinations": len(summaries),
        "Persisted_Runs": len(runs),
        "Run_Status": states,
        "Scheduler": json.loads((root / "scheduler.json").read_text())
        if (root / "scheduler.json").exists()
        else None,
    }
