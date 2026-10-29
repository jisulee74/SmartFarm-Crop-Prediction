from __future__ import annotations

import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

import pandas as pd

from ..core import digest, file_hash
from ..exhaustive.config import CPU_MODELS, GPU_MODELS, MODELS, SEEDS, SELECTION_POLICY, TARGETS
from ..exhaustive.engine import grid
from ..exhaustive.io import atomic_json, frozen_json
from .combinations import missing_semantic_combinations
from .config import CAMPAIGN_ID, DEFAULT_OUTPUT, PACKAGE_ROOT, VERSION
from .engine import run_lane
from .finalize import build_rankings, consolidate_test_outputs, evaluate_winner
from .prepare import prepare


def _implementation_hash():
    package_root = PACKAGE_ROOT.parent
    files = (
        sorted(PACKAGE_ROOT.glob("*.py"))
        + sorted((package_root / "exhaustive").glob("*.py"))
        + sorted((package_root / "stability").glob("*.py"))
        + sorted((package_root.parent / "common_regression").glob("*.py"))
        + [package_root / "core.py"]
    )
    return digest(
        {
            str(path.relative_to(package_root.parent)): file_hash(path)
            for path in files
        }
    )


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _gpu_workers():
    return max(1, int(os.getenv("AX_SEMANTIC_COMPLETION_GPU_WORKERS", "4")))


def _cpu_workers():
    return max(1, int(os.getenv("AX_SEMANTIC_COMPLETION_CPU_WORKERS", "8")))


def initialize(root: Path = DEFAULT_OUTPUT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prepared = prepare(root)
    combinations = missing_semantic_combinations()
    pd.DataFrame(
        [
            {
                "Combination_ID": row["Combination_ID"],
                "Variable_Groups": "|".join(row["Groups"]),
                "Group_Count": row["Group_Count"],
                "Label": row["Label"],
            }
            for row in combinations
        ]
    ).to_csv(root / "combination_catalog.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "version": VERSION,
        "purpose": "Complete the 48 E1/E2-containing semantic E6 combinations and produce a 127-combination final ranking.",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "new_combination_count": len(combinations),
        "full_ranking_combination_count": 127,
        "selection_policy": SELECTION_POLICY,
        "grids": {model: grid(model) for model in MODELS},
        "implementation_hash": _implementation_hash(),
        "prepared_manifest_hash": file_hash(root / "prepared_manifest.json"),
        "prepared_dataset_hashes": {
            target: spec["dataset_hashes"] for target, spec in prepared["targets"].items()
        },
        "resources": {
            "gpu_workers": _gpu_workers(),
            "cpu_workers": _cpu_workers(),
        },
    }
    frozen_json(root / "campaign_manifest.json", manifest)
    atomic_json(root / "resource_plan.json", manifest["resources"])
    return manifest


def _pool_tasks(executor, root: Path, models):
    return {
        executor.submit(run_lane, root, target, model): (target, model)
        for target in TARGETS
        for model in MODELS
        if model in models
    }


def _run_validation(root: Path, manifest: dict, stop: Path):
    completed = []
    context = multiprocessing.get_context("spawn")
    with (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=manifest["resources"]["cpu_workers"],
            mp_context=context,
            initializer=_worker_init,
        ) as cpu_pool,
        concurrent.futures.ProcessPoolExecutor(
            max_workers=manifest["resources"]["gpu_workers"],
            mp_context=context,
            max_tasks_per_child=1,
            initializer=_worker_init,
        ) as gpu_pool,
    ):
        futures = {}
        futures.update(_pool_tasks(cpu_pool, root, CPU_MODELS))
        futures.update(_pool_tasks(gpu_pool, root, GPU_MODELS))
        for future in concurrent.futures.as_completed(futures):
            target, model = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = {
                    "target": target,
                    "model": model,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            completed.append(value)
            atomic_json(
                root / "scheduler.json",
                {
                    "Status": "stopping" if stop.exists() else "validation_running",
                    "Completed_Lanes": completed,
                    "Total_Lanes": len(futures),
                    "Updated_At": time.time(),
                },
            )
    return completed


def _run_tests(root: Path, manifest: dict):
    tasks = [(target, model) for target in TARGETS for model in MODELS]
    cpu_tasks = [task for task in tasks if task[1] in CPU_MODELS]
    gpu_tasks = [task for task in tasks if task[1] in GPU_MODELS]
    records = []
    context = multiprocessing.get_context("spawn")
    with (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=manifest["resources"]["cpu_workers"],
            mp_context=context,
            initializer=_worker_init,
        ) as cpu_pool,
        concurrent.futures.ProcessPoolExecutor(
            max_workers=manifest["resources"]["gpu_workers"],
            mp_context=context,
            max_tasks_per_child=1,
            initializer=_worker_init,
        ) as gpu_pool,
    ):
        futures = {
            cpu_pool.submit(evaluate_winner, root, *task): task for task in cpu_tasks
        }
        futures.update(
            {gpu_pool.submit(evaluate_winner, root, *task): task for task in gpu_tasks}
        )
        for future in concurrent.futures.as_completed(futures):
            target, model = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = {
                    "Target_ID": target,
                    "Model_Type": model,
                    "Status": "failed",
                    "Error": f"{type(exc).__name__}: {exc}",
                }
            records.append(value)
            atomic_json(
                root / "test_scheduler.json",
                {
                    "Status": "running",
                    "Completed": len(records),
                    "Total": len(tasks),
                    "Records": records,
                    "Updated_At": time.time(),
                },
            )
    atomic_json(
        root / "test_scheduler.json",
        {
            "Status": "finished",
            "Completed": len(records),
            "Total": len(tasks),
            "Records": records,
            "Updated_At": time.time(),
        },
    )
    return records


def run(root: Path = DEFAULT_OUTPUT):
    root = Path(root)
    manifest = initialize(root)
    stop = root / "STOP"
    stop.unlink(missing_ok=True)
    with (root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def request_stop(*_):
            stop.touch()
            atomic_json(
                root / "scheduler.json",
                {"Status": "stopping", "Updated_At": time.time()},
            )

        previous = {
            sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            completed = _run_validation(root, manifest, stop)
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
            expected = 48 * len(TARGETS) * len(MODELS)
            actual = len(list((root / "summaries").glob("*/*/*.json")))
            if actual != expected:
                raise RuntimeError(f"Validation summaries incomplete: {actual}/{expected}")
            ranking = build_rankings(root)
            atomic_json(
                root / "scheduler.json",
                {
                    "Status": "test_running",
                    "Completed_Lanes": completed,
                    "Ranking": ranking,
                    "Updated_At": time.time(),
                },
            )
            tests = _run_tests(root, manifest)
            consolidated = consolidate_test_outputs(root)
            final = {
                "Status": "finished",
                "Completed_Lanes": completed,
                "Ranking": ranking,
                "Test_Count": len(tests),
                "Consolidated": consolidated,
                "Updated_At": time.time(),
            }
            atomic_json(root / "scheduler.json", final)
            return "finished"
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def status(root: Path = DEFAULT_OUTPUT):
    root = Path(root)
    summaries = list((root / "summaries").glob("*/*/*.json"))
    runs = list((root / "runs").glob("*/result.json"))
    states = {}
    for path in summaries:
        try:
            value = json.loads(path.read_text())
        except Exception:
            continue
        key = value.get("Status", "unknown")
        states[key] = states.get(key, 0) + 1
    total = 48 * len(TARGETS) * len(MODELS)
    return {
        "Campaign_ID": CAMPAIGN_ID,
        "Completed_Target_Model_Combinations": len(summaries),
        "Total_Target_Model_Combinations": total,
        "Percent": round(len(summaries) / total * 100, 2),
        "Summary_Statuses": states,
        "Persisted_Fits": len(runs),
        "Scheduler": json.loads((root / "scheduler.json").read_text())
        if (root / "scheduler.json").exists()
        else None,
    }
