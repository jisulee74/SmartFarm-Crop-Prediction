from __future__ import annotations
import os
import subprocess


def gpu_preflight():
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        probe = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
        float((probe @ probe).sum().cpu())
        del probe
        torch.cuda.empty_cache()
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).splitlines()[0]
        name, free, total, utilization = [x.strip() for x in query.split(",")]
        free, total, utilization = map(float, (free, total, utilization))
        workers = (
            4
            if free / total >= 0.65 and utilization < 80
            else 2
            if free / total >= 0.35
            else 1
        )
        workers = int(os.getenv("AX_GPU_WORKERS", workers))
        if workers not in (1, 2, 4):
            raise ValueError("AX_GPU_WORKERS must be 1, 2, or 4")
        return {
            "name": name,
            "free_mib": free,
            "total_mib": total,
            "utilization_percent": utilization,
            "gpu_workers": workers,
            "precision": "bf16-mixed",
        }
    except Exception as exc:
        raise RuntimeError(
            f"GPU/CUDA preflight failed: {type(exc).__name__}: {exc}"
        ) from exc


def cpu_workers(gpu_workers):
    cpus = os.cpu_count() or 1
    value = max(1, min(2, (cpus - 4 * gpu_workers) // 24))
    return int(os.getenv("AX_CPU_WORKERS", value))
