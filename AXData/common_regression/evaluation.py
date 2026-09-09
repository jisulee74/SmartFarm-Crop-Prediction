from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_frame(frame: pd.DataFrame, row_key: tuple[str, ...], target: str,
                     predicted: np.ndarray, rounded: bool = False) -> pd.DataFrame:
    result = frame[list(row_key)].copy()
    result["actual"] = frame[target].to_numpy(float)
    result["prediction"] = np.asarray(predicted, float)
    result["residual"] = result.actual - result.prediction
    if rounded:
        result["prediction_nonnegative_integer"] = np.rint(np.maximum(0, result.prediction)).astype(int)
    if result.duplicated(list(row_key)).any() or result[["actual", "prediction"]].isna().any().any():
        raise ValueError("Invalid prediction artifact")
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def evaluate_final_and_persistence(test: pd.DataFrame, row_key: tuple[str, ...], target: str,
                                   current_value: str, prediction: np.ndarray, output: Path,
                                   rounded: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    final = prediction_frame(test, row_key, target, prediction, rounded)
    persistence = prediction_frame(test, row_key, target, test[current_value].to_numpy(float), rounded)
    if not final[list(row_key)].equals(persistence[list(row_key)]):
        raise ValueError("Final model and Persistence row IDs differ")
    final.to_parquet(output / "final_test_predictions.parquet", index=False)
    persistence.to_parquet(output / "persistence_test_predictions.parquet", index=False)
    final_metrics = regression_metrics(final.actual, final.prediction)
    persistence_metrics = regression_metrics(persistence.actual, persistence.prediction)
    write_json(output / "final_test_metrics.json", final_metrics)
    write_json(output / "persistence_test_metrics.json", persistence_metrics)
    return {"final": final_metrics, "persistence": persistence_metrics,
            "baseline_beaten_on_test": final_metrics["rmse"] < persistence_metrics["rmse"]}
