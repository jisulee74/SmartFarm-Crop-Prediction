"""Audit completed predictions against the frozen cohort and selection rules."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AXData.ax_catalog_experiments.upper_truss_screening.prepare import OUT, digest
from AXData.ax_catalog_experiments.upper_truss_screening.run import MODELS, metrics


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    winners = json.loads((OUT / "frozen_winners.json").read_text())
    records = [json.loads(p.read_text()) for p in (OUT / "runs").glob("*.json")]
    assert len(records) == 60
    assert all(r["status"] == "success" for r in records)
    audited_rows = 0
    for crop, spec in manifest["crops"].items():
        frames = {}
        for stage in ["train", "validation", "test"]:
            path = OUT / crop / f"{stage}.parquet"
            assert digest(path) == spec["hashes"][stage]
            frames[stage] = pd.read_parquet(path)
        for model in MODELS:
            validation = [
                r for r in records
                if r["crop"] == crop and r["model"] == model and r["stage"] == "validation"
            ]
            assert {r["config"] for r in validation} == set(spec["features"])
            selected = min(validation, key=lambda r: (
                r["metrics"]["rmse"], r["metrics"]["mae"], r["features"], r["config"]
            ))
            frozen = next(r for r in winners["per_model"] if r["crop"] == crop and r["model"] == model)
            assert selected == frozen
            tests = [r for r in records if r["crop"] == crop and r["model"] == model and r["stage"] == "test"]
            assert len(tests) == 1
            test = tests[0]
            assert test["config"] == selected["config"]
            assert test["best_epoch"] == selected["best_epoch"]
            assert (OUT / "models" / f'{crop}__{test["config"]}__{model}__test.model').stat().st_size > 0
            for record in validation + tests:
                stage = record["stage"]
                key = f'{crop}__{record["config"]}__{model}__{stage}'
                predictions = pd.read_parquet(OUT / "predictions" / f"{key}.parquet")
                expected = frames[stage]
                for column in ["row_id", "truss", "feature_date", "target_date", "current", "target"]:
                    pd.testing.assert_series_equal(predictions[column], expected[column])
                assert predictions.row_id.is_unique
                assert np.isfinite(predictions[["prediction", "prediction_raw"]]).all().all()
                np.testing.assert_array_equal(predictions.prediction, np.maximum(predictions.prediction_raw, 0))
                calculated = metrics(predictions, predictions.prediction.to_numpy())
                for metric, value in calculated.items():
                    if value is None:
                        assert record["metrics"][metric] is None
                    else:
                        np.testing.assert_allclose(record["metrics"][metric], value, rtol=1e-10, atol=1e-10)
                assert {r["truss"] for r in record["per_truss"]} == set(range(1, spec["limit"] + 1))
                audited_rows += len(predictions)
    result = {"status": "passed", "runs": len(records), "prediction_rows": audited_rows,
              "checks": ["cohort hashes", "complete 4x6 validation grid", "validation-only selection",
                         "refit epochs", "prediction alignment", "finite nonnegative predictions",
                         "recomputed metrics", "all trusses", "saved test models"]}
    (OUT / "result_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
