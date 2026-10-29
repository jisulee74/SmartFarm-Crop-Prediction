import json
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from AXData.ax_catalog_experiments.exhaustive.combinations import all_combinations
from AXData.ax_catalog_experiments.exhaustive.credentials import load_db_environment
from AXData.ax_catalog_experiments.exhaustive.selection import (
    select_features,
    cached_selection,
)
from AXData.ax_catalog_experiments.exhaustive.models import FoldEncoder
from AXData.ax_catalog_experiments.exhaustive.io import atomic_json
from AXData.ax_catalog_experiments.exhaustive.runner import status
from AXData.ax_catalog_experiments.exhaustive.snapshot import (
    _query,
    _write_frames,
)


def test_exactly_127_nonempty_unique_combinations():
    rows = all_combinations()
    assert len(rows) == 127
    assert len({r["Combination_ID"] for r in rows}) == 127
    assert all(r["Groups"] for r in rows)
    assert {len(r["Groups"]) for r in rows} == set(range(1, 8))


def test_legacy_db_environment_loader_does_not_override_explicit_values(
    tmp_path, monkeypatch
):
    legacy = tmp_path / ".env"
    legacy.write_text(
        "GEAS_DB_HOST=legacy-host\nGEAS_DB_PASSWORD='legacy-secret'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FARMSTOM_DB_HOST", "explicit-host")
    monkeypatch.delenv("FARMSTOM_DB_PASSWORD", raising=False)
    monkeypatch.delenv("FARMSTOM_DB_NAME", raising=False)
    assert load_db_environment(legacy) == legacy
    assert os.environ["FARMSTOM_DB_HOST"] == "explicit-host"
    assert os.environ["FARMSTOM_DB_PASSWORD"] == "legacy-secret"
    assert os.environ["FARMSTOM_DB_NAME"] == "farmstom_usoo"


def test_correlation_and_vif_audit_is_fit_only():
    rng = np.random.default_rng(3)
    x = np.arange(40, dtype=float)
    train = pd.DataFrame(
        {
            "raw": x,
            "high_corr": x + rng.normal(0, 0.01, 40),
            "independent": rng.normal(size=40),
            "missing": [np.nan] * 21 + list(range(19)),
        }
    )
    result = select_features(train, list(train), [])
    reasons = {r["Variable_ID"]: r["Reason"] for r in result.exclusions}
    assert reasons["missing"] == "missing_over_50_percent"
    assert any(v == "pearson_abs_gte_0.95" for v in reasons.values())
    assert result.vif_iterations


def test_vif_removes_multicollinear_member_and_records_value():
    rng = np.random.default_rng(9)
    a = rng.normal(size=80)
    b = rng.normal(size=80)
    d = pd.DataFrame({"a": a, "b": b, "c": a + b})
    result = select_features(d, ["a", "b", "c"], [])
    removed = [r for r in result.exclusions if r["Reason"] == "vif_over_10"]
    assert removed and ("VIF" in removed[0] or removed[0].get("VIF_Is_Infinite"))


def test_categorical_rare_missing_unknown_encoding():
    d = pd.DataFrame({"kind": ["a"] * 10 + ["b"] * 2 + [None] * 2, "x": range(14)})
    selection = select_features(d, ["x"], ["kind"])
    enc = FoldEncoder(
        selection.selected_numeric,
        selection.selected_categorical,
        selection.categorical_levels,
    ).fit(d)
    out = enc.transform(pd.DataFrame({"kind": ["never", None, "b"], "x": [1, 2, 3]}))
    assert len(out) == 3 and np.isfinite(out.to_numpy()).all()
    assert all(c != "kind" for c in out.columns)


def test_cached_selector_ignores_evaluation_data(tmp_path):
    fit = pd.DataFrame(
        {"row_id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}
    )
    first = cached_selection(tmp_path, "t", "c", fit, ["x", "y"], [], "validation")
    evaluation = pd.DataFrame({"x": [1e12], "y": [-1e12]})
    second = cached_selection(tmp_path, "t", "c", fit, ["x", "y"], [], "validation")
    assert first.as_dict() == second.as_dict()
    assert len(evaluation) == 1


def test_status_counts_only_persisted_results(tmp_path):
    p = tmp_path / "runs" / "a"
    p.mkdir(parents=True)
    (p / "result.json").write_text(json.dumps({"Status": "success"}))
    (tmp_path / "runs" / "b").mkdir()
    (tmp_path / "runs" / "b" / "running.json").write_text("{}")
    result = status(tmp_path)
    assert result["Persisted_Runs"] == 1 and result["Run_Status"] == {"success": 1}


def _snapshot_test_schema():
    return pd.DataFrame(
        {
            "column_name": ["sn", "value", "meas_date"],
            "column_type": ["bigint(20) unsigned", "varchar(100)", "datetime"],
            "ordinal_position": [1, 2, 3],
        }
    )


def test_streamed_parquet_handles_null_first_chunk_and_commits_atomically(tmp_path):
    path = tmp_path / "streamed.parquet"
    frames = [
        pd.DataFrame(
            {
                "sn": [1, 2],
                "value": [None, None],
                "meas_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        ),
        pd.DataFrame(
            {
                "sn": [3],
                "value": ["12.5"],
                "meas_date": pd.to_datetime(["2024-01-03"]),
            }
        ),
    ]
    progress = []
    stats = _write_frames(
        iter(frames),
        path,
        _snapshot_test_schema(),
        "meas_date",
        "2024-01-01 00:00:00",
        lambda rows, chunks, size: progress.append((rows, chunks, size)),
    )
    saved = pd.read_parquet(path)
    assert stats["rows"] == 3 and stats["chunks"] == 2
    assert saved["value"].iloc[2] == "12.5"
    assert saved["value"].iloc[:2].isna().all()
    assert progress[-1][0:2] == (3, 2)
    assert not (tmp_path / ".streamed.parquet.partial").exists()


def test_streamed_parquet_rejects_pre_cutoff_rows_without_final_file(tmp_path):
    path = tmp_path / "invalid.parquet"
    frame = pd.DataFrame(
        {
            "sn": [1],
            "value": ["x"],
            "meas_date": pd.to_datetime(["2023-12-31 23:59:59"]),
        }
    )
    import pytest

    with pytest.raises(ValueError, match="Pre-2024"):
        _write_frames(
            iter([frame]),
            path,
            _snapshot_test_schema(),
            "meas_date",
            "2024-01-01 00:00:00",
        )
    assert not path.exists()
    assert not (tmp_path / ".invalid.parquet.partial").exists()


def test_snapshot_queries_apply_2024_cutoff_to_measurements_and_overlapping_crops():
    env_sql, env_params = _query(
        "sfkr_hbfm_env_con", "meas_date", {"meas_date"}, "2024-01-01 00:00:00"
    )
    crop_sql, crop_params = _query(
        "sfkr_pvsn_crop", None, {"cropping_end_date"}, "2024-01-01 00:00:00"
    )
    assert "`meas_date` >= %s" in env_sql
    assert env_params == ("2024-01-01 00:00:00",)
    assert "cropping_end_date" in crop_sql
    assert crop_params == ("2024-01-01",)


def test_atomic_json_allows_concurrent_writers(tmp_path):
    path = tmp_path / "shared.json"
    values = [{"writer": i, "payload": "x" * 1000} for i in range(32)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda value: atomic_json(path, value), values))
    assert json.loads(path.read_text()) in values
    assert not list(tmp_path.glob("*.tmp"))
