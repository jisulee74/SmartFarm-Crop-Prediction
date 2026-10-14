from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SEMANTIC = CACHE / "campaigns" / "semantic_control"


def test_single_public_semantic_campaign():
    catalog = json.loads((CACHE / "campaign_catalog.json").read_text(encoding="utf-8"))
    assert catalog["default_campaign"] == "semantic_control"
    assert [item["id"] for item in catalog["campaigns"]] == ["semantic_control"]
    assert not ("exp_campaign_panel" in (HERE / "index.html").read_text(encoding="utf-8"))


def test_final_semantic_index_and_winners():
    index = json.loads((SEMANTIC / "experiment_index.json").read_text(encoding="utf-8"))
    assert index["campaign"]["status"] == "completed"
    assert index["campaign"]["total_combinations"] == 127
    assert index["campaign"]["semantic_retrained_combinations"] == 64
    assert index["campaign"]["reused_combinations"] == 63
    assert index["campaign"]["out_of_scope_combinations"] == 0
    terminal = 0
    tests = 0
    for target in index["targets"].values():
        assert len(target["groups"]) == 127
        assert len(target["frozen"]["winners"]) == 6
        validation = [
            value for key, value in target["results"].items() if key.endswith("::validation")
        ]
        assert len(validation) == 127 * 6
        assert all(value["status"] in {"success", "failed", "ineligible"} for value in validation)
        terminal += len(validation)
        test_entries = [
            value for key, value in target["results"].items() if key.endswith("::test")
        ]
        assert len(test_entries) == 6
        assert all(value["status"] == "success" and value["completed_seeds"] == 3 for value in test_entries)
        tests += len(test_entries)
        for entry in test_entries:
            for item in entry["seeds"].values():
                assert (SEMANTIC / "predictions" / f"{item['run_id']}.json").exists()
    assert terminal == 6096
    assert tests == 48


def test_rankings_and_test_exports():
    rankings = pd.read_csv(SEMANTIC / "validation_rankings.csv")
    winners = pd.read_csv(SEMANTIC / "validation_winners.csv")
    tests = pd.read_csv(SEMANTIC / "test_results.csv")
    assert len(rankings) == 6096
    assert len(winners) == 48
    assert len(tests) == 48
    assert (winners.groupby(["Target_ID", "Model_Type"]).size() == 1).all()
    assert set(tests.Status) == {"success"}


def test_kpi_uses_final_semantic_validation():
    bundle = json.loads((HERE / "ax_direction" / "dashboard_bundle.json").read_text(encoding="utf-8"))
    rows = bundle["tables"]["kpi_planning_reference"]
    assert len(rows) == 8
    for row in rows:
        assert row["reference_is_provisional"] is False
        assert row["reference_method_label"] == "의미 통합 127개 조합 Validation BEST"
        assert row["reference_campaign_progress"] == {
            "processed": 6096,
            "total": 6096,
            "percent": 100.0,
        }
        assert abs(row["illustrative_10pct_rmse"] - row["reference_rmse"] * 0.9) < 1e-10


def test_granular_code_technical_archive_checksums():
    archive = HERE / "technical_archive" / "granular_control_codes_v1"
    checksums = json.loads((archive / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums["files"].items():
        path = archive / name
        assert path.exists()
        assert path.stat().st_size == expected["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    compact = pd.read_csv(archive / "validation_results_compact.csv")
    assert len(compact) == 6096

