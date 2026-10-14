from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SEMANTIC = CACHE / "campaigns" / "semantic_control"


def test_single_public_screening_campaign():
    catalog = json.loads((CACHE / "campaign_catalog.json").read_text(encoding="utf-8"))
    assert catalog["default_campaign"] == "semantic_control"
    assert [item["id"] for item in catalog["campaigns"]] == ["semantic_control"]
    assert catalog["campaigns"][0]["status"] == "screening_complete"
    assert catalog["campaigns"][0]["provisional"] is True


def test_screening_index_and_confirmed_winners():
    index = json.loads((SEMANTIC / "experiment_index.json").read_text(encoding="utf-8"))
    campaign = index["campaign"]
    assert campaign["status"] == "screening_complete"
    assert campaign["partial_results"] is True
    assert campaign["validation_method"] == "single_seed_fixed_configuration_screening"
    assert campaign["total_combinations"] == 127
    assert campaign["semantic_retrained_combinations"] == 127
    assert campaign["reused_combinations"] == 0
    validation_total = 0
    test_total = 0
    for target in index["targets"].values():
        assert len(target["groups"]) == 127
        assert len(target["frozen"]["winners"]) == 6
        validation = [
            value for key, value in target["results"].items() if key.endswith("::validation")
        ]
        assert len(validation) == 127 * 6
        assert all(value["status"] in {"success", "failed", "ineligible"} for value in validation)
        assert all(value.get("completed_seeds") == 1 for value in validation if value["status"] == "success")
        validation_total += len(validation)
        tests = [value for key, value in target["results"].items() if key.endswith("::test")]
        assert len(tests) == 6
        assert all(value["status"] == "success" and value["completed_seeds"] == 3 for value in tests)
        test_total += len(tests)
    assert validation_total == 6096
    assert test_total == 48


def test_screening_exports_and_kpi():
    rankings = pd.read_csv(SEMANTIC / "validation_rankings.csv")
    winners = pd.read_csv(SEMANTIC / "validation_winners.csv")
    tests = pd.read_csv(SEMANTIC / "test_results.csv")
    assert len(rankings) == 6096
    assert len(winners) == 48
    assert len(tests) == 48
    assert (winners.groupby(["Target_ID", "Model_Type"]).size() == 1).all()
    assert set(tests.Status) == {"success"}
    bundle = json.loads((HERE / "ax_direction" / "dashboard_bundle.json").read_text(encoding="utf-8"))
    for row in bundle["tables"]["kpi_planning_reference"]:
        assert row["reference_is_provisional"] is True
        assert row["reference_method_label"] == "의미 통합 127개 조합 1시드 잠정 Validation BEST"



def test_test_predictions_and_feature_freeze_match_source():
    import numpy as np
    root = HERE.parent / 'ax_catalog_experiments/outputs/exhaustive_e6_semantic_screening_v1'
    index = json.loads((SEMANTIC / 'experiment_index.json').read_text())
    for target_id, target in index['targets'].items():
        for model, item in target['frozen']['winners'].items():
            winner = json.loads((root / 'validation_winners' / target_id / f'{model}.json').read_text())
            selection = json.loads((root / 'selections' / target_id / f'{model}.json').read_text())
            assert winner['Selected_Features'] == selection['frozen_features']
            result = target['results'][f"{item['group']}::{model}::test"]
            assert {s['run_id'] for s in result['seeds'].values()} == set(selection['test_runs'])
            for seed in result['seeds'].values():
                prediction = json.loads((SEMANTIC / 'predictions' / f"{seed['run_id']}.json").read_text())
                frame = pd.DataFrame(prediction['predictions'])
                source = json.loads((root / 'runs' / seed['run_id'] / 'result.json').read_text())
                assert len(frame) == source['N_Eval']
                assert source['Selected_Features'] == winner['Selected_Features']
                rmse = np.sqrt(np.mean((frame.target - frame.prediction_bounded) ** 2))
                assert np.isclose(rmse, source['Metrics']['bounded']['rmse'], atol=1e-8)
        for key, result in target['results'].items():
            if key.endswith('::validation'):
                assert result['predictions_available'] is False
                assert all(s['run_id'] is None for s in result['seeds'].values())
                assert result['n_eval'] > 0


def test_validation_metrics_match_screening_and_kpi_best():
    import numpy as np
    rankings = pd.read_csv(SEMANTIC / 'validation_rankings.csv')
    index = json.loads((SEMANTIC / 'experiment_index.json').read_text())
    for row in rankings.itertuples():
        entry = index['targets'][row.Target_ID]['results'][f'{row.Combination_ID}::{row.Model_Type}::validation']
        for metric in ('RMSE', 'MAE', 'R2', 'CCC'):
            assert np.isclose(entry['bounded'][f'{metric.lower()}_mean'], getattr(row, f'Validation_{metric}'))
    bundle = json.loads((HERE / 'ax_direction/dashboard_bundle.json').read_text())
    for row in bundle['tables']['kpi_planning_reference']:
        best = rankings[rankings.Target_ID == row['target']].Validation_RMSE.min()
        assert np.isclose(row['reference_rmse'], best)
        assert np.isclose(row['illustrative_10pct_rmse'], best * .9)
