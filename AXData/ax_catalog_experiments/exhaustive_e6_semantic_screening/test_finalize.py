import json
from types import SimpleNamespace

import pandas as pd
import pytest

from AXData.ax_catalog_experiments.exhaustive_e6_semantic_screening import finalize as module


def test_test_uses_exact_screening_features_before_loading_test(tmp_path, monkeypatch):
    winner = {'Combination_ID': 'combo', 'Selected_Features': ['kept'], 'Params': {}, 'Best_Epoch': 3}
    path = tmp_path / 'validation_winners' / 'target' / 'model.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(winner))
    frozen = SimpleNamespace(selected=['kept'])
    monkeypatch.setattr(module, '_load_selection', lambda *args: frozen)
    monkeypatch.setattr(module, 'all_combinations', lambda: [{'Combination_ID': 'combo', 'Groups': ['E1']}])
    seen = []

    class Campaign:
        train = pd.DataFrame({'target': [1]})
        validation = pd.DataFrame({'target': [2]})
        def __init__(self, *args):
            pass
        def load(self, split):
            assert (tmp_path / 'test_feature_freeze/target/model.json').exists()
            assert split == 'test'
            return pd.DataFrame({'target': [3]})
        def evaluate(self, combination, model, params, seed, stage, fit, evaluation, **kwargs):
            assert kwargs['frozen_selection'] is frozen
            assert kwargs['epochs'] == 3
            assert stage == 'test'
            seen.append(seed)
            return {'Status': 'success', 'Run_ID': str(seed), 'Metrics': {'bounded': dict(rmse=1, mae=1, r2=0, ccc=0)}}

    monkeypatch.setattr(module, 'CompletionTargetCampaign', Campaign)
    record = module.evaluate_screening_winner(tmp_path, 'target', 'model')
    assert seen == [42, 52, 62]
    assert record['frozen_features'] == ['kept']
    assert record['Status'] == 'success'
    winner['Selected_Features'] = ['changed']
    path.write_text(json.dumps(winner))
    with pytest.raises(ValueError, match='freeze mismatch'):
        module.evaluate_screening_winner(tmp_path, 'target', 'model')
