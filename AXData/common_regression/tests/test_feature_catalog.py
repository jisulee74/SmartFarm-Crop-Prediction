import pandas as pd

from common_regression.feature_catalog import greedy_correlation_filter


def test_greedy_pruning_does_not_apply_transitive_connected_component_removal(monkeypatch):
    frame = pd.DataFrame({"A": range(10), "B": range(10), "C": range(10)})
    correlations = {("A", "B"): .95, ("B", "C"): .95, ("A", "C"): .70}
    original = pd.Series.corr
    def fake_corr(left, right, method=None):
        return correlations.get((left.name, right.name), correlations.get((right.name, left.name), original(left, right, method=method)))
    monkeypatch.setattr(pd.Series, "corr", fake_corr)
    catalog = pd.DataFrame([{
        "feature": name, "block": "growth", "semantic_group": "same", "source_variable": name,
        "status": "kept", "inference_available": True, "missing_rate": 0.0, "coverage": 1.0,
        "is_direct": name == "A", "derivation_complexity": i, "target_spearman": 0.0,
        "semantic_equivalent": True,
    } for i, name in enumerate("ABC")])
    kept, audit = greedy_correlation_filter(frame, catalog, {"growth": .90})
    assert kept == ["A", "C"]
    removed = audit[audit.status.eq("removed")]
    assert removed[["representative", "removed_feature"]].values.tolist() == [["A", "B"]]
    assert (removed.direct_abs_spearman >= removed.threshold).all()


def test_different_meaning_is_retained_even_when_highly_correlated():
    frame = pd.DataFrame({"temperature": range(10), "humidity": range(10)})
    catalog = pd.DataFrame([
        {"feature": "temperature", "block": "indoor", "semantic_group": "temperature", "source_variable": "TI", "status": "kept", "inference_available": True, "missing_rate": 0.0, "coverage": 1.0, "is_direct": True, "derivation_complexity": 0, "target_spearman": 0.0, "semantic_equivalent": True},
        {"feature": "humidity", "block": "indoor", "semantic_group": "humidity", "source_variable": "HI", "status": "kept", "inference_available": True, "missing_rate": 0.0, "coverage": 1.0, "is_direct": True, "derivation_complexity": 0, "target_spearman": 0.0, "semantic_equivalent": True},
    ])
    kept, _ = greedy_correlation_filter(frame, catalog, {"indoor": .90})
    assert kept == ["humidity", "temperature"]
