import pandas as pd

from common_regression import selection


class DummyAdapter:
    best_epoch = None
    def __init__(self, name): self.name = name
    def fit(self, *args, **kwargs): return self
    def predict(self, frame): return frame[f"prediction_{self.name}"].to_numpy()
    def save(self, path): path.write_bytes(self.name.encode())


def test_persistence_is_not_eligible_as_selected_learning_model(monkeypatch):
    models = iter(selection.LEARNING_MODELS)
    monkeypatch.setattr(selection, "make_adapter", lambda name, *args, **kwargs: DummyAdapter(name))
    monkeypatch.setattr(selection, "_measure_inference", lambda adapter, frame: 1.0)
    frame = pd.DataFrame({"y": [1., 2., 3.], "current": [1., 2., 3.]})
    for i, name in enumerate(selection.LEARNING_MODELS):
        frame[f"prediction_{name}"] = [1 + i, 2 + i, 3 + i]
    winner, table, _ = selection.validate_learning_models(
        frame, frame, "y", [], [], {m: {} for m in selection.LEARNING_MODELS},
        {m: None for m in selection.LEARNING_MODELS}, [42], "current")
    assert winner in selection.LEARNING_MODELS
    assert winner != "persistence"
    assert table.loc[table.model.eq("persistence"), "rmse"].iloc[0] == 0
