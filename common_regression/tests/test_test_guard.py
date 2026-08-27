from pathlib import Path

import pandas as pd
import pytest

from common_regression.contracts import SplitBundle, TargetSpec
from common_regression.runner import CommonRegressionRunner, Stage


def test_test_loader_is_not_called_before_final_fit(tmp_path):
    calls = []
    frame = pd.DataFrame({"entity": [1, 2], "feature_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                          "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                          "current": [1., 2.], "target": [2., 3.], "x": [1., 2.]})
    spec = TargetSpec("demo", "target", "current", "feature_date", "target_date", ("entity",),
                      ("entity", "target_date"), ("target", "target_date"), .9)
    path = tmp_path / "test.parquet"; frame.to_parquet(path)
    bundle = SplitBundle(frame, frame, path, spec, test_loader=lambda: calls.append(True) or frame)
    runner = CommonRegressionRunner(bundle, {"individual_history": ["x"]}, [], {}, tmp_path / "artifacts")
    for stage in (Stage.PREPARED, Stage.INTERNAL_FEATURE_SELECTION, Stage.INTERNAL_TUNING,
                  Stage.EXTERNAL_VALIDATION, Stage.FROZEN):
        runner.stage = stage
        with pytest.raises(RuntimeError): runner.test_once()
    assert not calls
