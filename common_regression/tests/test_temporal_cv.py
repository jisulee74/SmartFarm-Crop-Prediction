import numpy as np
import pandas as pd

from common_regression.temporal_cv import build_expanding_folds


def test_last_fold_includes_last_date_and_all_dates_are_covered():
    frame = pd.DataFrame({"target_date": pd.date_range("2025-01-01", periods=65)})
    folds = build_expanding_folds(frame, "target_date")
    assert len(folds) == 3
    assert frame.iloc[folds[-1][1]].target_date.max() == frame.target_date.max()
    covered = set(frame.iloc[:26].target_date)
    for fit, validation in folds:
        assert frame.iloc[fit].target_date.max() < frame.iloc[validation].target_date.min()
        covered.update(frame.iloc[validation].target_date)
    assert covered == set(frame.target_date)
