from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float | None]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Actual and predicted must be finite, aligned, and non-empty")
    ym, pm = y.mean(), p.mean()
    denominator = y.var() + p.var() + (ym - pm) ** 2
    if denominator == 0:
        ccc = 1.0 if np.array_equal(y, p) else None
    else:
        ccc = float(2 * np.mean((y - ym) * (p - pm)) / denominator)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if np.unique(y).size > 1 else None,
        "ccc": ccc,
    }
