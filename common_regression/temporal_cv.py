from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ContractError


def build_expanding_folds(
    train: pd.DataFrame,
    date_column: str,
    n_folds: int = 3,
    initial_fraction: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return date-aligned expanding folds, including the final Train date."""
    dates = np.array(sorted(pd.to_datetime(train[date_column]).unique()))
    if n_folds < 2 or not 0 < initial_fraction < 1:
        raise ContractError("Invalid expanding-window configuration")
    initial = max(1, int(len(dates) * initial_fraction))
    remaining = len(dates) - initial
    if remaining < n_folds:
        raise ContractError("Not enough unique dates for expanding folds")
    edges = [initial + (remaining * i) // n_folds for i in range(n_folds + 1)]
    edges[-1] = len(dates)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    covered: set[pd.Timestamp] = set(pd.Timestamp(x) for x in dates[:initial])
    values = pd.to_datetime(train[date_column])
    for fold, (left, right) in enumerate(zip(edges[:-1], edges[1:]), 1):
        fit_mask = values.isin(dates[:left])
        val_mask = values.isin(dates[left:right])
        fit_idx, val_idx = np.flatnonzero(fit_mask), np.flatnonzero(val_mask)
        if not len(fit_idx) or not len(val_idx):
            raise ContractError(f"Fold {fold} is empty")
        if values.iloc[fit_idx].max() >= values.iloc[val_idx].min():
            raise ContractError(f"Fold {fold} violates chronology")
        covered.update(pd.Timestamp(x) for x in dates[left:right])
        folds.append((fit_idx, val_idx))
    if covered != {pd.Timestamp(x) for x in dates}:
        raise ContractError("Expanding folds do not cover all Train dates")
    if values.iloc[folds[-1][1]].max() != pd.Timestamp(dates[-1]):
        raise ContractError("Last fold does not include final Train date")
    return folds
