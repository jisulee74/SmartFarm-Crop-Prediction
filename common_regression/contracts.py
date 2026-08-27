from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class ContractError(ValueError):
    """Raised when data violates the frozen experiment contract."""


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target: str
    current_value: str
    feature_date: str
    target_date: str
    entity_key: tuple[str, ...]
    row_key: tuple[str, ...]
    forced_exclusions: tuple[str, ...]
    correlation_threshold: float
    blocks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unavailable_blocks: dict[str, str] = field(default_factory=dict)

    def validate_frame(self, frame: pd.DataFrame, split: str) -> None:
        required = {*self.row_key, self.target, self.current_value, self.feature_date, self.target_date}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ContractError(f"{split}: missing required columns: {missing}")
        if frame.duplicated(list(self.row_key)).any():
            raise ContractError(f"{split}: duplicate row keys")
        feature_dates = pd.to_datetime(frame[self.feature_date], errors="raise")
        target_dates = pd.to_datetime(frame[self.target_date], errors="raise")
        if not feature_dates.lt(target_dates).all():
            raise ContractError(f"{split}: feature date must precede target date")
        if frame[self.target].isna().any() or frame[self.current_value].isna().any():
            raise ContractError(f"{split}: target/current value contains nulls")
        if frame[self.target].lt(0).any():
            raise ContractError(f"{split}: count target must be non-negative")

    def validate_features(self, features: list[str]) -> None:
        if not features or len(features) != len(set(features)):
            raise ContractError("Feature list must be non-empty and unique")
        forbidden = sorted(set(features).intersection(self.forced_exclusions))
        if forbidden:
            raise ContractError(f"Target leakage/tracking columns selected: {forbidden}")


@dataclass
class SplitBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test_path: Path
    spec: TargetSpec
    test_loader: Callable[[], pd.DataFrame] | None = None

    def __post_init__(self) -> None:
        self.spec.validate_frame(self.train, "train")
        self.spec.validate_frame(self.validation, "validation")

    def load_test(self) -> pd.DataFrame:
        return self.test_loader() if self.test_loader is not None else pd.read_parquet(self.test_path)


@dataclass(frozen=True)
class ModelTrialResult:
    model: str
    feature_set: str
    params: dict[str, Any]
    seed: int
    fold: int
    rmse: float
    mae: float
    r2: float | None
    ccc: float | None
    best_epoch: int | None = None
