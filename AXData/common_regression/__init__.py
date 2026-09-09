"""Leakage-safe common regression model-selection framework."""

from .contracts import TargetSpec
from .runner import CommonRegressionRunner

__all__ = ["CommonRegressionRunner", "TargetSpec"]
