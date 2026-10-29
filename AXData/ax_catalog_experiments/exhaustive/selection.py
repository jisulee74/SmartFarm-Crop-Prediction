from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import digest
from .config import SELECTION_POLICY
from .io import atomic_json


def _complexity(name: str) -> tuple[int, int]:
    """Raw/current and shorter windows win deterministic correlation ties."""
    text = str(name).lower()
    derived = 0 if text.endswith("__current") or "__" not in text else 1
    match = re.search(r"__(\d+)d_|__roll(\d+)_|__lag(\d+)$", text)
    window = int(next(x for x in match.groups() if x)) if match else 0
    return derived, window


def _retain_key(name: str, missing_rate: float) -> tuple:
    return (float(missing_rate), *_complexity(name), str(name))


def _vif_values(frame: pd.DataFrame) -> dict[str, float]:
    if frame.shape[1] < 2:
        return {c: 1.0 for c in frame}
    x = frame.to_numpy(float)
    values = {}
    for index, column in enumerate(frame.columns):
        y = x[:, index]
        others = np.delete(x, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        try:
            fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
            residual = float(np.sum((y - fitted) ** 2))
            total = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - residual / total if total > 0 else 1.0
            values[column] = (
                float("inf") if r2 >= 1 - 1e-12 else 1.0 / max(1e-12, 1.0 - r2)
            )
        except np.linalg.LinAlgError:
            values[column] = float("inf")
    return values


@dataclass
class SelectionResult:
    selected_numeric: list[str]
    selected_categorical: list[str]
    exclusions: list[dict]
    correlations: list[dict]
    vif_iterations: list[dict]
    categorical_levels: dict[str, dict]

    @property
    def selected(self):
        return self.selected_numeric + self.selected_categorical

    def as_dict(self):
        return {
            "selected_numeric": self.selected_numeric,
            "selected_categorical": self.selected_categorical,
            "exclusions": self.exclusions,
            "correlations": self.correlations,
            "vif_iterations": self.vif_iterations,
            "categorical_levels": self.categorical_levels,
            "policy": SELECTION_POLICY,
        }


def select_features(
    train: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> SelectionResult:
    exclusions, correlations, vif_iterations = [], [], []
    missing = {}
    valid_numeric = []
    normalized = {}
    for name in sorted(dict.fromkeys(numeric)):
        series = (
            pd.to_numeric(train[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if name in train
            else pd.Series(np.nan, index=train.index)
        )
        rate = float(series.isna().mean())
        missing[name] = rate
        reason = None
        if rate > SELECTION_POLICY["missing_rate_gt"]:
            reason = "missing_over_50_percent"
        elif series.dropna().nunique() < 2:
            reason = "constant_or_empty"
        if reason:
            exclusions.append(
                {"Variable_ID": name, "Reason": reason, "Missing_Rate": rate}
            )
        else:
            values = series.fillna(series.median()).astype(float)
            values = (values - values.mean()) / values.std(ddof=0)
            normalized[name] = values
            valid_numeric.append(name)

    # Detect exact duplicates before threshold correlation so the reason remains explicit.
    kept = []
    for name in valid_numeric:
        duplicate = next(
            (
                other
                for other in kept
                if np.array_equal(
                    normalized[name].to_numpy(), normalized[other].to_numpy()
                )
            ),
            None,
        )
        if duplicate:
            exclusions.append(
                {
                    "Variable_ID": name,
                    "Reason": "exact_duplicate",
                    "Related_Variable": duplicate,
                    "Missing_Rate": missing[name],
                    "Correlation": 1.0,
                }
            )
        else:
            kept.append(name)

    # Recompute after every removal; choose the pair with largest absolute correlation.
    iteration = 0
    while len(kept) > 1:
        corr = pd.DataFrame({c: normalized[c] for c in kept}).corr(method="pearson")
        pairs = []
        for i, left in enumerate(kept):
            for right in kept[i + 1 :]:
                value = float(corr.loc[left, right])
                correlations.append(
                    {
                        "Iteration": iteration,
                        "Variable_A": left,
                        "Variable_B": right,
                        "Correlation": value,
                        "Abs_Correlation": abs(value),
                    }
                )
                if abs(value) >= SELECTION_POLICY["correlation_abs_gte"]:
                    pairs.append((abs(value), left, right, value))
        if not pairs:
            break
        _, left, right, value = max(pairs, key=lambda x: (x[0], x[1], x[2]))
        drop = max((left, right), key=lambda c: _retain_key(c, missing[c]))
        keep = right if drop == left else left
        exclusions.append(
            {
                "Variable_ID": drop,
                "Reason": "pearson_abs_gte_0.95",
                "Related_Variable": keep,
                "Missing_Rate": missing[drop],
                "Correlation": value,
                "Iteration": iteration,
            }
        )
        kept.remove(drop)
        iteration += 1

    # Iterative VIF on the same fit-only, median-imputed, standardized matrix.
    iteration = 0
    while len(kept) > 1:
        values = _vif_values(pd.DataFrame({c: normalized[c] for c in kept}))
        for name, value in values.items():
            vif_iterations.append(
                {
                    "Iteration": iteration,
                    "Variable_ID": name,
                    "VIF": None if not np.isfinite(value) else float(value),
                    "VIF_Is_Infinite": not np.isfinite(value),
                }
            )
        over = [c for c in kept if values[c] > SELECTION_POLICY["vif_gt"]]
        if not over:
            break
        worst = max(values[c] for c in over)
        tied = [
            c
            for c in over
            if (not np.isfinite(worst) and not np.isfinite(values[c]))
            or np.isclose(values[c], worst)
        ]
        drop = max(tied, key=lambda c: _retain_key(c, missing[c]))
        exclusions.append(
            {
                "Variable_ID": drop,
                "Reason": "vif_over_10",
                "Missing_Rate": missing[drop],
                "VIF": None if not np.isfinite(values[drop]) else float(values[drop]),
                "VIF_Is_Infinite": not np.isfinite(values[drop]),
                "Iteration": iteration,
            }
        )
        kept.remove(drop)
        iteration += 1

    selected_cat, levels = [], {}
    for name in sorted(dict.fromkeys(categorical)):
        raw = train[name] if name in train else pd.Series(np.nan, index=train.index)
        rate = float(raw.isna().mean())
        text = raw.fillna("__MISSING__").astype(str)
        if rate > SELECTION_POLICY["missing_rate_gt"]:
            exclusions.append(
                {
                    "Variable_ID": name,
                    "Reason": "missing_over_50_percent",
                    "Missing_Rate": rate,
                }
            )
            continue
        if text.nunique() < 2:
            exclusions.append(
                {
                    "Variable_ID": name,
                    "Reason": "constant_or_empty",
                    "Missing_Rate": rate,
                }
            )
            continue
        counts = text.value_counts()
        rare = sorted(
            counts[
                (counts < SELECTION_POLICY["rare_count_lt"])
                | (counts / len(text) < SELECTION_POLICY["rare_fraction_lt"])
            ].index.astype(str)
        )
        known = sorted(set(text.unique()) - set(rare))
        levels[name] = {
            "known": known,
            "rare": rare,
            "missing_token": "__MISSING__",
            "unknown_token": "__UNKNOWN__",
        }
        selected_cat.append(name)
    if not numeric:
        vif_iterations.append(
            {
                "Iteration": 0,
                "Variable_ID": None,
                "VIF": None,
                "Status": "not_applicable_no_numeric_features",
            }
        )
    return SelectionResult(
        kept, selected_cat, exclusions, correlations, vif_iterations, levels
    )


def cached_selection(
    root: Path,
    target: str,
    combo: str,
    fit: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    stage: str,
    fold=None,
) -> SelectionResult:
    fit_hash = digest(sorted(fit.row_id.astype(str).tolist()))
    identity = {
        "target": target,
        "combination": combo,
        "fit_hash": fit_hash,
        "numeric": sorted(numeric),
        "categorical": sorted(categorical),
        "stage": stage,
        "fold": fold,
        "policy": SELECTION_POLICY,
    }
    key = digest(identity)[:24]
    path = Path(root) / "feature_selection" / target / combo / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        result = select_features(fit, numeric, categorical)
        data = {"identity": identity, **result.as_dict()}
        atomic_json(path, data)
    return SelectionResult(
        data["selected_numeric"],
        data["selected_categorical"],
        data["exclusions"],
        data["correlations"],
        data["vif_iterations"],
        data["categorical_levels"],
    )
