from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import digest
from ..exhaustive.config import SELECTION_POLICY
from ..exhaustive.io import atomic_json


def _complexity(name: str) -> tuple[int, int]:
    text = str(name).lower()
    derived = 0 if text.endswith("__current") or "__" not in text else 1
    match = re.search(r"__(\d+)d_|__roll(\d+)_|__lag(\d+)$", text)
    window = int(next(x for x in match.groups() if x)) if match else 0
    return derived, window


def _retain_key(name: str, missing_rate: float) -> tuple:
    return (float(missing_rate), *_complexity(name), str(name))


@dataclass
class CompactSelection:
    selected_numeric: list[str]
    selected_categorical: list[str]
    exclusions: list[dict]
    categorical_levels: dict[str, dict]
    diagnostics: dict

    @property
    def selected(self):
        return self.selected_numeric + self.selected_categorical


def _standardized_numeric(train: pd.DataFrame, numeric: list[str]):
    missing, normalized, exclusions = {}, {}, []
    for name in sorted(dict.fromkeys(numeric)):
        series = (
            pd.to_numeric(train[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if name in train
            else pd.Series(np.nan, index=train.index)
        )
        rate = float(series.isna().mean())
        missing[name] = rate
        if rate > SELECTION_POLICY["missing_rate_gt"]:
            exclusions.append({"Variable_ID": name, "Reason": "missing_over_50_percent", "Missing_Rate": rate})
            continue
        if series.dropna().nunique() < 2:
            exclusions.append({"Variable_ID": name, "Reason": "constant_or_empty", "Missing_Rate": rate})
            continue
        values = series.fillna(series.median()).to_numpy(np.float64)
        values = (values - values.mean()) / values.std(ddof=0)
        normalized[name] = values
    return missing, normalized, exclusions


def _drop_duplicates(normalized, missing, exclusions):
    kept, signatures = [], {}
    for name in sorted(normalized):
        values = normalized[name]
        signature = digest(values.tolist())
        duplicate = next((other for other in signatures.get(signature, []) if np.array_equal(values, normalized[other])), None)
        if duplicate:
            exclusions.append({
                "Variable_ID": name,
                "Reason": "exact_duplicate",
                "Related_Variable": duplicate,
                "Missing_Rate": missing[name],
                "Correlation": 1.0,
            })
        else:
            signatures.setdefault(signature, []).append(name)
            kept.append(name)
    return kept


def _drop_correlated(kept, normalized, missing, exclusions):
    if len(kept) < 2:
        return kept, 0
    matrix = np.column_stack([normalized[c] for c in kept])
    corr = np.corrcoef(matrix, rowvar=False)
    upper = np.triu_indices(len(kept), 1)
    candidates = []
    threshold = SELECTION_POLICY["correlation_abs_gte"]
    for i, j, value in zip(upper[0], upper[1], corr[upper]):
        value = float(value)
        if np.isfinite(value) and abs(value) >= threshold:
            candidates.append((abs(value), kept[i], kept[j], value))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    active = set(kept)
    removed = 0
    for _, left, right, value in candidates:
        if left not in active or right not in active:
            continue
        drop = max((left, right), key=lambda c: _retain_key(c, missing[c]))
        keep = right if drop == left else left
        exclusions.append({
            "Variable_ID": drop,
            "Reason": "pearson_abs_gte_0.95",
            "Related_Variable": keep,
            "Missing_Rate": missing[drop],
            "Correlation": value,
        })
        active.remove(drop)
        removed += 1
    return [c for c in kept if c in active], removed


def _drop_vif(kept, normalized, missing, exclusions):
    """Iterative VIF using inverse-correlation downdates.

    The original implementation solved p regressions and serialized every VIF
    at every removal. This computes the same diagnostic from the correlation
    inverse and records only the variable that caused each exclusion.
    """
    if len(kept) < 2:
        return kept, 0
    matrix = np.column_stack([normalized[c] for c in kept])
    corr = np.corrcoef(matrix, rowvar=False)
    # A tiny ridge makes the inverse stable when p approaches the effective rank.
    inverse = np.linalg.pinv(corr + np.eye(len(kept)) * 1e-10, hermitian=True)
    names = list(kept)
    removed = 0
    while len(names) > 1:
        values = np.diag(inverse)
        over = np.flatnonzero(values > SELECTION_POLICY["vif_gt"])
        if not len(over):
            break
        worst = float(np.max(values[over]))
        tied = [int(i) for i in over if np.isclose(values[i], worst, rtol=1e-7, atol=1e-9)]
        index = max(tied, key=lambda i: _retain_key(names[i], missing[names[i]]))
        name = names[index]
        exclusions.append({
            "Variable_ID": name,
            "Reason": "vif_over_10",
            "Missing_Rate": missing[name],
            "VIF": worst,
            "VIF_Is_Infinite": not np.isfinite(worst),
        })
        mask = np.ones(len(names), dtype=bool)
        mask[index] = False
        pivot = float(inverse[index, index])
        reduced = inverse[np.ix_(mask, mask)]
        if np.isfinite(pivot) and abs(pivot) > 1e-14:
            column = inverse[mask, index]
            reduced = reduced - np.outer(column, column) / pivot
        names.pop(index)
        inverse = reduced
        removed += 1
    return names, removed


def select_compact(train: pd.DataFrame, numeric: list[str], categorical: list[str]) -> CompactSelection:
    missing, normalized, exclusions = _standardized_numeric(train, numeric)
    kept = _drop_duplicates(normalized, missing, exclusions)
    before_corr = len(kept)
    kept, corr_removed = _drop_correlated(kept, normalized, missing, exclusions)
    before_vif = len(kept)
    kept, vif_removed = _drop_vif(kept, normalized, missing, exclusions)

    selected_cat, levels = [], {}
    for name in sorted(dict.fromkeys(categorical)):
        raw = train[name] if name in train else pd.Series(np.nan, index=train.index)
        rate = float(raw.isna().mean())
        text = raw.fillna("__MISSING__").astype(str)
        if rate > SELECTION_POLICY["missing_rate_gt"]:
            exclusions.append({"Variable_ID": name, "Reason": "missing_over_50_percent", "Missing_Rate": rate})
            continue
        if text.nunique() < 2:
            exclusions.append({"Variable_ID": name, "Reason": "constant_or_empty", "Missing_Rate": rate})
            continue
        counts = text.value_counts()
        rare = sorted(counts[(counts < SELECTION_POLICY["rare_count_lt"]) | (counts / len(text) < SELECTION_POLICY["rare_fraction_lt"])].index.astype(str))
        known = sorted(set(text.unique()) - set(rare))
        levels[name] = {"known": known, "rare": rare, "missing_token": "__MISSING__", "unknown_token": "__UNKNOWN__"}
        selected_cat.append(name)
    return CompactSelection(
        selected_numeric=kept,
        selected_categorical=selected_cat,
        exclusions=exclusions,
        categorical_levels=levels,
        diagnostics={
            "candidate_numeric": len(set(numeric)),
            "candidate_categorical": len(set(categorical)),
            "after_basic_and_duplicate": before_corr,
            "correlation_removed": corr_removed,
            "before_vif": before_vif,
            "vif_removed": vif_removed,
            "selected_total": len(kept) + len(selected_cat),
            "recording_mode": "exclusions_only_compact",
        },
    )


def load_or_select(root: Path, target: str, combo: str, train: pd.DataFrame, numeric: list[str], categorical: list[str]):
    path = Path(root) / "feature_selection" / target / f"{combo}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return CompactSelection(data["selected_numeric"], data["selected_categorical"], data["exclusions"], data["categorical_levels"], data["diagnostics"])
    result = select_compact(train, numeric, categorical)
    atomic_json(path, {
        "target": target,
        "combination": combo,
        "selected_numeric": result.selected_numeric,
        "selected_categorical": result.selected_categorical,
        "exclusions": result.exclusions,
        "categorical_levels": result.categorical_levels,
        "diagnostics": result.diagnostics,
        "policy": SELECTION_POLICY,
    })
    return result
