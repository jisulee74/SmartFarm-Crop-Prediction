from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityRules:
    near_constant_ratio: float = 0.99
    numeric_rate_min: float | None = None
    missing_rate_max: float | None = None
    coverage_min: float | None = None


def quality_audit(train: pd.DataFrame, catalog: pd.DataFrame, rules: QualityRules) -> pd.DataFrame:
    rows = []
    for item in catalog.itertuples(index=False):
        values = train[item.feature]
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        unique = int(values.nunique(dropna=True))
        dominant = float(values.value_counts(dropna=False, normalize=True).iloc[0]) if len(values) else 1.0
        numeric_rate = float(numeric.notna().mean())
        missing_rate = float(values.isna().mean())
        indicator = bool(getattr(item, "is_quality_indicator", False))
        status = "kept"
        reason = ""
        if unique < 2:
            status, reason = "removed", "constant"
        elif dominant >= rules.near_constant_ratio and not indicator:
            status, reason = "removed", "near_constant"
        elif rules.numeric_rate_min is not None and numeric_rate < rules.numeric_rate_min:
            status, reason = "removed", "numeric_rate_below_threshold"
        elif rules.missing_rate_max is not None and missing_rate > rules.missing_rate_max:
            status, reason = "removed", "missing_rate_above_threshold"
        rows.append({
            "feature": item.feature, "block": item.block, "semantic_group": item.semantic_group,
            "source_variable": item.source_variable, "unique_count": unique,
            "dominant_ratio": dominant, "numeric_rate": numeric_rate, "missing_rate": missing_rate,
            "status": status, "reason": reason,
        })
    return pd.DataFrame(rows)


def _priority_tuple(row: pd.Series) -> tuple:
    return (
        -int(bool(row.get("inference_available", True))),
        float(row.get("missing_rate", 0.0)),
        -float(row.get("coverage", 0.0)),
        -int(bool(row.get("is_direct", False))),
        int(row.get("derivation_complexity", 99)),
        -abs(float(row.get("target_spearman", 0.0) or 0.0)),
        str(row["feature"]),
    )


def greedy_correlation_filter(
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    threshold_by_block: dict[str, float],
) -> tuple[list[str], pd.DataFrame]:
    """Prune only direct high-correlation duplicates within semantic groups."""
    kept: list[str] = []
    audit: list[dict] = []
    eligible = catalog[catalog["status"].eq("kept")].copy()
    for semantic_group, group in eligible.groupby("semantic_group", sort=True, dropna=False):
        ordered = group.assign(_priority=group.apply(_priority_tuple, axis=1)).sort_values("_priority")
        pending = ordered["feature"].tolist()
        while pending:
            representative = pending.pop(0)
            kept.append(representative)
            survivors: list[str] = []
            rep_row = ordered.set_index("feature").loc[representative]
            threshold = float(threshold_by_block.get(str(rep_row["block"]), 1.01))
            for feature in pending:
                corr = train[representative].corr(train[feature], method="spearman")
                direct = 0.0 if pd.isna(corr) else abs(float(corr))
                feature_row = ordered.set_index("feature").loc[feature]
                same_meaning = bool(rep_row.get("semantic_equivalent", True)) and bool(
                    feature_row.get("semantic_equivalent", True)
                )
                remove = same_meaning and direct >= threshold
                audit.append({
                    "semantic_group": semantic_group,
                    "representative": representative,
                    "removed_feature": feature if remove else "",
                    "compared_feature": feature,
                    "direct_abs_spearman": direct,
                    "threshold": threshold,
                    "representative_source": rep_row["source_variable"],
                    "compared_source": feature_row["source_variable"],
                    "semantic_equivalent": same_meaning,
                    "status": "removed" if remove else (
                        "high_correlation_retained" if direct >= threshold else "retained"
                    ),
                    "reason": "direct_high_correlation_same_meaning" if remove else (
                        "different_physical_meaning" if direct >= threshold else "below_threshold"
                    ),
                })
                if not remove:
                    survivors.append(feature)
            pending = survivors
    return kept, pd.DataFrame(audit)


def apply_kept_features(blocks: dict[str, Iterable[str]], kept: Iterable[str]) -> dict[str, list[str]]:
    allowed = set(kept)
    return {name: [feature for feature in features if feature in allowed] for name, features in blocks.items()}
