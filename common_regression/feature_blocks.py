from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class FeatureSetChoice:
    name: str
    features: tuple[str, ...]
    normalized_mean_cv_rmse: float


def unique_union(*parts: list[str]) -> list[str]:
    return list(dict.fromkeys(feature for part in parts for feature in part))


def flower_feature_sets(blocks: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
    combined = unique_union(blocks["individual_history"], blocks["growth"])
    return {
        "individual_history": blocks["individual_history"],
        "individual_history_growth": combined,
    }, {"full": "individual_history_growth"}


def summarize_feature_sets(metrics: pd.DataFrame, baseline_name: str = "individual_history") -> pd.DataFrame:
    required = {"feature_set", "model", "rmse", "mae", "feature_count"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Feature-set metrics missing: {sorted(required - set(metrics.columns))}")
    grouped = metrics.groupby(["feature_set", "model"], as_index=False).agg(
        cv_rmse=("rmse", "mean"), cv_mae=("mae", "mean")
    )
    baseline = grouped[grouped.feature_set.eq(baseline_name)].set_index("model").cv_rmse
    if set(grouped.model.unique()) != set(baseline.index):
        raise ValueError("Every model requires an individual-history baseline")
    grouped["normalized_rmse"] = [row.cv_rmse / baseline.loc[row.model] for row in grouped.itertuples()]
    grouped["rank"] = grouped.groupby("model")["cv_rmse"].rank(method="average")
    summary = grouped.groupby("feature_set", as_index=False).agg(
        normalized_mean_cv_rmse=("normalized_rmse", "mean"),
        mean_rank=("rank", "mean"), mean_cv_mae=("cv_mae", "mean"),
    )
    counts = metrics.groupby("feature_set", as_index=False).feature_count.first()
    return summary.merge(counts, on="feature_set", validate="one_to_one")


def choose_final_feature_set(summary: pd.DataFrame, tolerance: float = 0.01) -> pd.Series:
    best = float(summary.normalized_mean_cv_rmse.min())
    tied = summary[summary.normalized_mean_cv_rmse.le(best * (1 + tolerance))]
    return tied.sort_values(
        ["mean_rank", "mean_cv_mae", "feature_count", "feature_set"], kind="stable"
    ).iloc[0]


def forward_select_blocks(
    blocks: dict[str, list[str]],
    evaluate: Callable[[str, list[str]], pd.DataFrame],
    improvement_threshold: float = 0.01,
    non_worse_models: int = 3,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    current_name = "individual_history"
    current = list(blocks[current_name])
    evaluated = {current_name: current}
    metric_parts = [evaluate(current_name, current)]
    remaining = [name for name in ("growth", "nutrient", "indoor", "outdoor") if blocks.get(name)]
    history: list[dict] = []
    while remaining:
        candidate_parts = []
        for block in remaining:
            name = f"{current_name}+{block}"
            features = unique_union(current, blocks[block])
            values = evaluate(name, features)
            candidate_parts.append((block, name, features, values))
            evaluated[name] = features
            metric_parts.append(values)
        all_metrics = pd.concat(metric_parts, ignore_index=True)
        summary = summarize_feature_sets(all_metrics)
        current_score = float(summary.loc[summary.feature_set.eq(current_name), "normalized_mean_cv_rmse"].iloc[0])
        eligible = []
        current_model = all_metrics[all_metrics.feature_set.eq(current_name)].groupby("model").rmse.mean()
        for block, name, features, values in candidate_parts:
            score = float(summary.loc[summary.feature_set.eq(name), "normalized_mean_cv_rmse"].iloc[0])
            candidate_model = values.groupby("model").rmse.mean()
            improvement = (current_score - score) / current_score
            non_worse = int((candidate_model <= current_model).sum())
            accepted = improvement > improvement_threshold and non_worse >= non_worse_models
            history.append({"current": current_name, "block": block, "candidate": name,
                            "normalized_mean_cv_rmse": score, "improvement": improvement,
                            "non_worse_models": non_worse, "eligible": accepted})
            if accepted:
                eligible.append((score, name, block, features))
        if not eligible:
            break
        _, current_name, selected_block, current = min(eligible, key=lambda x: (x[0], len(x[3]), x[1]))
        remaining.remove(selected_block)
    full = unique_union(*(blocks[name] for name in ("individual_history", "growth", "nutrient", "indoor", "outdoor") if blocks.get(name)))
    if full and full not in evaluated.values():
        evaluated["full"] = full
        metric_parts.append(evaluate("full", full))
    return evaluated, pd.DataFrame(history)
