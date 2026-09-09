import pandas as pd

from common_regression.feature_blocks import choose_final_feature_set, summarize_feature_sets


def test_normalized_rmse_is_primary_and_rank_only_breaks_one_percent_tie():
    rows = []
    for model, base in zip(("poisson", "random_forest", "catboost", "mlp", "tabm"), (2, 3, 4, 5, 6)):
        rows += [
            {"feature_set": "individual_history", "model": model, "rmse": base, "mae": base, "feature_count": 10},
            {"feature_set": "candidate", "model": model, "rmse": base * .98, "mae": base, "feature_count": 20},
        ]
    summary = summarize_feature_sets(pd.DataFrame(rows))
    assert choose_final_feature_set(summary).feature_set == "candidate"


def test_one_percent_tie_prefers_smaller_set_after_equal_rank_and_mae():
    summary = pd.DataFrame([
        {"feature_set": "small", "normalized_mean_cv_rmse": 1.0, "mean_rank": 1.0, "mean_cv_mae": 1.0, "feature_count": 10},
        {"feature_set": "large", "normalized_mean_cv_rmse": .995, "mean_rank": 1.0, "mean_cv_mae": 1.0, "feature_count": 20},
    ])
    assert choose_final_feature_set(summary).feature_set == "small"
