from __future__ import annotations


from ..catalog import AX

VERSION = "1.0"
CUTOFF = "2024-01-01 00:00:00"
COMBINABLE_GROUPS = tuple(f"E{i}" for i in range(1, 8))
MODELS = ("poisson", "random_forest", "catboost", "mlp", "tabm", "tft")
GPU_MODELS = frozenset(("catboost", "mlp", "tabm", "tft"))
CPU_MODELS = frozenset(("poisson", "random_forest"))
SEEDS = (42, 52, 62)
PARTS = {"first": 1, "second": 2, "third": 3, "sum123": None}
TARGETS = {
    f"{crop}_{part}": (crop, part)
    for crop in ("tomato", "strawberry")
    for part in PARTS
}
DEFAULT_OUTPUT = AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1"
DEFINITION = AX / "AX_변수군_정의서_260914.xlsx"

SELECTION_POLICY = {
    "missing_rate_gt": 0.50,
    "correlation_abs_gte": 0.95,
    "vif_gt": 10.0,
    "rare_fraction_lt": 0.01,
    "rare_count_lt": 5,
    "correlation_method": "pearson",
    "fit_only": True,
}

TFT_GRID = [
    {
        "hidden_size": h,
        "attention_head_size": 2,
        "hidden_continuous_size": 8,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 100,
    }
    for h in (16, 32)
]
