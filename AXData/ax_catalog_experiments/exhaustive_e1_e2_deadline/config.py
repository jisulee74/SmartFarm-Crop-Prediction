from __future__ import annotations

import os
from pathlib import Path

from ..catalog import AX
from ..exhaustive.config import MODELS, TARGETS

CAMPAIGN_ID = os.getenv("AX_DEADLINE_CAMPAIGN_ID", "exhaustive_e1_e2_deadline_v1")
METHOD = os.getenv(
    "AX_DEADLINE_METHOD", "deadline_fixed_configuration_screening"
)
ACTIVE_SEEDS = tuple(
    int(value.strip())
    for value in os.getenv("AX_DEADLINE_SEEDS", "42,52,62").split(",")
    if value.strip()
)
VERSION = "1.0"
SOURCE_ROOT = Path(
    os.getenv(
        "AX_DEADLINE_SOURCE_ROOT",
        str(AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_backfill_v1"),
    )
).expanduser().resolve()
V1_ROOT = Path(
    os.getenv(
        "AX_DEADLINE_V1_ROOT",
        str(AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1"),
    )
).expanduser().resolve()
DEFAULT_OUTPUT = AX / "ax_catalog_experiments" / "outputs" / CAMPAIGN_ID
PACKAGE_ROOT = Path(__file__).parent

# Fixed-configuration screening keeps the three registered validation seeds but
# reuses each target/model's v1 winner instead of repeating per-combination CV.
# All 127 combinations are rerun so rankings remain internally comparable.
GPU_MODELS = ("catboost", "mlp", "tabm", "tft")
CPU_MODELS = ("poisson", "random_forest")
