from __future__ import annotations

from pathlib import Path

from ..catalog import AX
from ..exhaustive.config import MODELS, SEEDS, SELECTION_POLICY, TARGETS

VERSION = "1.0"
CAMPAIGN_ID = "exhaustive_e1_e2_backfill_v1"
V1_ROOT = AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1"
DEFAULT_OUTPUT = AX / "ax_catalog_experiments" / "outputs" / CAMPAIGN_ID
PACKAGE_ROOT = Path(__file__).parent

__all__ = [
    "CAMPAIGN_ID",
    "DEFAULT_OUTPUT",
    "MODELS",
    "PACKAGE_ROOT",
    "SEEDS",
    "SELECTION_POLICY",
    "TARGETS",
    "V1_ROOT",
    "VERSION",
]
