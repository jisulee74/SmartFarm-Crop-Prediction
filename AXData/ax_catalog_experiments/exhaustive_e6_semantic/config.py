from __future__ import annotations

from pathlib import Path

from ..catalog import AX
from ..exhaustive.config import MODELS, SEEDS, SELECTION_POLICY, TARGETS

VERSION = "2.0"
CAMPAIGN_ID = "exhaustive_e6_semantic_v2"
V1_ROOT = AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1"
DEFAULT_OUTPUT = AX / "ax_catalog_experiments" / "outputs" / CAMPAIGN_ID
PACKAGE_ROOT = Path(__file__).parent
REFERENCE_FILE = PACKAGE_ROOT / "sfkr_col_info_cr.csv"
WINDOW_DAYS = (1, 3, 7)

__all__ = [
    "CAMPAIGN_ID",
    "DEFAULT_OUTPUT",
    "MODELS",
    "REFERENCE_FILE",
    "SEEDS",
    "SELECTION_POLICY",
    "TARGETS",
    "V1_ROOT",
    "VERSION",
    "WINDOW_DAYS",
]
