from __future__ import annotations

from pathlib import Path

from ..catalog import AX
from ..exhaustive.config import MODELS, SEEDS, SELECTION_POLICY, TARGETS

VERSION = "2.0"
CAMPAIGN_ID = "exhaustive_e6_semantic_completion_v2"
DEFAULT_OUTPUT = AX / "ax_catalog_experiments" / "outputs" / CAMPAIGN_ID
PACKAGE_ROOT = Path(__file__).parent

BACKFILL_ROOT = (
    AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_backfill_v1"
)
DEADLINE_ROOT = (
    AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_deadline_v1"
)
SEMANTIC_V2_ROOT = (
    AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e6_semantic_v2"
)
V1_ROOT = AX / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1"

__all__ = [
    "BACKFILL_ROOT",
    "CAMPAIGN_ID",
    "DEADLINE_ROOT",
    "DEFAULT_OUTPUT",
    "MODELS",
    "PACKAGE_ROOT",
    "SEEDS",
    "SELECTION_POLICY",
    "SEMANTIC_V2_ROOT",
    "TARGETS",
    "V1_ROOT",
    "VERSION",
]
