from __future__ import annotations

from itertools import combinations

import pandas as pd

from ..core import digest
from .config import COMBINABLE_GROUPS


def all_combinations(groups=COMBINABLE_GROUPS):
    """Return all non-empty subsets in stable cardinality/lexical order."""
    groups = tuple(groups)
    result = []
    for size in range(1, len(groups) + 1):
        for members in combinations(groups, size):
            result.append(
                {
                    "Combination_ID": "COMBO_" + digest(list(members))[:12],
                    "Groups": list(members),
                    "Group_Count": size,
                    "Label": "+".join(members),
                }
            )
    if len(result) != 127 or len({x["Combination_ID"] for x in result}) != 127:
        raise AssertionError(
            "E1-E7 must produce exactly 127 unique non-empty combinations"
        )
    return result


def write_catalog(path):
    rows = all_combinations()
    pd.DataFrame(
        {
            "Combination_ID": [x["Combination_ID"] for x in rows],
            "Variable_Groups": ["|".join(x["Groups"]) for x in rows],
            "Group_Count": [x["Group_Count"] for x in rows],
            "Label": [x["Label"] for x in rows],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")
    return rows
