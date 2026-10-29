from __future__ import annotations

from ..exhaustive.combinations import all_combinations


def semantic_combinations():
    """The 16 executable E6 combinations when E1/E2 remain unavailable."""
    rows = [
        row
        for row in all_combinations()
        if "E6" in row["Groups"] and not ({"E1", "E2"} & set(row["Groups"]))
    ]
    if len(rows) != 16 or len({r["Combination_ID"] for r in rows}) != 16:
        raise AssertionError("Semantic E6 scope must contain exactly 16 combinations")
    return rows


def reused_v1_combinations():
    """The 15 non-empty combinations of the other available v1 groups."""
    rows = [
        row
        for row in all_combinations()
        if "E6" not in row["Groups"]
        and not ({"E1", "E2"} & set(row["Groups"]))
    ]
    if len(rows) != 15:
        raise AssertionError("Expected 15 reusable non-E6 combinations")
    return rows


def composite_combinations():
    rows = reused_v1_combinations() + semantic_combinations()
    if len(rows) != 31 or len({r["Combination_ID"] for r in rows}) != 31:
        raise AssertionError("Composite v2 ranking must contain 31 combinations")
    return rows
