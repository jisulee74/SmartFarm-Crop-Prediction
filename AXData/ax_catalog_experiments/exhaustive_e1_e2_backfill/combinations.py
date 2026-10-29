from __future__ import annotations

from ..exhaustive.combinations import all_combinations


def backfill_combinations():
    """The 96 combinations that were empty because they contain E1 or E2."""
    rows = [
        row
        for row in all_combinations()
        if {"E1", "E2"} & set(row["Groups"])
    ]
    if len(rows) != 96 or len({row["Combination_ID"] for row in rows}) != 96:
        raise AssertionError("E1/E2 backfill scope must contain 96 unique combinations")
    return rows


def reused_v1_combinations():
    """The 31 already evaluated combinations containing only E3-E7."""
    rows = [
        row
        for row in all_combinations()
        if not ({"E1", "E2"} & set(row["Groups"]))
    ]
    if len(rows) != 31:
        raise AssertionError("Expected 31 reusable v1 combinations")
    return rows
