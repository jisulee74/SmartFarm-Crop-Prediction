from __future__ import annotations

from ..exhaustive.combinations import all_combinations


def missing_semantic_combinations():
    """E6 combinations omitted from v2 because E1/E2 were unavailable then."""
    rows = [
        row
        for row in all_combinations()
        if "E6" in row["Groups"] and ({"E1", "E2"} & set(row["Groups"]))
    ]
    if len(rows) != 48 or len({row["Combination_ID"] for row in rows}) != 48:
        raise AssertionError("Semantic E6 completion scope must contain 48 combinations")
    return rows


def semantic_v2_combinations():
    rows = [
        row
        for row in all_combinations()
        if "E6" in row["Groups"] and not ({"E1", "E2"} & set(row["Groups"]))
    ]
    if len(rows) != 16:
        raise AssertionError("Existing semantic v2 scope must contain 16 combinations")
    return rows


def e1_e2_without_e6_combinations():
    rows = [
        row
        for row in all_combinations()
        if "E6" not in row["Groups"] and ({"E1", "E2"} & set(row["Groups"]))
    ]
    if len(rows) != 48:
        raise AssertionError("E1/E2 non-E6 scope must contain 48 combinations")
    return rows


def original_v1_combinations():
    rows = [
        row
        for row in all_combinations()
        if "E6" not in row["Groups"] and not ({"E1", "E2"} & set(row["Groups"]))
    ]
    if len(rows) != 15:
        raise AssertionError("Original reusable v1 scope must contain 15 combinations")
    return rows
