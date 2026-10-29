from .combinations import (
    e1_e2_without_e6_combinations,
    missing_semantic_combinations,
    original_v1_combinations,
    semantic_v2_combinations,
)


def test_partition_covers_all_127_combinations_once():
    groups = [
        original_v1_combinations(),
        e1_e2_without_e6_combinations(),
        semantic_v2_combinations(),
        missing_semantic_combinations(),
    ]
    ids = [row["Combination_ID"] for group in groups for row in group]
    assert [len(group) for group in groups] == [15, 48, 16, 48]
    assert len(ids) == 127
    assert len(set(ids)) == 127


def test_missing_scope_requires_e6_and_e1_or_e2():
    for row in missing_semantic_combinations():
        groups = set(row["Groups"])
        assert "E6" in groups
        assert groups & {"E1", "E2"}
