from __future__ import annotations

import pandas as pd
import pytest

from .combinations import composite_combinations, reused_v1_combinations, semantic_combinations
from .features import semantic_features_for_facility
from .mapping import TYPO_ALIASES, build_mapping, load_mapping, validate_observed_codes


def test_combination_scope_is_exact():
    semantic = semantic_combinations()
    assert len(semantic) == 16
    assert all("E6" in row["Groups"] for row in semantic)
    assert all(not ({"E1", "E2"} & set(row["Groups"])) for row in semantic)
    assert len(reused_v1_combinations()) == 15
    assert len(composite_combinations()) == 31


def test_mapping_collapses_direction_and_layer_but_not_device_or_measurement():
    _, mapping, _ = load_mapping()
    lookup = mapping.set_index("source_code")
    roof = lookup.loc[["CC01_1_L_V", "CC01_2_R_V", "CC01_3_B_V"]]
    assert roof.device_family.nunique() == 1
    assert roof.measurement_type.nunique() == 1
    assert lookup.loc["CC01_1_L_V", "device_family"] != lookup.loc["CC03_1_L_V", "device_family"]
    assert lookup.loc["CC01_1_L_V", "measurement_type"] != lookup.loc["CC01_1_L_C", "measurement_type"]
    assert not bool(lookup.loc["CC01", "included"])
    assert not bool(lookup.loc["CC03", "included"])
    assert lookup.loc["CC03_1_R_0", "canonical_code"] == TYPO_ALIASES["CC03_1_R_0"]


def test_unknown_and_duplicate_codes_are_blocked():
    _, mapping, _ = load_mapping()
    with pytest.raises(ValueError, match="Unknown E6 codes"):
        validate_observed_codes(["NOT_REGISTERED"], mapping)
    reference = pd.DataFrame(
        [
            {"sect_code": "CR", "fld_code": "FG", "com_code": "CC0403", "com_name": "차광커튼작동상태", "unit": "Y/N"},
            {"sect_code": "CR", "fld_code": "FG", "com_code": "CC0403", "com_name": "중복", "unit": "Y/N"},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        build_mapping(reference)


def _stream(extra_future=False):
    rows = [
        {"facility_id": "F1", "source_code": "CC01_1_L_V", "canonical_code": "CC01_1_L_V", "device_family": "roof_window", "measurement_type": "opening_percent", "sensor_value": 0.0, "meas_date": "2026-01-09 12:00"},
        {"facility_id": "F1", "source_code": "CC01_1_L_V", "canonical_code": "CC01_1_L_V", "device_family": "roof_window", "measurement_type": "opening_percent", "sensor_value": 20.0, "meas_date": "2026-01-09 18:00"},
        {"facility_id": "F1", "source_code": "CC01_2_R_V", "canonical_code": "CC01_2_R_V", "device_family": "roof_window", "measurement_type": "opening_percent", "sensor_value": 40.0, "meas_date": "2026-01-09 18:00"},
        {"facility_id": "F1", "source_code": "CC0403", "canonical_code": "CC0403", "device_family": "shade_curtain", "measurement_type": "binary_status", "sensor_value": 0.0, "meas_date": "2026-01-09 12:00"},
    ]
    if extra_future:
        rows.append({"facility_id": "F1", "source_code": "CC01_3_B_V", "canonical_code": "CC01_3_B_V", "device_family": "roof_window", "measurement_type": "opening_percent", "sensor_value": 99.0, "meas_date": "2026-01-11 00:00"})
    return pd.DataFrame(rows)


def test_semantic_aggregation_is_leak_free_and_device_balanced():
    keys = pd.DataFrame([{"facility_id": "F1", "feature_date": pd.Timestamp("2026-01-10")}])
    before = semantic_features_for_facility(keys, _stream(False))
    after = semantic_features_for_facility(keys, _stream(True))
    pd.testing.assert_frame_equal(before, after)
    prefix = "E6S__roof_window__opening_percent__1d"
    # Member means are 10 and 40, so the facility mean is 25 (not raw-row weighted 20).
    assert before.loc[0, prefix + "__facility_mean"] == 25.0
    assert before.loc[0, prefix + "__reporting_device_count"] == 2
    assert before.loc[0, prefix + "__connection_rate"] == 1.0


def test_missing_record_is_not_converted_to_zero():
    keys = pd.DataFrame(
        [
            {"facility_id": "F1", "feature_date": pd.Timestamp("2026-01-10")},
            {"facility_id": "F1", "feature_date": pd.Timestamp("2026-01-12")},
        ]
    )
    features = semantic_features_for_facility(keys, _stream(False))
    prefix = "E6S__shade_curtain__binary_status__1d"
    assert features.loc[0, prefix + "__active_fraction_mean"] == 0.0
    assert pd.isna(features.loc[1, prefix + "__active_fraction_mean"])
    assert features.loc[1, prefix + "__reporting_device_count"] == 0
