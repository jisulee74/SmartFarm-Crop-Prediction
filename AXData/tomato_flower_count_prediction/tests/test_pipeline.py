from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import add_simple_sensor_features, prepare_growth, split_and_filter_features


def growth_row(facility, crop, date, truss, value, sn):
    return {"source_sn": sn, "facility_id": facility, "crop_sn": crop, "sample_num": 1,
            "cropping_date": "2025-01-01", "cropping_end_date": "2025-12-31",
            "examin_date": date, "growth_measure_code": 10000233 + truss, "flower_count": value}


def test_target_stays_within_facility_and_crop_and_sums_contiguous_trusses():
    rows = []
    sn = 1
    for facility in ("PF_A_01", "PF_B_01"):
        for date, values in (("2025-02-01", [2, 3]), ("2025-02-08", [3, 4]), ("2025-02-15", [4, 5])):
            for truss, value in enumerate(values, 1):
                rows.append(growth_row(facility, 10 if facility == "PF_A_01" else 20, date, truss, value, sn)); sn += 1
    candidates, audits = prepare_growth(pd.DataFrame(rows))
    assert len(candidates) == 4
    assert set(candidates.current_total_flower_count) == {5, 7}
    assert set(candidates.next_total_flower_count) == {7, 9}
    assert audits["incomplete_truss_dates"].empty
    assert (candidates.target_date > candidates.feature_date).all()


def test_conflict_and_internal_truss_gap_remove_whole_survey_date():
    rows = [growth_row("PF_A_01", 1, "2025-02-01", 1, 2, 1),
            growth_row("PF_A_01", 1, "2025-02-01", 1, 9, 2),
            growth_row("PF_A_01", 1, "2025-02-08", 1, 2, 3),
            growth_row("PF_A_01", 1, "2025-02-08", 3, 2, 4),
            growth_row("PF_A_01", 1, "2025-02-15", 1, 3, 5),
            growth_row("PF_A_01", 1, "2025-02-22", 1, 4, 6)]
    candidates, audits = prepare_growth(pd.DataFrame(rows))
    assert len(audits["conflicting_duplicates"]) == 1
    assert len(audits["incomplete_truss_dates"]) == 1
    assert len(candidates) == 1


def test_sensor_window_excludes_survey_day_and_rejects_temperature_outlier():
    growth = pd.DataFrame([growth_row("PF_A_01", 1, "2025-02-08", 1, 2, 1),
                           growth_row("PF_A_01", 1, "2025-02-15", 1, 3, 2)])
    candidates, _ = prepare_growth(growth)
    base = {"source_sn": 1, "facility_id": "PF_A_01", "item_code": "080300",
            "fld_code": "A", "maker_id": "M", "sen_id": "S"}
    env = pd.DataFrame([
        {**base, "sect_code": "EI", "fatr_code": "TI", "sen_val": 20, "meas_date": "2025-02-07 12:00"},
        {**base, "source_sn": 2, "sect_code": "EI", "fatr_code": "TI", "sen_val": 2000, "meas_date": "2025-02-07 13:00"},
        {**base, "source_sn": 3, "sect_code": "EI", "fatr_code": "TI", "sen_val": 30, "meas_date": "2025-02-08 01:00"},
    ])
    enriched, quality = add_simple_sensor_features(candidates, env)
    assert enriched.indoor_temperature_7d_mean.iloc[0] == 20
    assert quality.indoor_temperature_7d_mean_outliers.iloc[0] == 1


def test_facility_split_has_no_overlap_and_train_only_medians():
    rows = []
    for i in range(7):
        for j in range(3):
            rows.append({"facility_id": f"F{i}", "crop_sn": i, "row_id": f"{i}-{j}",
                         "feature_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=j),
                         "target_date": pd.Timestamp("2025-01-02") + pd.Timedelta(days=j),
                         "next_total_flower_count": 2, **{x: 1.0 for x in [
                             "current_total_flower_count", "previous_total_flower_count", "total_flower_delta",
                             "current_max_truss_number", "positive_truss_count", "observed_truss_count",
                             "days_since_previous", "days_since_crop_start", "week_of_year_sin", "week_of_year_cos",
                             "indoor_temperature_7d_mean", "indoor_humidity_7d_mean", "indoor_co2_7d_mean",
                             "heating_7d_duty", "co2_control_7d_duty", "ventilation_7d_duty", "shade_curtain_7d_duty"]}})
    frames, _, manifest = split_and_filter_features(pd.DataFrame(rows))
    sets = {k: set(v.facility_id) for k, v in frames.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert np.isfinite(list(manifest["train_medians"].values())).all()

