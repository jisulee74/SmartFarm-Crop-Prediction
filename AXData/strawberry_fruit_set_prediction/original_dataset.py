from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from common_regression.contracts import SplitBundle, TargetSpec

ROOT = Path(__file__).resolve().parent
WINDOWS = (1, 3, 7)
STATS = ("mean", "median", "std", "min", "max", "first", "last", "delta", "slope",
         "observation_count", "timestamp_count", "coverage", "max_gap_hours", "change_count",
         "unique_value_count", "missing")


def _window_stats(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    window = frame[frame.meas_date.gt(start) & frame.meas_date.le(end)]
    if window.empty:
        return {name: (1.0 if name == "missing" else 0.0 if name in {
            "observation_count", "timestamp_count", "coverage", "max_gap_hours", "change_count",
            "unique_value_count"} else np.nan) for name in STATS}
    values = window.sensor_value.astype(float); times = window.meas_date.drop_duplicates().sort_values()
    elapsed = (window.meas_date - window.meas_date.iloc[0]).dt.total_seconds().to_numpy() / 3600
    slope = 0.0 if len(window) < 2 or np.ptp(elapsed) == 0 else float(np.polyfit(elapsed, values, 1)[0])
    gaps = times.diff().dt.total_seconds().div(3600).dropna()
    expected = max(1.0, (end - start).total_seconds() / 600)
    return {"mean": values.mean(), "median": values.median(), "std": values.std(ddof=0),
            "min": values.min(), "max": values.max(), "first": values.iloc[0], "last": values.iloc[-1],
            "delta": values.iloc[-1] - values.iloc[0], "slope": slope,
            "observation_count": len(window), "timestamp_count": times.nunique(),
            "coverage": min(1.0, times.nunique() / expected),
            "max_gap_hours": gaps.max() if len(gaps) else 0.0,
            "change_count": values.ne(values.shift()).sum() - 1,
            "unique_value_count": values.nunique(), "missing": 0.0}


def _add_sensor_features(data: pd.DataFrame, sensor: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    grouped = {(facility, variable): frame.sort_values("meas_date").reset_index(drop=True)
               for (facility, variable), frame in sensor.groupby(["facility_id", "variable_id"], sort=False)}
    records = []
    empty = sensor.iloc[0:0]
    for row in data.itertuples(index=False):
        result = {}
        end = pd.Timestamp(row.feature_date)
        for variable in variables:
            frame = grouped.get((row.facility_id, variable), empty)
            for days in WINDOWS:
                for stat, value in _window_stats(frame, end - pd.Timedelta(days=days), end).items():
                    result[f"sensor_{variable}_{days}d_{stat}"] = value
        records.append(result)
    return pd.concat([data.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _select_sensor_variables(data: pd.DataFrame, sensor: pd.DataFrame, metadata: pd.DataFrame) -> list[str]:
    quality = _add_sensor_features(data, sensor, metadata.variable_id.tolist())
    train = quality[quality.target_split.eq("train")]
    selected = []
    for row in metadata.itertuples(index=False):
        missing = train[f"sensor_{row.variable_id}_7d_missing"]
        coverage = train[f"sensor_{row.variable_id}_7d_coverage"]
        mean = train[f"sensor_{row.variable_id}_7d_mean"]
        observed = missing.eq(0)
        facilities = train.loc[observed, "facility_id"].nunique()
        if facilities >= 2 and missing.mean() <= .50 and coverage.mean() >= .10 and mean[observed].nunique() > 1:
            selected.append(row.variable_id)
    return selected


def _growth_blocks(columns: set[str]) -> dict[str, list[str]]:
    prefixes = ("current_fruit_count", "previous_fruit_count", "fruit_count_delta", "fruit_count_roll")
    exact = ["days_since_previous", "days_since_first_measurement", "days_since_first_positive",
             "history_count", "week_of_year_sin", "week_of_year_cos", "fruit_column"]
    exact_set = set(exact)
    growth_all = [f for f in [
        "current_fruit_count", "previous_fruit_count", "fruit_count_delta",
        *[f"fruit_count_roll{w}_{s}" for w in (2, 3, 4) for s in ("mean", "std", "min", "max", "slope")],
        "current_first_fruits_num", "observed_first_fruits_num", "current_second_fruits_num",
        "observed_second_fruits_num", "current_third_fruits_num", "observed_third_fruits_num",
        "leaves_length", "leaves_num", "petiole_length", "theca_diameter", "fruit_cluster_num",
        "first_flower_num", "second_flower_num", "third_flower_num", "first_fruits_num",
        "second_fruits_num", "third_fruits_num", *exact] if f in columns]
    history = [f for f in growth_all if f.startswith(prefixes) or f in exact_set]
    return {"individual_history": history, "growth": [f for f in growth_all if f not in history]}


def load_dataset() -> tuple[SplitBundle, dict[str, list[str]], pd.DataFrame]:
    frames = {s: pd.read_parquet(ROOT / "processed_data" / f"{s}_candidates.parquet")
              for s in ("train", "val")}
    combined = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    metadata = pd.read_csv(ROOT / "data/sensor_category_cache/sensor_metadata.csv")
    sensor = pd.read_parquet(ROOT / "data/sensor_category_cache/sensor_collapsed.parquet")
    sensor["meas_date"] = pd.to_datetime(sensor.meas_date)
    selected = _select_sensor_variables(combined, sensor, metadata)
    enriched = _add_sensor_features(combined, sensor, selected)
    frames = {s: enriched[enriched.target_split.eq(s)].reset_index(drop=True) for s in ("train", "val")}
    blocks = _growth_blocks(set(enriched.columns))
    canonical = json.loads((ROOT / "data/canonical_feature_groups_219.json").read_text())
    canonical_sensor_features = set(canonical["sensor_only"]["features"]) - {"fruit_column"}
    sensor_features = [f for f in enriched if f in canonical_sensor_features]
    for section, block in (("NT", "nutrient"), ("EI", "indoor"), ("EO", "outdoor"), ("CR", "control")):
        variables = set(metadata.loc[metadata.sect_code.eq(section), "variable_id"]).intersection(selected)
        blocks[block] = [f for f in sensor_features if any(f.startswith(f"sensor_{v}_") for v in variables)]
    records = []
    for block, features in blocks.items():
        for f in features:
            if f.startswith("sensor_"):
                import re
                match = re.match(r"^sensor_(.+)_(?:1|3|7)d_", f)
                if match is None:
                    raise ValueError(f"Invalid sensor feature name: {f}")
                source = match.group(1)
                semantic = source
                indicator = f.endswith(("_missing", "_coverage"))
                direct = f.endswith("_mean")
            else:
                source = f.split("_roll")[0].replace("current_", "").replace("previous_", "")
                semantic = source; indicator = f.startswith("observed_"); direct = f == source or f.startswith("current_")
            records.append({"feature": f, "block": block, "source_variable": source,
                            "semantic_group": semantic, "is_quality_indicator": indicator,
                            "inference_available": True, "is_direct": direct,
                            "derivation_complexity": 0 if direct else 1, "semantic_equivalent": True})
    exclusions = ("next_fruit_count", "target_sn", "target_date", "target_split", "raw_split", "split",
                  "sn", "user_id", "sample_num", "crop_sn", "feature_date", "meas_date", "fruit_count")
    spec = TargetSpec("fruit_set", "next_fruit_count", "current_fruit_count", "feature_date", "target_date",
                      ("user_id", "sample_num", "crop_sn", "fruit_column"),
                      ("user_id", "sample_num", "crop_sn", "fruit_column", "target_date"), exclusions, .95,
                      {k: tuple(v) for k, v in blocks.items()},
                      {"control": "no_control_variable_passed_train_quality_filter"} if not blocks["control"] else {})
    def load_test() -> pd.DataFrame:
        test = pd.read_parquet(ROOT / "processed_data/test_candidates.parquet")
        return _add_sensor_features(test, sensor, selected)
    return SplitBundle(frames["train"], frames["val"], ROOT / "processed_data/test_candidates.parquet",
                       spec, test_loader=load_test), blocks, pd.DataFrame(records)
