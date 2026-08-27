from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRUSS_CODE_MIN = 10000234
TRUSS_CODE_MAX = 10000267
ENTITY = ["facility_id", "crop_sn", "sample_num"]
ENV_SPECS = {
    "indoor_temperature_7d_mean": ("EI", "TI", -10.0, 60.0),
    "indoor_humidity_7d_mean": ("EI", "HI", 0.0, 100.0),
    "indoor_co2_7d_mean": ("EI", "CI", 0.0, 5000.0),
}
CONTROL_CODES = {
    "heating_7d_duty": lambda c: c == "CC0903",
    "co2_control_7d_duty": lambda c: c in {"CC1403", "CC2503"},
    "ventilation_7d_duty": lambda c: c.startswith("CC01_") or c.startswith("CC03_"),
    "shade_curtain_7d_duty": lambda c: c == "CC0403",
}
HISTORY_FEATURES = [
    "current_total_flower_count", "previous_total_flower_count", "total_flower_delta",
    "current_max_truss_number", "positive_truss_count", "observed_truss_count",
    "days_since_previous", "days_since_crop_start", "week_of_year_sin", "week_of_year_cos",
]


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def prepare_growth(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    required = {"facility_id", "crop_sn", "sample_num", "cropping_date", "cropping_end_date", "examin_date",
                "growth_measure_code", "flower_count"}
    _require(raw, required, "growth")
    data = raw.copy()
    for col in ("cropping_date", "cropping_end_date", "examin_date"):
        data[col] = pd.to_datetime(data[col], errors="raise").dt.normalize()
    data["growth_measure_code"] = pd.to_numeric(data.growth_measure_code, errors="coerce")
    data["flower_count"] = pd.to_numeric(data.flower_count, errors="coerce")
    ambiguous_crop_assignments = pd.DataFrame()
    if "source_sn" in data:
        assignment_counts = data.groupby("source_sn", dropna=False).crop_sn.nunique()
        ambiguous_ids = set(assignment_counts[assignment_counts.gt(1)].index)
        ambiguous_crop_assignments = data[data.source_sn.isin(ambiguous_ids)].copy()
        data = data[~data.source_sn.isin(ambiguous_ids)].copy()
    outside_crop = data[~data.examin_date.between(data.cropping_date, data.cropping_end_date)].copy()
    outside_code = data[~data.growth_measure_code.between(TRUSS_CODE_MIN, TRUSS_CODE_MAX)].copy()
    data = data[data.examin_date.between(data.cropping_date, data.cropping_end_date)]
    data = data[data.growth_measure_code.between(TRUSS_CODE_MIN, TRUSS_CODE_MAX)]
    data = data[data.flower_count.notna() & data.flower_count.ge(0)].copy()
    data["truss_number"] = data.growth_measure_code.astype(int) - TRUSS_CODE_MIN + 1

    key = [*ENTITY, "examin_date", "truss_number"]
    variants = data.groupby(key, dropna=False).flower_count.nunique().rename("value_variants").reset_index()
    conflicts = variants[variants.value_variants.gt(1)].copy()
    conflict_dates = conflicts[[*ENTITY, "examin_date"]].drop_duplicates()
    if len(conflict_dates):
        marked = data.merge(conflict_dates.assign(_conflict=True), on=[*ENTITY, "examin_date"], how="left")
        data = marked[marked._conflict.isna()].drop(columns="_conflict")
    data = data.sort_values("source_sn" if "source_sn" in data else "examin_date", kind="stable")
    data = data.drop_duplicates(key + ["flower_count"], keep="last")

    grouped = data.groupby([*ENTITY, "examin_date"], sort=False)
    summaries = grouped.agg(
        current_total_flower_count=("flower_count", "sum"),
        current_max_truss_number=("truss_number", "max"),
        observed_truss_count=("truss_number", "nunique"),
        positive_truss_count=("flower_count", lambda s: int(s.gt(0).sum())),
        cropping_date=("cropping_date", "first"), cropping_end_date=("cropping_end_date", "first"),
    ).reset_index()
    incomplete = summaries[
        summaries.current_max_truss_number.ne(summaries.observed_truss_count)
    ][[*ENTITY, "examin_date", "current_max_truss_number", "observed_truss_count"]].copy()
    if len(incomplete):
        bad = pd.MultiIndex.from_frame(incomplete[[*ENTITY, "examin_date"]])
        idx = pd.MultiIndex.from_frame(summaries[[*ENTITY, "examin_date"]])
        summaries = summaries[~idx.isin(bad)].copy()

    summaries = summaries.sort_values([*ENTITY, "examin_date"], kind="stable").reset_index(drop=True)
    series = summaries.groupby(ENTITY, sort=False, dropna=False)
    summaries["previous_total_flower_count"] = series.current_total_flower_count.shift(1)
    summaries["previous_total_flower_count"] = summaries.previous_total_flower_count.fillna(
        summaries.current_total_flower_count)
    summaries["total_flower_delta"] = (
        summaries.current_total_flower_count - summaries.previous_total_flower_count)
    summaries["days_since_previous"] = series.examin_date.diff().dt.days.fillna(0).astype(float)
    summaries["days_since_crop_start"] = (summaries.examin_date - summaries.cropping_date).dt.days.astype(float)
    week = summaries.examin_date.dt.isocalendar().week.astype(float)
    summaries["week_of_year_sin"] = np.sin(2 * np.pi * week / 52.1775)
    summaries["week_of_year_cos"] = np.cos(2 * np.pi * week / 52.1775)
    summaries["feature_date"] = summaries.examin_date
    summaries["target_date"] = series.examin_date.shift(-1)
    summaries["next_total_flower_count"] = series.current_total_flower_count.shift(-1)
    candidates = summaries[summaries.target_date.notna()].copy().reset_index(drop=True)
    candidates["row_id"] = candidates.apply(
        lambda r: f"{r.facility_id}|{r.crop_sn}|{r.sample_num}|{pd.Timestamp(r.target_date).date()}", axis=1)
    audits = {"ambiguous_crop_assignments": ambiguous_crop_assignments,
              "outside_crop": outside_crop, "outside_truss_1_34": outside_code,
              "conflicting_duplicates": conflicts, "incomplete_truss_dates": incomplete}
    return candidates, audits


def clean_environment(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"facility_id", "sect_code", "fatr_code", "sen_val", "meas_date"}
    _require(raw, required, "environment")
    env = raw.copy()
    env["meas_date"] = pd.to_datetime(env.meas_date, errors="raise")
    logical = ["facility_id", "fld_code", "sect_code", "fatr_code", "maker_id", "sen_id", "meas_date"]
    logical = [c for c in logical if c in env]
    if "source_sn" in env:
        env = env.sort_values("source_sn", kind="stable")
    env = env.drop_duplicates(logical, keep="last")
    env["value"] = pd.to_numeric(env.sen_val, errors="coerce")
    return env[env.value.notna()].sort_values(["facility_id", "meas_date"], kind="stable")


def _window(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    start = end.normalize() - pd.Timedelta(days=7)
    times = frame.meas_date.to_numpy(dtype="datetime64[ns]")
    left = int(np.searchsorted(times, np.datetime64(start), side="left"))
    right = int(np.searchsorted(times, np.datetime64(end.normalize()), side="left"))
    return frame.iloc[left:right]


def add_simple_sensor_features(candidates: pd.DataFrame, raw_env: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = clean_environment(raw_env)
    empty = env.iloc[0:0]
    sensor_streams = {
        key: frame.sort_values("meas_date", kind="stable").reset_index(drop=True)
        for key, frame in env[env.sect_code.eq("EI")].groupby(
            ["facility_id", "sect_code", "fatr_code"], sort=False)
    }
    control_parts = []
    controls = env[env.sect_code.eq("CR")].copy()
    for name, predicate in CONTROL_CODES.items():
        selected = controls[controls.fatr_code.fillna("").astype(str).map(predicate)].copy()
        if selected.empty:
            continue
        selected["control_feature"] = name
        control_parts.append(selected)
    control_streams = {}
    if control_parts:
        combined = pd.concat(control_parts, ignore_index=True)
        combined["active"] = combined.value.gt(0)
        combined = combined.groupby(
            ["facility_id", "control_feature", "meas_date"], as_index=False
        ).active.max().sort_values("meas_date", kind="stable")
        control_streams = {
            key: frame.reset_index(drop=True)
            for key, frame in combined.groupby(["facility_id", "control_feature"], sort=False)
        }
    rows, quality = [], []
    for candidate in candidates.itertuples(index=False):
        values: dict[str, float] = {}
        q = {"row_id": candidate.row_id, "facility_id": candidate.facility_id}
        for name, (sect, code, low, high) in ENV_SPECS.items():
            stream = sensor_streams.get((candidate.facility_id, sect, code), empty)
            selected = _window(stream, candidate.feature_date)
            valid = selected[selected.value.between(low, high)]
            values[name] = float(valid.value.mean()) if len(valid) else np.nan
            q[f"{name}_observations"] = len(valid)
            q[f"{name}_coverage"] = valid.meas_date.dt.normalize().nunique() / 7
            q[f"{name}_outliers"] = len(selected) - len(valid)
        for name in CONTROL_CODES:
            stream = control_streams.get((candidate.facility_id, name), empty)
            selected = _window(stream, candidate.feature_date)
            if len(selected):
                values[name] = float(selected.active.mean())
                q[f"{name}_coverage"] = selected.meas_date.dt.normalize().nunique() / 7
            else:
                values[name] = np.nan
                q[f"{name}_coverage"] = 0.0
        rows.append(values); quality.append(q)
    result = pd.concat([candidates.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return result, pd.DataFrame(quality)


def choose_facility_split(data: pd.DataFrame) -> dict[str, list[str]]:
    counts = data.groupby("facility_id").size().sort_index()
    facilities = list(counts.index)
    if len(facilities) < 6:
        raise ValueError("At least six eligible facilities are required for facility holdout")
    n_val = max(1, round(len(facilities) * .15)); n_test = max(1, round(len(facilities) * .15))
    rng = np.random.default_rng(42)
    best: tuple[Any, ...] | None = None
    trials = min(100_000, max(5_000, len(facilities) * 2_000))
    for _ in range(trials):
        order = list(rng.permutation(facilities))
        val, test = sorted(order[:n_val]), sorted(order[n_val:n_val + n_test])
        val_rows, test_rows = counts[val].sum(), counts[test].sum()
        ratios = ((len(data) - val_rows - test_rows) / len(data), val_rows / len(data), test_rows / len(data))
        score = (max(abs(a-b) for a, b in zip(ratios, (.70, .15, .15))),
                 sum((a-b)**2 for a, b in zip(ratios, (.70, .15, .15))), tuple(val), tuple(test))
        if best is None or score < best[0]:
            best = (score, val, test)
    assert best is not None
    val, test = list(best[1]), list(best[2])
    return {"train": [x for x in facilities if x not in set(val + test)], "validation": val, "test": test}


def split_and_filter_features(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, Any]]:
    partition = choose_facility_split(data)
    frames = {k: data[data.facility_id.isin(v)].copy().reset_index(drop=True) for k, v in partition.items()}
    train = frames["train"]
    env_features = list(ENV_SPECS)
    control_features = []
    control_audit = {}
    for feature in CONTROL_CODES:
        installed = train.loc[train[feature].notna(), "facility_id"].nunique() / train.facility_id.nunique()
        missing = train[feature].isna().mean()
        passed = installed >= .50 and missing <= .50
        control_audit[feature] = {"installed_facility_ratio": installed, "train_missing_rate": missing, "selected": passed}
        if passed:
            control_features.append(feature)
    medians = {}
    for feature in env_features + control_features:
        median = float(train[feature].median()) if train[feature].notna().any() else 0.0
        medians[feature] = median
        for frame in frames.values():
            frame[feature] = frame[feature].fillna(median)
    sets = {
        "history": HISTORY_FEATURES,
        "history+environment": HISTORY_FEATURES + env_features,
        "history+control": HISTORY_FEATURES + control_features,
        "history+environment+control": HISTORY_FEATURES + env_features + control_features,
    }
    # Identical sets are aliases, not separate statistical comparisons.
    unique, seen = {}, set()
    for name, features in sets.items():
        signature = tuple(features)
        if signature not in seen:
            unique[name] = features; seen.add(signature)
    return frames, unique, {"partition": partition, "train_medians": medians, "control_quality": control_audit}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

