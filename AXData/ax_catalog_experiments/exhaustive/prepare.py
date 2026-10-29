from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ..catalog import AX
from ..core import digest, file_hash, growth_features, sensor_features
from .config import COMBINABLE_GROUPS, PARTS
from .io import atomic_json

ENTITY = ["facility_id", "crop_sn", "sample_num"]
GROWTH_MAP = {
    "leaves_length": "leaf_len",
    "leaves_num": "leaf_cnt",
    "leaves_width": "leaf_wdth",
    "petiole_length": "ptl_len",
    "stem_diameter": "stem_dmtr",
    "theca_diameter": "crwn_dmtr",
    "flower_length": "plnt_hgt",
    "grow_length": "grw_len",
    "flower_top": "flclstr_hgt",
}
CONTEXT = [
    "cropping_system",
    "cultivation_area",
    "cal_cultivation_area",
    "plant_num",
    "cal_plant_num",
    "plant_density",
    "cropping_season_name",
]


def _read_snapshot(root, table):
    return pd.read_parquet(Path(root) / "data_snapshot" / f"{table}.parquet")


def _facility_map(ids: pd.DataFrame) -> dict:
    user_candidates = [os.getenv("FARMSTOM_ID_USER_COLUMN"), "user_id", "id"]
    facility_candidates = [
        os.getenv("FARMSTOM_ID_FACILITY_COLUMN"),
        "facility_id",
        "fclt_id",
        "farm_id",
    ]
    user = next((c for c in user_candidates if c and c in ids), None)
    facility = next((c for c in facility_candidates if c and c in ids), None)
    if not user or not facility:
        raise ValueError("sfkr_pvsn_id must contain user_id and facility_id columns")
    pairs = ids[[user, facility]].dropna().astype(str).drop_duplicates()
    if pairs.groupby(user)[facility].nunique().gt(1).any():
        raise ValueError("sfkr_pvsn_id has ambiguous user-to-facility mappings")
    return dict(pairs.itertuples(index=False, name=None))


def _assign_crop(rows, crops, user_column, date_column):
    rows = rows.copy()
    crops = crops.copy()
    rows[date_column] = pd.to_datetime(
        rows[date_column], errors="coerce"
    ).dt.normalize()
    crops["cropping_date"] = pd.to_datetime(
        crops.cropping_date, errors="coerce"
    ).dt.normalize()
    crops["cropping_end_date"] = pd.to_datetime(
        crops.cropping_end_date, errors="coerce"
    ).dt.normalize()
    crops["cropping_end_date"] = crops.cropping_end_date.fillna(
        pd.Timestamp.max.normalize()
    )
    left = rows.reset_index(names="_source_index")
    crops = (
        crops.rename(columns={"sn": "crop_sn"})
        if "sn" in crops and "crop_sn" not in crops
        else crops
    )
    merged = left.merge(
        crops,
        left_on=[user_column, "item_code"],
        right_on=["user_id", "item_code"],
        how="left",
        suffixes=("", "_crop"),
    )
    valid = merged[date_column].between(merged.cropping_date, merged.cropping_end_date)
    merged = merged[valid]
    counts = merged.groupby("_source_index").size()
    ambiguous = set(counts[counts != 1].index)
    merged = merged[~merged._source_index.isin(ambiguous)].drop_duplicates(
        "_source_index"
    )
    return merged, {"unmatched_or_ambiguous_crop_rows": int(len(rows) - len(merged))}


def _tomato(root):
    raw = _read_snapshot(root, "sfkr_hbfm_grow")
    crops = _read_snapshot(root, "sfkr_pvsn_crop")
    from AXData.tomato_flower_count_prediction.extract import _value_column

    value = _value_column(set(raw.columns), os.getenv("FARMSTOM_GROWTH_VALUE_COLUMN"))
    required = {
        "facility_id",
        "item_code",
        "examin_date",
        "sample_number",
        "growth_measure_code",
        value,
    }
    if required - set(raw):
        raise ValueError(
            f"Tomato source columns missing: {sorted(required - set(raw))}"
        )
    d = raw[raw.item_code.astype(str).eq("080300")].copy()
    d["_join_user"] = d.facility_id.astype(str).str.rsplit("_", n=1).str[0]
    d, audit = _assign_crop(d, crops, "_join_user", "examin_date")
    d["truss"] = pd.to_numeric(d.growth_measure_code, errors="coerce") - 10000234 + 1
    d["value"] = pd.to_numeric(d[value], errors="coerce")
    d = d[d.truss.between(1, 34) & d.value.ge(0)].copy()
    d = d.rename(
        columns={
            "examin_date": "feature_date",
            "sample_number": "sample_num",
            "crop_sn_crop": "crop_sn",
        }
    )
    d["feature_date"] = pd.to_datetime(d.feature_date).dt.normalize()
    d["sample_num"] = d.sample_num.astype(str)
    if "crop_sn" not in d:
        raise ValueError("Crop-cycle primary key did not join")
    keys = ENTITY + ["feature_date"]
    conflict = d.groupby(keys + ["truss"]).value.nunique().gt(1)
    bad = conflict[conflict].reset_index()[keys].drop_duplicates()
    d = d.merge(bad.assign(_bad=True), on=keys, how="left")
    d = d[d._bad.isna()].drop_duplicates(keys + ["truss"])
    wide = (
        d[d.truss.isin([1, 2, 3])]
        .pivot(index=keys, columns="truss", values="value")
        .reindex(columns=[1, 2, 3])
    )
    wide.columns = [f"count_{i}" for i in (1, 2, 3)]
    rows = (
        d.groupby(keys)
        .agg(
            cropping_date=("cropping_date", "first"),
            current_max_truss_number=("truss", "max"),
            observed_truss_count=("truss", "nunique"),
            positive_truss_count=("value", lambda x: int(x.gt(0).sum())),
        )
        .reset_index()
    )
    rows = rows.merge(wide.reset_index(), on=keys, how="left", validate="one_to_one")
    for column in CONTEXT:
        if column in d:
            rows = rows.merge(
                d[keys + [column]].drop_duplicates(keys),
                on=keys,
                how="left",
                validate="one_to_one",
            )
    # Optional vegetative survey rows from the shared growth table enrich E2 on matching dates.
    survey = _read_snapshot(root, "sfkr_pvsn_grow")
    if {"user_id", "item_code", "sample_num", "meas_date"} <= set(survey):
        survey = survey[survey.item_code.astype(str).eq("080300")].copy()
        survey, survey_audit = _assign_crop(survey, crops, "user_id", "meas_date")
        ids = _read_snapshot(root, "sfkr_pvsn_id")
        facility_map = _facility_map(ids)
        survey["facility_id"] = survey.user_id.astype(str).map(facility_map)
        survey = survey[survey.facility_id.notna()].copy()
        survey = survey.rename(columns={"meas_date": "feature_date"})
        survey["feature_date"] = pd.to_datetime(survey.feature_date).dt.normalize()
        survey["sample_num"] = survey.sample_num.astype(str)
        values = survey[keys].copy()
        for source, alias in GROWTH_MAP.items():
            if source in survey:
                values[alias] = pd.to_numeric(survey[source], errors="coerce")
        if len(values.columns) > len(keys):
            values = values.groupby(keys, dropna=False).first().reset_index()
            rows = rows.merge(values, on=keys, how="left", validate="one_to_one")
        audit["tomato_vegetative_unmatched_or_ambiguous"] = survey_audit[
            "unmatched_or_ambiguous_crop_rows"
        ]
    audit["conflicting_observation_keys"] = len(bad)
    return rows, audit


def _strawberry(root):
    raw = _read_snapshot(root, "sfkr_pvsn_grow")
    crops = _read_snapshot(root, "sfkr_pvsn_crop")
    ids = _read_snapshot(root, "sfkr_pvsn_id")
    required = {
        "user_id",
        "item_code",
        "sample_num",
        "meas_date",
        "first_fruits_num",
        "second_fruits_num",
        "third_fruits_num",
    }
    if required - set(raw):
        raise ValueError(
            f"Strawberry source columns missing: {sorted(required - set(raw))}"
        )
    d = raw[raw.item_code.astype(str).eq("080400")].copy()
    d, audit = _assign_crop(d, crops, "user_id", "meas_date")
    mapping = _facility_map(ids)
    d["facility_id"] = d.user_id.astype(str).map(mapping).fillna(d.user_id.astype(str))
    d = d.rename(
        columns={
            "meas_date": "feature_date",
            "crop_sn_crop": "crop_sn",
            "first_fruits_num": "count_1",
            "second_fruits_num": "count_2",
            "third_fruits_num": "count_3",
        }
    )
    if "crop_sn" not in d:
        raise ValueError("Crop-cycle primary key did not join")
    for c in ["count_1", "count_2", "count_3"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").where(lambda x: x.ge(0))
    keys = ENTITY + ["feature_date"]
    targets = d[
        keys
        + ["count_1", "count_2", "count_3", "cropping_date"]
        + [c for c in CONTEXT if c in d]
    ].copy()
    conflicts = (
        targets.groupby(keys)[["count_1", "count_2", "count_3"]]
        .nunique()
        .max(axis=1)
        .gt(1)
    )
    bad = conflicts[conflicts].reset_index()[keys]
    targets = targets.merge(bad.assign(_bad=True), on=keys, how="left")
    targets = targets[targets._bad.isna()].drop_duplicates(keys).drop(columns="_bad")
    for source, alias in GROWTH_MAP.items():
        if source in d:
            values = d[keys + [source]].copy()
            values[alias] = pd.to_numeric(values[source], errors="coerce")
            values = values.drop(columns=source).drop_duplicates(keys)
            if not values.duplicated(keys).any():
                targets = targets.merge(
                    values, on=keys, how="left", validate="one_to_one"
                )
    audit["conflicting_observation_keys"] = len(bad)
    return targets, audit


def _add_sensors(root, rows):
    raw = _read_snapshot(root, "sfkr_hbfm_env_con")
    required = {"facility_id", "sect_code", "fatr_code", "sen_val", "meas_date"}
    if required - set(raw):
        raise ValueError(
            f"Sensor source columns missing: {sorted(required - set(raw))}"
        )
    raw = raw.copy()
    raw["facility_id"] = raw.facility_id.astype(str)
    raw["variable_id"] = raw.sect_code.astype(str) + "__" + raw.fatr_code.astype(str)
    raw["sensor_value"] = pd.to_numeric(raw.sen_val, errors="coerce")
    definitions = {}
    for variable, part in raw.groupby("variable_id"):
        values = part.sensor_value.dropna().unique()
        definitions[variable] = bool(len(values) and set(values).issubset({0, 1}))
    enriched = sensor_features(rows, raw, definitions)
    groups = {g: [] for g in COMBINABLE_GROUPS}
    for variable, binary in definitions.items():
        section = variable.split("__", 1)[0]
        group = {"EI": "E3", "EO": "E4", "NT": "E5", "CR": "E6"}.get(section)
        if not group:
            continue
        suffixes = ("active_fraction",) if binary else ("mean", "std", "min", "max")
        groups[group].extend(
            f"{variable}__{days}d_{stat}" for days in (1, 3, 7) for stat in suffixes
        )
    return enriched, groups


def _historical_split(crop):
    target = "tomato_first" if crop == "tomato" else "strawberry_first"
    path = (
        AX
        / "ax_catalog_experiments"
        / "outputs"
        / "stability_v2"
        / "prepared"
        / target
        / "manifest.json"
    )
    if not path.exists():
        raise ValueError(f"Historical fixed facility split missing: {path}")
    return json.loads(path.read_text())["split"]


def prepare(root: Path):
    root = Path(root)
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    if (root / "prepared_manifest.json").exists():
        result = json.loads((root / "prepared_manifest.json").read_text())
        for target, spec in result["targets"].items():
            for split, sha in spec["dataset_hashes"].items():
                if file_hash(prepared / target / f"{split}.parquet") != sha:
                    raise ValueError("Prepared dataset changed")
        return result
    target_specs, quality = {}, []
    for crop, loader in (("tomato", _tomato), ("strawberry", _strawberry)):
        rows, audit = loader(root)
        rows["feature_date"] = pd.to_datetime(rows.feature_date).dt.normalize()
        rows, sensor_groups = _add_sensors(root, rows)
        split = _historical_split(crop)
        assigned = {str(f): name for name, fs in split.items() for f in fs}
        rows["split"] = rows.facility_id.astype(str).map(assigned)
        quality.append(
            {
                "Crop": crop,
                "Reason": "unassigned_facility_rows",
                "Rows": int(rows.split.isna().sum()),
            }
        )
        rows = rows[rows.split.notna()].sort_values(ENTITY + ["feature_date"]).copy()
        rows["count_sum123"] = rows[["count_1", "count_2", "count_3"]].sum(
            axis=1, min_count=3
        )
        grouped = rows.groupby(ENTITY, dropna=False)
        rows["target_date"] = grouped.feature_date.shift(-1)
        for i in (1, 2, 3):
            rows[f"target_truss_{i}"] = grouped[f"count_{i}"].shift(-1)
        for part, component in PARTS.items():
            target = f"{crop}_{part}"
            frame = rows.copy()
            frame["target"] = (
                frame[[f"target_truss_{i}" for i in (1, 2, 3)]].sum(axis=1, min_count=3)
                if component is None
                else frame[f"target_truss_{component}"]
            )
            frame = frame[frame.target_date.notna() & frame.target.notna()].copy()
            bases = ["count_1", "count_2", "count_3"] + [
                c
                for c in [
                    "current_max_truss_number",
                    "observed_truss_count",
                    "positive_truss_count",
                ]
                if c in frame
            ]
            frame = growth_features(frame, ENTITY, bases)
            if "cropping_date" in frame:
                frame["days_since_crop_start"] = (
                    frame.feature_date - pd.to_datetime(frame.cropping_date)
                ).dt.days
            e1 = sorted(
                c for c in frame if any(c.startswith(base + "__") for base in bases)
            )
            e2_bases = [c for c in GROWTH_MAP.values() if c in frame]
            if e2_bases:
                frame = growth_features(frame, ENTITY, e2_bases)
            e2 = sorted(
                c for c in frame if any(c.startswith(base + "__") for base in e2_bases)
            )
            e7 = [c for c in CONTEXT + ["days_since_crop_start"] if c in frame]
            for c in [
                "cultivation_area",
                "cal_cultivation_area",
                "plant_num",
                "cal_plant_num",
                "plant_density",
                "days_since_crop_start",
            ]:
                if c in frame:
                    frame[c] = pd.to_numeric(frame[c], errors="coerce")
            groups = {"E1": e1, "E2": e2, **sensor_groups, "E7": e7}
            unavailable = [g for g in COMBINABLE_GROUPS if not groups.get(g)]
            categorical = [c for c in e7 if not pd.api.types.is_numeric_dtype(frame[c])]
            numeric = sorted(
                {c for fs in groups.values() for c in fs if c not in categorical}
            )
            columns = list(
                dict.fromkeys(
                    [
                        "row_id",
                        *ENTITY,
                        "feature_date",
                        "target_date",
                        "target",
                        "target_truss_1",
                        "target_truss_2",
                        "target_truss_3",
                        *numeric,
                        *categorical,
                    ]
                )
            )
            frame["row_id"] = [
                digest([target, *v])[:24]
                for v in frame[ENTITY + ["feature_date", "target_date"]].itertuples(
                    index=False, name=None
                )
            ]
            destination = prepared / target
            destination.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for split_name in ("train", "validation", "test"):
                part_frame = (
                    frame.loc[frame.split.eq(split_name), columns]
                    .sort_values("row_id")
                    .reset_index(drop=True)
                )
                if part_frame.empty:
                    raise ValueError(f"{target}: empty {split_name} split")
                path = destination / f"{split_name}.parquet"
                part_frame.to_parquet(path, index=False)
                hashes[split_name] = file_hash(path)
                quality.append(
                    {
                        "Crop": crop,
                        "Target_ID": target,
                        "Split": split_name,
                        "Reason": "target_distribution",
                        "Rows": len(part_frame),
                        "Zero_Rate": float(part_frame.target.eq(0).mean()),
                        "Target_Mean": float(part_frame.target.mean()),
                        "Target_Max": float(part_frame.target.max()),
                    }
                )
            spec = {
                "target_id": target,
                "crop": crop,
                "part": part,
                "components": [1, 2, 3] if component is None else [component],
                "entity": ENTITY,
                "groups": groups,
                "numeric": numeric,
                "categorical": categorical,
                "unavailable_groups": unavailable,
                "split": split,
                "dataset_hashes": hashes,
            }
            atomic_json(destination / "manifest.json", spec)
            target_specs[target] = spec
        quality.extend({"Crop": crop, "Reason": k, "Rows": v} for k, v in audit.items())
    pd.DataFrame(quality).to_csv(
        root / "data_quality.csv", index=False, encoding="utf-8-sig"
    )
    result = {"targets": target_specs, "target_count": len(target_specs)}
    atomic_json(root / "prepared_manifest.json", result)
    return result
