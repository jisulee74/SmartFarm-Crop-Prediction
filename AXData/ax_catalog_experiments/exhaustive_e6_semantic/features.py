from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import WINDOW_DAYS
from .mapping import validate_observed_codes

RAW_COLUMNS = ["facility_id", "sect_code", "fatr_code", "sen_val", "meas_date"]


def _partition_name(facility_id: str) -> str:
    return hashlib.sha256(str(facility_id).encode()).hexdigest()[:20] + ".parquet"


def partition_semantic_streams(snapshot_file: Path, output_dir: Path, mapping: pd.DataFrame):
    """Stream the large snapshot once and atomically partition included E6 rows."""
    output_dir = Path(output_dir)
    success = output_dir / "_SUCCESS.json"
    if success.exists():
        return json.loads(success.read_text())
    partial = output_dir.with_name(output_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    lookup = mapping.set_index("source_code").to_dict("index")
    observed, writers, counts, index_rows = set(), {}, {}, {}
    schema = pa.schema(
        [
            ("facility_id", pa.string()),
            ("source_code", pa.string()),
            ("canonical_code", pa.string()),
            ("device_family", pa.string()),
            ("device_family_kr", pa.string()),
            ("measurement_type", pa.string()),
            ("measurement_type_kr", pa.string()),
            ("is_binary", pa.bool_()),
            ("sensor_value", pa.float64()),
            ("meas_date", pa.timestamp("ns")),
        ]
    )
    try:
        source = pq.ParquetFile(snapshot_file)
        for batch in source.iter_batches(columns=RAW_COLUMNS, batch_size=250_000):
            frame = batch.to_pandas()
            frame = frame.loc[frame.sect_code.astype(str).eq("CR")].copy()
            if frame.empty:
                continue
            frame["source_code"] = frame.fatr_code.astype(str)
            observed.update(frame.source_code.unique())
            frame = frame.loc[frame.source_code.isin(lookup)].copy()
            frame = frame.join(mapping.set_index("source_code"), on="source_code")
            frame = frame.loc[frame.included.astype(bool)].copy()
            if frame.empty:
                continue
            frame["sensor_value"] = pd.to_numeric(frame.sen_val, errors="coerce")
            frame["meas_date"] = pd.to_datetime(frame.meas_date, errors="coerce")
            frame = frame.loc[frame.sensor_value.notna() & frame.meas_date.notna()]
            for facility, part in frame.groupby(frame.facility_id.astype(str), sort=False):
                name = _partition_name(facility)
                path = partial / name
                if facility not in writers:
                    writers[facility] = pq.ParquetWriter(path, schema, compression="zstd")
                    index_rows[facility] = name
                    counts[facility] = 0
                values = part[
                    [
                        "source_code", "canonical_code", "device_family",
                        "device_family_kr", "measurement_type",
                        "measurement_type_kr", "is_binary", "sensor_value", "meas_date",
                    ]
                ].copy()
                values.insert(0, "facility_id", str(facility))
                writers[facility].write_table(
                    pa.Table.from_pandas(values, schema=schema, preserve_index=False)
                )
                counts[facility] += len(values)
        validate_observed_codes(observed, mapping)
    finally:
        for writer in writers.values():
            writer.close()
    pd.DataFrame(
        [
            {"facility_id": facility, "file": index_rows[facility], "rows": counts[facility]}
            for facility in sorted(index_rows)
        ]
    ).to_csv(partial / "partition_index.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "observed_codes": sorted(observed),
        "facility_count": len(index_rows),
        "included_rows": int(sum(counts.values())),
    }
    (partial / "_SUCCESS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    partial.replace(output_dir)
    return manifest


def _feature_names(device: str, measurement: str, days: int):
    prefix = f"E6S__{device}__{measurement}__{days}d"
    if measurement == "opening_percent":
        stats = ("facility_mean", "overall_max", "overall_min", "within_device_std")
    elif measurement in {"binary_status", "binary_onoff"}:
        stats = ("active_fraction_mean", "active_fraction_max")
    else:
        stats = ("mean", "std", "min", "max")
    return [f"{prefix}__{s}" for s in (*stats, "reporting_device_count", "connection_rate")]


def semantic_feature_catalog(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    included = mapping.loc[mapping.included.astype(bool)].drop_duplicates(
        ["device_family", "measurement_type"]
    )
    for item in included.itertuples(index=False):
        for days in WINDOW_DAYS:
            for name in _feature_names(item.device_family, item.measurement_type, days):
                rows.append(
                    {
                        "Variable_ID": name,
                        "Group": "E6",
                        "Device_Family": item.device_family,
                        "Device_Family_KR": item.device_family_kr,
                        "Measurement_Type": item.measurement_type,
                        "Measurement_Type_KR": item.measurement_type_kr,
                        "Window_Days": days,
                        "Statistic": name.rsplit("__", 1)[-1],
                    }
                )
    return pd.DataFrame(rows)


def semantic_features_for_facility(rows: pd.DataFrame, stream: pd.DataFrame) -> pd.DataFrame:
    """Create leak-free semantic features for one facility's prediction dates."""
    rows = rows[["facility_id", "feature_date"]].drop_duplicates().copy()
    rows["feature_date"] = pd.to_datetime(rows.feature_date)
    stream = stream.copy()
    stream["meas_date"] = pd.to_datetime(stream.meas_date)
    stream["sensor_value"] = pd.to_numeric(stream.sensor_value, errors="coerce")
    stream = stream.loc[stream.sensor_value.notna() & stream.meas_date.notna()]
    groups = {}
    for (device, measurement), semantic_part in stream.groupby(
        ["device_family", "measurement_type"], sort=True
    ):
        members = {}
        for code, member in semantic_part.groupby("canonical_code", sort=True):
            member = member.sort_values("meas_date")
            members[str(code)] = (
                member.meas_date.to_numpy("datetime64[ns]"),
                member.sensor_value.to_numpy(float),
            )
        groups[(str(device), str(measurement))] = members
    records = []
    for row in rows.itertuples(index=False):
        end = np.datetime64(row.feature_date)
        record = {"facility_id": str(row.facility_id), "feature_date": row.feature_date}
        for (device, measurement), members in groups.items():
            for days in WINDOW_DAYS:
                prefix = f"E6S__{device}__{measurement}__{days}d"
                start = np.datetime64(row.feature_date - pd.Timedelta(days=days))
                values_by_member = []
                known = 0
                for times, values in members.values():
                    if len(times) and times[0] < end:
                        known += 1
                    left = np.searchsorted(times, start, side="left")
                    right = np.searchsorted(times, end, side="left")
                    x = values[left:right]
                    x = x[np.isfinite(x)]
                    if measurement in {"binary_status", "binary_onoff"}:
                        x = x[np.isin(x, [0, 1])]
                    if len(x):
                        values_by_member.append(x)
                reporting = len(values_by_member)
                if measurement == "opening_percent":
                    record[prefix + "__facility_mean"] = (
                        float(np.mean([x.mean() for x in values_by_member])) if reporting else np.nan
                    )
                    record[prefix + "__overall_max"] = (
                        float(max(x.max() for x in values_by_member)) if reporting else np.nan
                    )
                    record[prefix + "__overall_min"] = (
                        float(min(x.min() for x in values_by_member)) if reporting else np.nan
                    )
                    record[prefix + "__within_device_std"] = (
                        float(np.mean([x.std(ddof=0) for x in values_by_member])) if reporting else np.nan
                    )
                elif measurement in {"binary_status", "binary_onoff"}:
                    fractions = [float(np.mean(x == 1)) for x in values_by_member]
                    record[prefix + "__active_fraction_mean"] = float(np.mean(fractions)) if fractions else np.nan
                    record[prefix + "__active_fraction_max"] = float(np.max(fractions)) if fractions else np.nan
                else:
                    pooled = np.concatenate(values_by_member) if reporting else np.array([])
                    for stat in ("mean", "std", "min", "max"):
                        record[prefix + "__" + stat] = (
                            float(getattr(np, stat)(pooled)) if len(pooled) else np.nan
                        )
                record[prefix + "__reporting_device_count"] = int(reporting)
                record[prefix + "__connection_rate"] = (
                    float(reporting / known) if known else np.nan
                )
        records.append(record)
    return pd.DataFrame(records)


def build_all_features(keys: pd.DataFrame, stream_dir: Path):
    index = pd.read_csv(Path(stream_dir) / "partition_index.csv", dtype={"facility_id": str})
    files = dict(zip(index.facility_id, index.file))
    outputs, audit = [], []
    for facility, rows in keys.groupby(keys.facility_id.astype(str), sort=True):
        path = Path(stream_dir) / files[facility] if facility in files else None
        stream = pd.read_parquet(path) if path and path.exists() else pd.DataFrame(
            columns=["canonical_code", "device_family", "measurement_type", "sensor_value", "meas_date"]
        )
        if len(stream):
            stream["period"] = pd.to_datetime(stream.meas_date).dt.year
            for (period, device, measurement), part in stream.groupby(
                ["period", "device_family", "measurement_type"], dropna=False
            ):
                audit.append(
                    {
                        "Facility_ID": facility,
                        "Period": int(period),
                        "Device_Family": device,
                        "Measurement_Type": measurement,
                        "Source_Codes": "|".join(sorted(part.source_code.astype(str).unique())),
                        "Canonical_Device_Count": int(part.canonical_code.nunique()),
                        "Observation_Count": int(len(part)),
                        "Min_Date": str(part.meas_date.min()),
                        "Max_Date": str(part.meas_date.max()),
                    }
                )
        outputs.append(semantic_features_for_facility(rows, stream))
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audit)
