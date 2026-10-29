from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import digest, file_hash, growth_features
from ..exhaustive.config import PARTS, TARGETS
from ..exhaustive.io import atomic_json
from ..exhaustive.prepare import ENTITY, GROWTH_MAP, _strawberry, _tomato
from .config import V1_ROOT


def _row_hash(frame: pd.DataFrame) -> str:
    return digest(sorted(frame.row_id.astype(str).tolist()))


def _verify_v1(v1_root: Path):
    snapshot_path = v1_root / "data_snapshot" / "manifest.json"
    prepared_path = v1_root / "prepared_manifest.json"
    if not snapshot_path.exists() or not prepared_path.exists():
        raise FileNotFoundError("Frozen v1 snapshot/prepared manifest is missing")
    snapshot = json.loads(snapshot_path.read_text())
    for table, item in snapshot["tables"].items():
        path = v1_root / "data_snapshot" / item["file"]
        if not path.exists() or file_hash(path) != item["sha256"]:
            raise ValueError(f"Frozen v1 snapshot changed: {table}")
    return file_hash(snapshot_path), file_hash(prepared_path)


def _derive_crop(v1_root: Path, crop: str):
    loader = _tomato if crop == "tomato" else _strawberry
    rows, source_audit = loader(v1_root)
    rows["feature_date"] = pd.to_datetime(rows.feature_date).dt.normalize()
    rows = rows.sort_values(ENTITY + ["feature_date"]).copy()
    rows["count_sum123"] = rows[["count_1", "count_2", "count_3"]].sum(
        axis=1, min_count=3
    )
    grouped = rows.groupby(ENTITY, dropna=False)
    rows["target_date"] = grouped.feature_date.shift(-1)
    for number in (1, 2, 3):
        rows[f"target_truss_{number}"] = grouped[f"count_{number}"].shift(-1)

    derived = {}
    for part, component in PARTS.items():
        target_id = f"{crop}_{part}"
        frame = rows.copy()
        frame["target"] = (
            frame[[f"target_truss_{i}" for i in (1, 2, 3)]].sum(axis=1, min_count=3)
            if component is None
            else frame[f"target_truss_{component}"]
        )
        frame = frame[frame.target_date.notna() & frame.target.notna()].copy()
        e1_bases = ["count_1", "count_2", "count_3"] + [
            column
            for column in (
                "current_max_truss_number",
                "observed_truss_count",
                "positive_truss_count",
            )
            if column in frame
        ]
        frame = growth_features(frame, ENTITY, e1_bases)
        e1 = sorted(
            column
            for column in frame
            if any(column.startswith(base + "__") for base in e1_bases)
        )
        e2_bases = [alias for alias in GROWTH_MAP.values() if alias in frame]
        if e2_bases:
            frame = growth_features(frame, ENTITY, e2_bases)
        e2 = sorted(
            column
            for column in frame
            if any(column.startswith(base + "__") for base in e2_bases)
        )
        frame["row_id"] = [
            digest([target_id, *values])[:24]
            for values in frame[
                ENTITY + ["feature_date", "target_date"]
            ].itertuples(index=False, name=None)
        ]
        identity = [
            "row_id",
            *ENTITY,
            "feature_date",
            "target_date",
            "target",
            "target_truss_1",
            "target_truss_2",
            "target_truss_3",
        ]
        derived[target_id] = {
            "frame": frame[identity + e1 + e2].copy(),
            "e1": e1,
            "e2": e2,
            "e1_bases": e1_bases,
            "e2_bases": e2_bases,
        }
    return derived, source_audit


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(left) or pd.api.types.is_datetime64_any_dtype(right):
        a_dates = pd.to_datetime(left, errors="coerce").to_numpy(dtype="datetime64[ns]")
        b_dates = pd.to_datetime(right, errors="coerce").to_numpy(dtype="datetime64[ns]")
        return bool(np.array_equal(a_dates, b_dates, equal_nan=True))
    a_numeric = pd.to_numeric(left, errors="coerce")
    b_numeric = pd.to_numeric(right, errors="coerce")
    if int(a_numeric.notna().sum()) == int(left.notna().sum()) and int(b_numeric.notna().sum()) == int(right.notna().sum()):
        return bool(np.allclose(a_numeric.to_numpy(float), b_numeric.to_numpy(float), equal_nan=True, rtol=0, atol=0))
    a_text = left.where(left.notna(), "__NULL__").astype(str).to_numpy()
    b_text = right.where(right.notna(), "__NULL__").astype(str).to_numpy()
    return bool(np.array_equal(a_text, b_text))


def prepare(root: Path, v1_root: Path = V1_ROOT):
    root, v1_root = Path(root), Path(v1_root)
    manifest_path = root / "prepared_manifest.json"
    prepared = root / "prepared"
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text())
        for target, spec in result["targets"].items():
            for split, sha in spec["dataset_hashes"].items():
                if file_hash(prepared / target / f"{split}.parquet") != sha:
                    raise ValueError("E1/E2 prepared dataset changed")
        return result

    root.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    snapshot_hash, v1_prepared_hash = _verify_v1(v1_root)
    all_derived = {}
    source_audits = {}
    for crop in ("tomato", "strawberry"):
        crop_derived, source_audit = _derive_crop(v1_root, crop)
        all_derived.update(crop_derived)
        source_audits[crop] = source_audit

    target_specs, row_audit = {}, []
    for target in TARGETS:
        source_dir = v1_root / "prepared" / target
        old_spec = json.loads((source_dir / "manifest.json").read_text())
        details = all_derived[target]
        extra = details["frame"]
        if extra.row_id.duplicated().any():
            raise ValueError(f"Duplicate E1/E2 row IDs: {target}")
        destination = prepared / target
        destination.mkdir(parents=True, exist_ok=True)
        hashes, row_hashes = {}, {}
        for split in ("train", "validation", "test"):
            source_path = source_dir / f"{split}.parquet"
            if file_hash(source_path) != old_spec["dataset_hashes"][split]:
                raise ValueError(f"v1 prepared dataset changed: {target}/{split}")
            source = pd.read_parquet(source_path)
            before_hash = _row_hash(source)
            matched = source.merge(extra, on="row_id", how="left", validate="one_to_one", suffixes=("", "__audit"))
            if len(matched) != len(source) or matched[details["e1"]].isna().all(axis=1).all():
                raise ValueError(f"E1/E2 rows did not attach correctly: {target}/{split}")
            for column in [*ENTITY, "feature_date", "target_date", "target", "target_truss_1", "target_truss_2", "target_truss_3"]:
                audit_column = column + "__audit"
                if audit_column in matched and not _same_values(matched[column], matched[audit_column]):
                    raise ValueError(f"Source identity/target mismatch: {target}/{split}/{column}")
            drop_audit = [column for column in matched if column.endswith("__audit")]
            frame = matched.drop(columns=drop_audit).sort_values("row_id").reset_index(drop=True)
            after_hash = _row_hash(frame)
            if before_hash != after_hash:
                raise ValueError(f"v1 row identity changed: {target}/{split}")
            output_path = destination / f"{split}.parquet"
            frame.to_parquet(output_path, index=False)
            hashes[split] = file_hash(output_path)
            row_hashes[split] = after_hash
            row_audit.append({
                "Target_ID": target,
                "Split": split,
                "Rows": len(frame),
                "V1_Row_Hash": before_hash,
                "Backfill_Row_Hash": after_hash,
                "Match": True,
            })

        groups = {key: list(value) for key, value in old_spec["groups"].items()}
        groups["E1"] = details["e1"]
        groups["E2"] = details["e2"]
        unavailable = [
            group
            for group in old_spec.get("unavailable_groups", [])
            if group not in {"E1", "E2"}
        ]
        if not details["e1"]:
            unavailable.append("E1")
        if not details["e2"]:
            unavailable.append("E2")
        spec = {
            **old_spec,
            "groups": groups,
            "numeric": sorted(set(old_spec["numeric"]) | set(details["e1"]) | set(details["e2"])),
            "unavailable_groups": sorted(set(unavailable)),
            "dataset_hashes": hashes,
            "row_hashes": row_hashes,
            "source_v1_dataset_hashes": old_spec["dataset_hashes"],
            "source_v1_manifest_hash": file_hash(source_dir / "manifest.json"),
            "e1_bases": details["e1_bases"],
            "e2_bases": details["e2_bases"],
            "e1_e2_backfill_version": "1.0",
        }
        atomic_json(destination / "manifest.json", spec)
        target_specs[target] = spec

    pd.DataFrame(row_audit).to_csv(root / "row_identity_audit.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(v1_root / "data_quality.csv", root / "data_quality.csv")
    result = {
        "targets": target_specs,
        "target_count": len(target_specs),
        "v1_root": str(v1_root.resolve()),
        "v1_snapshot_manifest_hash": snapshot_hash,
        "v1_prepared_manifest_hash": v1_prepared_hash,
        "source_audits": source_audits,
    }
    atomic_json(manifest_path, result)
    return result
