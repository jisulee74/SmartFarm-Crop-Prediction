from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import digest, file_hash
from ..exhaustive.config import TARGETS
from ..exhaustive.engine import temporal_folds
from ..exhaustive.io import atomic_json
from .config import REFERENCE_FILE, V1_ROOT
from .features import (
    build_all_features,
    partition_semantic_streams,
    semantic_feature_catalog,
)
from .mapping import load_mapping


def _verify_v1_snapshot(v1_root: Path):
    manifest_path = v1_root / "data_snapshot" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"v1 snapshot manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for table, item in manifest["tables"].items():
        path = v1_root / "data_snapshot" / item["file"]
        if not path.exists() or file_hash(path) != item["sha256"]:
            raise ValueError(f"v1 snapshot file changed: {path}")
    return manifest, file_hash(manifest_path)


def _row_hash(frame: pd.DataFrame) -> str:
    return digest(sorted(frame.row_id.astype(str).tolist()))


def _source_keys(v1_root: Path):
    frames = []
    row_audit = []
    for target in TARGETS:
        source = v1_root / "prepared" / target
        spec = json.loads((source / "manifest.json").read_text())
        for split in ("train", "validation", "test"):
            path = source / f"{split}.parquet"
            if file_hash(path) != spec["dataset_hashes"][split]:
                raise ValueError(f"v1 prepared dataset changed: {path}")
            frame = pd.read_parquet(path, columns=["row_id", "facility_id", "feature_date"])
            frames.append(frame[["facility_id", "feature_date"]])
            row_audit.append(
                {
                    "Target_ID": target,
                    "Split": split,
                    "Rows": len(frame),
                    "V1_Row_Hash": _row_hash(frame),
                }
            )
    keys = pd.concat(frames, ignore_index=True).drop_duplicates()
    keys["facility_id"] = keys.facility_id.astype(str)
    keys["feature_date"] = pd.to_datetime(keys.feature_date)
    return keys, pd.DataFrame(row_audit)


def _cv_preflight(root: Path, target_specs: dict, semantic_columns: list[str]):
    rows = []
    target_eligibility = {}
    threshold = 0.50
    for target, spec in target_specs.items():
        train = pd.read_parquet(root / "prepared" / target / "train.parquet")
        fold_survivors = []
        for fold, fit, _ in temporal_folds(train):
            survivors = []
            for variable in semantic_columns:
                values = pd.to_numeric(fit[variable], errors="coerce").replace([np.inf, -np.inf], np.nan)
                missing = float(values.isna().mean())
                unique = int(values.nunique(dropna=True))
                eligible = missing <= threshold and unique >= 2
                if eligible:
                    survivors.append(variable)
                rows.append(
                    {
                        "Target_ID": target,
                        "Fold_ID": fold,
                        "Variable_ID": variable,
                        "Fit_Rows": len(fit),
                        "Missing_Rate": missing,
                        "Unique_Values": unique,
                        "Eligible_Before_Correlation_VIF": eligible,
                        "Reason": "eligible" if eligible else (
                            "missing_over_50_percent" if missing > threshold else "constant_or_empty"
                        ),
                    }
                )
            fold_survivors.append(len(survivors))
        target_eligibility[target] = {
            "eligible": all(count >= 1 for count in fold_survivors),
            "surviving_e6_features_by_fold": fold_survivors,
        }
    pd.DataFrame(rows).to_csv(root / "e6_cv_preflight.csv", index=False, encoding="utf-8-sig")
    atomic_json(root / "e6_target_eligibility.json", target_eligibility)
    return target_eligibility


def prepare(root: Path, v1_root: Path = V1_ROOT):
    root, v1_root = Path(root), Path(v1_root)
    prepared = root / "prepared"
    manifest_path = root / "prepared_manifest.json"
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text())
        for target, spec in result["targets"].items():
            for split, sha in spec["dataset_hashes"].items():
                if file_hash(prepared / target / f"{split}.parquet") != sha:
                    raise ValueError("Semantic v2 prepared dataset changed")
        return result

    root.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    snapshot_manifest, snapshot_hash = _verify_v1_snapshot(v1_root)
    reference, mapping, reference_hash = load_mapping(REFERENCE_FILE)
    mapping.to_csv(root / "e6_semantic_mapping.csv", index=False, encoding="utf-8-sig")
    reference.to_csv(root / "sfkr_col_info_frozen.csv", index=False, encoding="utf-8-sig")

    stream_manifest = partition_semantic_streams(
        v1_root / "data_snapshot" / "sfkr_hbfm_env_con.parquet",
        root / "semantic_streams",
        mapping,
    )
    observed = set(stream_manifest["observed_codes"])
    active_mapping = mapping.loc[mapping.source_code.astype(str).isin(observed)].copy()
    catalog = semantic_feature_catalog(active_mapping)
    semantic_columns = catalog.Variable_ID.tolist()
    if not semantic_columns:
        raise ValueError("No semantic E6 variables were generated")

    keys, row_audit = _source_keys(v1_root)
    semantic, coverage = build_all_features(keys, root / "semantic_streams")
    for column in semantic_columns:
        if column not in semantic:
            semantic[column] = np.nan
    semantic = semantic[["facility_id", "feature_date", *semantic_columns]]
    semantic.to_parquet(root / "semantic_e6_features.parquet", index=False)
    coverage.to_csv(root / "e6_coverage_audit.csv", index=False, encoding="utf-8-sig")
    catalog.to_csv(root / "semantic_feature_catalog.csv", index=False, encoding="utf-8-sig")

    target_specs = {}
    row_results = []
    for target in TARGETS:
        source_dir = v1_root / "prepared" / target
        source_spec = json.loads((source_dir / "manifest.json").read_text())
        old_e6 = set(source_spec["groups"].get("E6", []))
        destination = prepared / target
        destination.mkdir(parents=True, exist_ok=True)
        hashes, row_hashes = {}, {}
        for split in ("train", "validation", "test"):
            source = pd.read_parquet(source_dir / f"{split}.parquet")
            source["facility_id"] = source.facility_id.astype(str)
            source["feature_date"] = pd.to_datetime(source.feature_date)
            base = source.drop(columns=[c for c in old_e6 if c in source])
            frame = base.merge(
                semantic,
                on=["facility_id", "feature_date"],
                how="left",
                validate="many_to_one",
            )
            if len(frame) != len(source) or set(frame.row_id) != set(source.row_id):
                raise ValueError(f"Target rows changed while attaching semantic E6: {target}/{split}")
            frame = frame.sort_values("row_id").reset_index(drop=True)
            path = destination / f"{split}.parquet"
            frame.to_parquet(path, index=False)
            hashes[split] = file_hash(path)
            row_hashes[split] = _row_hash(frame)
            v1_hash = _row_hash(source)
            if row_hashes[split] != v1_hash:
                raise ValueError(f"v1/v2 row hash mismatch: {target}/{split}")
            row_results.append(
                {"Target_ID": target, "Split": split, "Rows": len(frame), "V1_Row_Hash": v1_hash, "V2_Row_Hash": row_hashes[split], "Match": True}
            )
        groups = dict(source_spec["groups"])
        groups["E6"] = semantic_columns
        numeric = sorted((set(source_spec["numeric"]) - old_e6) | set(semantic_columns))
        unavailable = [g for g in source_spec["unavailable_groups"] if g != "E6"]
        spec = {
            **source_spec,
            "groups": groups,
            "numeric": numeric,
            "unavailable_groups": unavailable,
            "dataset_hashes": hashes,
            "row_hashes": row_hashes,
            "source_v1_dataset_hashes": source_spec["dataset_hashes"],
            "source_v1_manifest_hash": file_hash(source_dir / "manifest.json"),
            "e6_semantic_version": "2.0",
        }
        atomic_json(destination / "manifest.json", spec)
        target_specs[target] = spec

    pd.DataFrame(row_results).to_csv(root / "v1_v2_row_hash_audit.csv", index=False, encoding="utf-8-sig")
    eligibility = _cv_preflight(root, target_specs, semantic_columns)
    result = {
        "targets": target_specs,
        "target_count": len(target_specs),
        "semantic_feature_count": len(semantic_columns),
        "target_eligibility": eligibility,
        "v1_root": str(v1_root.resolve()),
        "v1_snapshot_manifest_hash": snapshot_hash,
        "v1_snapshot_tables": snapshot_manifest["tables"],
        "sfkr_col_info_reference_hash": reference_hash,
        "mapping_hash": file_hash(root / "e6_semantic_mapping.csv"),
        "semantic_feature_catalog_hash": file_hash(root / "semantic_feature_catalog.csv"),
        "semantic_stream_manifest": stream_manifest,
        "pid": os.getpid(),
    }
    atomic_json(manifest_path, result)
    return result
