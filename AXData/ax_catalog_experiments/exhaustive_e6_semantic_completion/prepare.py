from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ..core import digest, file_hash
from ..exhaustive.config import TARGETS
from ..exhaustive.io import atomic_json
from ..exhaustive_e6_semantic.prepare import _cv_preflight
from .config import BACKFILL_ROOT, SEMANTIC_V2_ROOT


def _row_hash(frame: pd.DataFrame) -> str:
    return digest(sorted(frame.row_id.astype(str).tolist()))


def _verify_source(root: Path, manifest_name: str):
    path = root / manifest_name
    if not path.exists():
        raise FileNotFoundError(f"Required source manifest is missing: {path}")
    return json.loads(path.read_text()), file_hash(path)


def prepare(
    root: Path,
    backfill_root: Path = BACKFILL_ROOT,
    semantic_root: Path = SEMANTIC_V2_ROOT,
):
    """Attach frozen semantic E6 features to the audited E1/E2 prepared rows."""
    root = Path(root)
    backfill_root = Path(backfill_root)
    semantic_root = Path(semantic_root)
    manifest_path = root / "prepared_manifest.json"
    prepared = root / "prepared"
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text())
        for target, spec in result["targets"].items():
            for split, sha in spec["dataset_hashes"].items():
                if file_hash(prepared / target / f"{split}.parquet") != sha:
                    raise ValueError("Combined semantic E6 prepared dataset changed")
        return result

    root.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    backfill_manifest, backfill_hash = _verify_source(
        backfill_root, "prepared_manifest.json"
    )
    semantic_manifest, semantic_hash = _verify_source(
        semantic_root, "prepared_manifest.json"
    )
    semantic_path = semantic_root / "semantic_e6_features.parquet"
    catalog_path = semantic_root / "semantic_feature_catalog.csv"
    mapping_path = semantic_root / "e6_semantic_mapping.csv"
    for required in (semantic_path, catalog_path, mapping_path):
        if not required.exists():
            raise FileNotFoundError(f"Required semantic E6 artifact missing: {required}")

    catalog = pd.read_csv(catalog_path)
    semantic_columns = sorted(catalog.Variable_ID.astype(str).unique().tolist())
    if not semantic_columns:
        raise ValueError("Semantic E6 feature catalog is empty")
    semantic = pd.read_parquet(semantic_path)
    semantic["facility_id"] = semantic.facility_id.astype(str)
    semantic["feature_date"] = pd.to_datetime(semantic.feature_date)
    semantic = semantic[["facility_id", "feature_date", *semantic_columns]]
    if semantic.duplicated(["facility_id", "feature_date"]).any():
        raise ValueError("Semantic E6 feature keys are not unique")

    target_specs = {}
    audits = []
    for target in TARGETS:
        source_dir = backfill_root / "prepared" / target
        source_spec = json.loads((source_dir / "manifest.json").read_text())
        if target not in backfill_manifest["targets"]:
            raise ValueError(f"Target missing from E1/E2 manifest: {target}")
        old_e6 = set(source_spec["groups"].get("E6", []))
        destination = prepared / target
        destination.mkdir(parents=True, exist_ok=True)
        hashes, row_hashes = {}, {}
        for split in ("train", "validation", "test"):
            source_path = source_dir / f"{split}.parquet"
            if file_hash(source_path) != source_spec["dataset_hashes"][split]:
                raise ValueError(f"E1/E2 prepared source changed: {target}/{split}")
            source = pd.read_parquet(source_path)
            before = _row_hash(source)
            source["facility_id"] = source.facility_id.astype(str)
            source["feature_date"] = pd.to_datetime(source.feature_date)
            base = source.drop(columns=[column for column in old_e6 if column in source])
            frame = base.merge(
                semantic,
                on=["facility_id", "feature_date"],
                how="left",
                validate="many_to_one",
            )
            if len(frame) != len(source) or set(frame.row_id) != set(source.row_id):
                raise ValueError(f"Rows changed while attaching semantic E6: {target}/{split}")
            frame = frame.sort_values("row_id").reset_index(drop=True)
            after = _row_hash(frame)
            if before != after:
                raise ValueError(f"Row hash changed while attaching semantic E6: {target}/{split}")
            output = destination / f"{split}.parquet"
            frame.to_parquet(output, index=False)
            hashes[split] = file_hash(output)
            row_hashes[split] = after
            audits.append(
                {
                    "Target_ID": target,
                    "Split": split,
                    "Rows": len(frame),
                    "E1_E2_Row_Hash": before,
                    "Combined_Row_Hash": after,
                    "Match": True,
                }
            )

        groups = {key: list(value) for key, value in source_spec["groups"].items()}
        groups["E6"] = semantic_columns
        unavailable = [
            group for group in source_spec.get("unavailable_groups", []) if group != "E6"
        ]
        spec = {
            **source_spec,
            "groups": groups,
            "numeric": sorted(
                (set(source_spec["numeric"]) - old_e6) | set(semantic_columns)
            ),
            "unavailable_groups": sorted(set(unavailable)),
            "dataset_hashes": hashes,
            "row_hashes": row_hashes,
            "source_e1_e2_dataset_hashes": source_spec["dataset_hashes"],
            "source_e1_e2_manifest_hash": file_hash(source_dir / "manifest.json"),
            "semantic_e6_feature_hash": file_hash(semantic_path),
            "semantic_e6_completion_version": "1.0",
        }
        if not spec["groups"].get("E1") or not spec["groups"].get("E2"):
            raise ValueError(f"E1/E2 features are empty after combination: {target}")
        atomic_json(destination / "manifest.json", spec)
        target_specs[target] = spec

    pd.DataFrame(audits).to_csv(
        root / "row_identity_audit.csv", index=False, encoding="utf-8-sig"
    )
    shutil.copy2(catalog_path, root / "semantic_feature_catalog.csv")
    shutil.copy2(mapping_path, root / "e6_semantic_mapping.csv")
    for name in ("e6_coverage_audit.csv", "sfkr_col_info_frozen.csv"):
        source = semantic_root / name
        if source.exists():
            shutil.copy2(source, root / name)
    eligibility = _cv_preflight(root, target_specs, semantic_columns)
    result = {
        "targets": target_specs,
        "target_count": len(target_specs),
        "semantic_feature_count": len(semantic_columns),
        "target_eligibility": eligibility,
        "backfill_root": str(backfill_root.resolve()),
        "backfill_prepared_manifest_hash": backfill_hash,
        "semantic_root": str(semantic_root.resolve()),
        "semantic_prepared_manifest_hash": semantic_hash,
        "semantic_feature_hash": file_hash(semantic_path),
        "semantic_catalog_hash": file_hash(catalog_path),
        "semantic_mapping_hash": file_hash(mapping_path),
    }
    atomic_json(manifest_path, result)
    return result
