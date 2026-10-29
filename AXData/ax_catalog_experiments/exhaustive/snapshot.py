from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..core import file_hash
from .config import CUTOFF
from .credentials import load_db_environment
from .io import atomic_json

REQUIRED = {
    "sfkr_hbfm_grow": "examin_date",
    "sfkr_pvsn_grow": "meas_date",
    "sfkr_pvsn_crop": None,
    "sfkr_hbfm_env_con": "meas_date",
    "sfkr_pvsn_id": None,
}
DEFAULT_CHUNK_ROWS = 100_000


def _connect():
    from pymysql.cursors import SSCursor
    from AXData.tomato_flower_count_prediction.extract import connect

    connection = connect()
    # pandas chunksize only limits memory when the DB driver also streams rows.
    connection.cursorclass = SSCursor
    return connection


def _schema(connection):
    placeholders = ",".join(["%s"] * len(REQUIRED))
    sql = f"""SELECT table_name,column_name,column_type,column_comment,ordinal_position
              FROM information_schema.columns WHERE table_schema=DATABASE()
                AND table_name IN ({placeholders}) ORDER BY table_name,ordinal_position"""
    return pd.read_sql(sql, connection, params=tuple(REQUIRED))


def _arrow_type(mysql_type: str) -> pa.DataType:
    value = str(mysql_type).lower()
    if any(token in value for token in ("bigint", "mediumint", "smallint", "tinyint", "int")):
        return pa.int64()
    if any(token in value for token in ("decimal", "numeric", "float", "double", "real")):
        return pa.float64()
    if "datetime" in value or "timestamp" in value:
        return pa.timestamp("us")
    if value.startswith("date"):
        return pa.date32()
    if any(token in value for token in ("binary", "blob")):
        return pa.binary()
    return pa.string()


def _arrow_schema(table_schema: pd.DataFrame) -> pa.Schema:
    ordered = table_schema.sort_values("ordinal_position")
    return pa.schema(
        [
            pa.field(row.column_name, _arrow_type(row.column_type), nullable=True)
            for row in ordered.itertuples(index=False)
        ]
    )


def _write_frames(
    frames: Iterable[pd.DataFrame],
    path: Path,
    table_schema: pd.DataFrame,
    date_column: str | None,
    cutoff: str,
    on_chunk=None,
) -> dict:
    """Write frames incrementally and atomically with a stable DB-derived schema."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.unlink(missing_ok=True)
    arrow_schema = _arrow_schema(table_schema)
    writer = pq.ParquetWriter(
        temporary,
        arrow_schema,
        compression="zstd",
        use_dictionary=True,
    )
    rows = 0
    chunks = 0
    min_date = None
    max_date = None
    try:
        for frame in frames:
            if frame.empty:
                continue
            frame = frame.reindex(columns=arrow_schema.names)
            if date_column:
                dates = pd.to_datetime(frame[date_column], errors="coerce")
                valid = dates.dropna()
                if not valid.empty:
                    current_min = valid.min()
                    current_max = valid.max()
                    if current_min < pd.Timestamp(cutoff):
                        raise ValueError(
                            f"Pre-2024 row escaped cutoff in {path.stem}: {current_min}"
                        )
                    min_date = current_min if min_date is None else min(min_date, current_min)
                    max_date = current_max if max_date is None else max(max_date, current_max)
            batch = pa.Table.from_pandas(
                frame,
                schema=arrow_schema,
                preserve_index=False,
                safe=False,
            )
            writer.write_table(batch)
            rows += len(frame)
            chunks += 1
            if on_chunk:
                on_chunk(rows, chunks, temporary.stat().st_size)
        writer.close()
        writer = None
        temporary.replace(path)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return {
        "rows": rows,
        "chunks": chunks,
        "columns": list(arrow_schema.names),
        "min_date": str(min_date) if min_date is not None else None,
        "max_date": str(max_date) if max_date is not None else None,
    }


def _query(table: str, date_column: str | None, columns: set[str], cutoff: str):
    if table == "sfkr_pvsn_crop":
        if "cropping_end_date" not in columns:
            raise ValueError("sfkr_pvsn_crop.cropping_end_date missing")
        return (
            f"SELECT * FROM `{table}` "
            "WHERE CAST(cropping_end_date AS DATE) >= %s "
            "OR cropping_end_date IS NULL",
            (cutoff[:10],),
        )
    if date_column:
        return (
            f"SELECT * FROM `{table}` WHERE `{date_column}` >= %s "
            f"ORDER BY `{date_column}`",
            (cutoff,),
        )
    return f"SELECT * FROM `{table}`", None


def snapshot_db(root: Path, cutoff=CUTOFF):
    """Create an immutable, streamed, read-only DB snapshot."""
    root = Path(root)
    dest = root / "data_snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    progress_path = dest / "snapshot_progress.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for table, item in manifest["tables"].items():
            path = dest / item["file"]
            if not path.exists() or file_hash(path) != item["sha256"]:
                raise ValueError(f"Frozen DB snapshot changed: {path}")
        return manifest
    load_db_environment()
    if not os.getenv("FARMSTOM_DB_PASSWORD"):
        raise RuntimeError(
            "FARMSTOM_DB_PASSWORD is required and no legacy GEAS DB setting was found; "
            "DB re-extraction is mandatory"
        )
    chunk_rows = int(os.getenv("AX_DB_CHUNK_ROWS", str(DEFAULT_CHUNK_ROWS)))
    if chunk_rows < 1:
        raise ValueError("AX_DB_CHUNK_ROWS must be a positive integer")
    connection = _connect()
    started_at = time.time()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT")
        schema = _schema(connection)
        present = set(schema.table_name)
        missing_tables = sorted(set(REQUIRED) - present)
        if missing_tables:
            raise ValueError(f"Required DB tables missing: {missing_tables}")
        schema.to_csv(dest / "schema.csv", index=False, encoding="utf-8-sig")
        tables = {}
        for index, (table, date_column) in enumerate(REQUIRED.items(), start=1):
            table_schema = schema.loc[schema.table_name.eq(table)].copy()
            columns = set(table_schema.column_name)
            if date_column and date_column not in columns:
                raise ValueError(f"Required date column missing: {table}.{date_column}")
            sql, params = _query(table, date_column, columns, cutoff)
            path = dest / f"{table}.parquet"
            atomic_json(
                progress_path,
                {
                    "Status": "extracting",
                    "Current_Table": table,
                    "Table_Index": index,
                    "Table_Count": len(REQUIRED),
                    "Rows_Written": 0,
                    "Chunks_Written": 0,
                    "Chunk_Rows": chunk_rows,
                    "Cutoff": cutoff,
                    "Started_At": started_at,
                    "Updated_At": time.time(),
                },
            )

            def update_progress(rows, chunks, bytes_written):
                atomic_json(
                    progress_path,
                    {
                        "Status": "extracting",
                        "Current_Table": table,
                        "Table_Index": index,
                        "Table_Count": len(REQUIRED),
                        "Rows_Written": rows,
                        "Chunks_Written": chunks,
                        "Bytes_Written": bytes_written,
                        "Chunk_Rows": chunk_rows,
                        "Cutoff": cutoff,
                        "Started_At": started_at,
                        "Updated_At": time.time(),
                    },
                )

            frames = pd.read_sql(
                sql,
                connection,
                params=params,
                chunksize=chunk_rows,
            )
            stats = _write_frames(
                frames,
                path,
                table_schema,
                date_column,
                cutoff,
                update_progress,
            )
            tables[table] = {
                "file": path.name,
                **stats,
                "date_column": date_column,
                "sha256": file_hash(path),
                "sql": sql,
                "params": list(params or ()),
            }
        connection.rollback()
        manifest = {
            "cutoff": cutoff,
            "read_only": True,
            "streamed": True,
            "chunk_rows": chunk_rows,
            "created_at": time.time(),
            "database": os.getenv("FARMSTOM_DB_NAME", "farmstom_usoo"),
            "schema_sha256": file_hash(dest / "schema.csv"),
            "tables": tables,
            "credentials_recorded": False,
        }
        atomic_json(manifest_path, manifest)
        atomic_json(
            progress_path,
            {
                "Status": "complete",
                "Table_Count": len(REQUIRED),
                "Rows_Written": sum(item["rows"] for item in tables.values()),
                "Cutoff": cutoff,
                "Started_At": started_at,
                "Updated_At": time.time(),
            },
        )
        return manifest
    except BaseException as exc:
        connection.rollback()
        atomic_json(
            progress_path,
            {
                "Status": "failed",
                "Error": f"{type(exc).__name__}: {exc}",
                "Cutoff": cutoff,
                "Started_At": started_at,
                "Updated_At": time.time(),
            },
        )
        raise
    finally:
        connection.close()


def preflight_snapshot(root: Path):
    manifest = snapshot_db(root)
    empty = [name for name, item in manifest["tables"].items() if item["rows"] == 0]
    if empty:
        raise ValueError(f"Required DB extraction is empty: {empty}")
    return manifest
