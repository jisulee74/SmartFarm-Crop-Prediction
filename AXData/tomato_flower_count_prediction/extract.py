from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def connect():
    import pymysql
    password = os.environ.get("FARMSTOM_DB_PASSWORD")
    if not password:
        raise RuntimeError("FARMSTOM_DB_PASSWORD is required with --rebuild-cache")
    return pymysql.connect(
        host=os.environ.get("FARMSTOM_DB_HOST", "211.195.9.227"),
        port=int(os.environ.get("FARMSTOM_DB_PORT", "3306")),
        user=os.environ.get("FARMSTOM_DB_USER", "root"), password=password,
        database=os.environ.get("FARMSTOM_DB_NAME", "farmstom_usoo"), charset="utf8mb4",
        connect_timeout=10, read_timeout=600,
    )


def _columns(connection, table: str) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(f"SHOW COLUMNS FROM `{table}`")
        return {row[0] for row in cursor.fetchall()}


def _value_column(columns: set[str], configured: str | None) -> str:
    if configured:
        if configured not in columns:
            raise ValueError(f"Configured growth value column does not exist: {configured}")
        return configured
    candidates = ["growth_measure_value", "growth_measure_val", "measure_value", "growth_value", "sen_val", "value"]
    found = [x for x in candidates if x in columns]
    if len(found) != 1:
        raise ValueError(f"Cannot uniquely infer flower-count column; candidates found={found}. Use --growth-value-column")
    return found[0]


def rebuild_cache(output: Path, value_column: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    connection = connect()
    try:
        columns = _columns(connection, "sfkr_hbfm_grow")
        value = _value_column(columns, value_column)
        sn = "shg.sn AS source_sn," if "sn" in columns else ""
        growth_sql = f"""
          SELECT {sn} shg.facility_id,SUBSTRING_INDEX(shg.facility_id,'_',2) AS user_id,
                 shg.item_code,shg.examin_date,shg.sample_number AS sample_num,shg.growth_measure_code,
                 shg.`{value}` AS flower_count,sc.sn AS crop_sn,sc.cropping_serl_no,
                 sc.cropping_date,sc.cropping_end_date
          FROM sfkr_hbfm_grow shg JOIN sfkr_pvsn_crop sc
            ON shg.item_code=sc.item_code
           AND SUBSTRING_INDEX(shg.facility_id,'_',2)=sc.user_id
           AND shg.examin_date BETWEEN CAST(sc.cropping_date AS DATE) AND CAST(sc.cropping_end_date AS DATE)
          WHERE shg.item_code='080300'
            AND shg.growth_measure_code BETWEEN '10000234' AND '10000283'
        """
        growth = pd.read_sql(growth_sql, connection)
        growth.to_parquet(output / "growth.parquet", index=False)
        facilities = sorted(growth.facility_id.dropna().unique())
        env_parts = []
        for facility in facilities:
            dates = pd.to_datetime(growth.loc[growth.facility_id.eq(facility), "examin_date"])
            start = dates.min() - pd.Timedelta(days=7)
            end = dates.max() + pd.Timedelta(days=1)
            sql = """
              SELECT sn AS source_sn,facility_id,item_code,fld_code,sect_code,fatr_code,
                     maker_id,sen_id,sen_val,meas_date
              FROM sfkr_hbfm_env_con FORCE INDEX (sfkr_hbfm_env_con_facility_id_meas_date)
              WHERE facility_id=%s AND meas_date >= %s AND meas_date < %s
                AND (item_code='080300' OR item_code IS NULL OR item_code='')
                AND ((sect_code='EI' AND fatr_code IN ('TI','HI','CI'))
                     OR (sect_code='CR' AND (fatr_code IN ('CC0903','CC1403','CC2503','CC0403')
                         OR LEFT(fatr_code,5) IN ('CC01_','CC03_'))))
              ORDER BY meas_date,sn
            """
            env_parts.append(pd.read_sql(sql, connection, params=(facility, start, end)))
        env = pd.concat(env_parts, ignore_index=True) if env_parts else pd.DataFrame()
        env.to_parquet(output / "environment.parquet", index=False)
    finally:
        connection.close()
    return {"growth_rows": len(growth), "environment_rows": len(env), "facilities": len(facilities),
            "growth_value_column": value}

