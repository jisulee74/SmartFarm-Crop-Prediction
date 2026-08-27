#!/usr/bin/env python3
"""Read-only pre-split joinability audit for growth and environment data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
TARGET = ["first_fruits_num", "second_fruits_num", "third_fruits_num"]
DATE_CUTOFF = pd.Timestamp("2024-05-02")


def connect():
    password = os.environ.get("FARMSTOM_DB_PASSWORD")
    if not password:
        raise RuntimeError("FARMSTOM_DB_PASSWORD is required")
    return pymysql.connect(
        host=os.environ.get("FARMSTOM_DB_HOST", "211.195.9.227"),
        port=int(os.environ.get("FARMSTOM_DB_PORT", "3306")),
        user=os.environ.get("FARMSTOM_DB_USER", "root"),
        password=password,
        database=os.environ.get("FARMSTOM_DB_NAME", "farmstom_usoo"),
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=300,
        cursorclass=pymysql.cursors.DictCursor,
    )


def query(connection, sql, params=None):
    with connection.cursor() as cursor:
        cursor.execute(sql, params or ())
        return pd.DataFrame(cursor.fetchall())


def load_growth(connection):
    growth = query(
        connection,
        """
        SELECT sn,user_id,item_code,sample_num,meas_date,
               first_fruits_num,second_fruits_num,third_fruits_num
        FROM sfkr_pvsn_grow
        WHERE meas_date >= '2024-05-02'
          AND first_fruits_num IS NOT NULL
          AND second_fruits_num IS NOT NULL
          AND third_fruits_num IS NOT NULL
        ORDER BY user_id,meas_date,sample_num,sn
        """,
    )
    growth["meas_date"] = pd.to_datetime(growth["meas_date"])
    for column in TARGET:
        growth[column] = pd.to_numeric(growth[column], errors="raise")
    growth["fruit_count_target"] = growth[TARGET].sum(axis=1)
    return growth


def apply_duplicate_policy(growth):
    key = ["user_id", "meas_date", "sample_num"]
    audit_rows = []
    conflict_dates = set()
    for values, frame in growth.groupby(key, dropna=False, sort=False):
        versions = frame[TARGET].drop_duplicates()
        status = "unique" if len(frame) == 1 else ("identical" if len(versions) == 1 else "conflict")
        audit_rows.append({
            "user_id": values[0], "meas_date": values[1], "sample_num": values[2],
            "rows": len(frame), "target_versions": len(versions), "status": status,
            "source_sns": ",".join(frame["sn"].astype(str)),
        })
        if status == "conflict":
            conflict_dates.add((values[0], pd.Timestamp(values[1])))
    audit = pd.DataFrame(audit_rows)
    deduped = growth.drop_duplicates(key + TARGET, keep="first")
    index = pd.MultiIndex.from_frame(deduped[["user_id", "meas_date"]])
    conflict_index = pd.MultiIndex.from_tuples(sorted(conflict_dates), names=["user_id", "meas_date"])
    excluded_mask = index.isin(conflict_index)
    return deduped.loc[~excluded_mask].copy(), deduped.loc[excluded_mask].copy(), audit


def id_sets(connection, eligible_users):
    grow_all = query(connection, "SELECT DISTINCT user_id FROM sfkr_pvsn_grow WHERE user_id IS NOT NULL ORDER BY user_id")
    env_fac = query(
        connection,
        "SELECT DISTINCT facility_id FROM sfkr_hbfm_env_con FORCE INDEX (sfkr_hbfm_env_con_facility_id_meas_date) WHERE facility_id IS NOT NULL ORDER BY facility_id",
    )
    mapping = query(connection, "SELECT DISTINCT user_id,facility_id,item_code FROM sfkr_pvsn_id WHERE user_id IS NOT NULL AND facility_id IS NOT NULL")
    mapped_env = mapping[mapping["facility_id"].isin(set(env_fac["facility_id"]))].copy()
    grow_set = set(grow_all["user_id"])
    env_set = set(mapped_env["user_id"])
    eligible_set = set(eligible_users)
    records = []
    for uid in sorted(grow_set | env_set):
        records.append({
            "user_id": uid, "in_growth_all": uid in grow_set, "in_environment_mapped": uid in env_set,
            "in_model_eligible_growth": uid in eligible_set,
            "set_category": "common" if uid in grow_set and uid in env_set else ("growth_only" if uid in grow_set else "environment_only"),
        })
    unmapped_env = env_fac[~env_fac["facility_id"].isin(set(mapping["facility_id"]))].copy()
    return pd.DataFrame(records), mapping, env_fac, unmapped_env


def five_min_floor(series):
    return pd.to_datetime(series).dt.floor("5min")


def audit_facility(connection, user_id, facility_id, growth_user):
    survey_dates = sorted(pd.to_datetime(growth_user["meas_date"]).dt.normalize().unique())
    survey_start = pd.Timestamp(survey_dates[0])
    survey_end = pd.Timestamp(survey_dates[-1])
    extract_start = survey_start - pd.Timedelta(days=7)
    extract_end = survey_end + pd.Timedelta(days=1)
    lo = int(extract_start.strftime("%Y%m%d00"))
    hi = int(extract_end.strftime("%Y%m%d00"))
    env = query(
        connection,
        """
        SELECT sn,item_code,fld_code,sect_code,fatr_code,maker_id,sen_id,sen_val,meas_date
        FROM sfkr_hbfm_env_con FORCE INDEX (sfkr_hbfm_env_con_facility_id_meas_date)
        WHERE facility_id=%s AND meas_ymd_hh >= %s AND meas_ymd_hh < %s
        ORDER BY meas_ymd_hh,meas_date,sn
        """,
        (facility_id, lo, hi),
    )
    if env.empty:
        coverage = pd.DataFrame({"user_id": user_id, "facility_id": facility_id, "survey_date": survey_dates})
        for name in ["same_day_has_env", "prior_has_env", "coverage_1d", "coverage_3d", "coverage_7d"]:
            coverage[name] = 0.0
        return coverage, {"user_id": user_id, "facility_id": facility_id, "env_rows": 0}, pd.DataFrame(), pd.DataFrame()
    env["meas_date"] = pd.to_datetime(env["meas_date"])
    env["date"] = env["meas_date"].dt.normalize()
    env["slot_5m"] = five_min_floor(env["meas_date"])
    env_dates = set(env["date"])
    env_min, env_max = env["meas_date"].min(), env["meas_date"].max()

    coverage_rows = []
    for survey_date in survey_dates:
        day = pd.Timestamp(survey_date)
        coverage_rows.append({
            "user_id": user_id, "facility_id": facility_id, "survey_date": day,
            "same_day_has_env": day in env_dates,
            "prior_has_env": bool((env["meas_date"] < day).any()),
            "coverage_1d": float(sum(day - pd.Timedelta(days=k) in env_dates for k in range(1, 2))),
            "coverage_3d": sum(day - pd.Timedelta(days=k) in env_dates for k in range(1, 4)) / 3,
            "coverage_7d": sum(day - pd.Timedelta(days=k) in env_dates for k in range(1, 8)) / 7,
        })
    coverage = pd.DataFrame(coverage_rows)

    raw_ts = env.groupby("meas_date").size()
    logical = ["fld_code", "sect_code", "fatr_code", "maker_id", "sen_id", "meas_date"]
    logical_dups = env.groupby(logical, dropna=False).size()
    stream_cols = ["fld_code", "sect_code", "fatr_code", "maker_id", "sen_id"]
    interval_records = []
    long_gap_records = []
    for stream_key, frame in env.groupby(stream_cols, dropna=False):
        times = pd.Series(frame["meas_date"].drop_duplicates().sort_values().values)
        gaps = times.diff().dt.total_seconds().div(60).dropna()
        slots = frame["slot_5m"].drop_duplicates().sort_values()
        if len(slots):
            expected = int((slots.iloc[-1] - slots.iloc[0]).total_seconds() // 300) + 1
            continuity = len(slots) / expected if expected else np.nan
        else:
            continuity = np.nan
        interval_records.append({
            "user_id": user_id, "facility_id": facility_id,
            **{name: value for name, value in zip(stream_cols, stream_key)},
            "observations": len(times), "median_gap_minutes": gaps.median() if len(gaps) else np.nan,
            "p95_gap_minutes": gaps.quantile(.95) if len(gaps) else np.nan,
            "max_gap_minutes": gaps.max() if len(gaps) else np.nan,
            "gap_gt_1h_count": int((gaps > 60).sum()), "gap_gt_24h_count": int((gaps > 1440).sum()),
            "five_min_slot_continuity": continuity,
        })
        if len(gaps):
            idx = gaps[gaps > 1440].index
            for i in idx:
                long_gap_records.append({
                    "user_id": user_id, "facility_id": facility_id,
                    **{name: value for name, value in zip(stream_cols, stream_key)},
                    "gap_start": times.iloc[i - 1], "gap_end": times.iloc[i], "gap_hours": gaps.loc[i] / 60,
                })
    interval = pd.DataFrame(interval_records)
    long_gaps = pd.DataFrame(long_gap_records)
    growth_items = sorted(growth_user["item_code"].dropna().astype(str).unique())
    env_items = sorted(env["item_code"].dropna().astype(str).unique())
    summary = {
        "user_id": user_id, "facility_id": facility_id, "growth_min_date": survey_start,
        "growth_max_date": survey_end, "env_extract_min_timestamp": env_min,
        "env_extract_max_timestamp": env_max, "period_overlap": env_min.normalize() <= survey_end and env_max.normalize() >= survey_start,
        "growth_rows": len(growth_user), "growth_survey_dates": len(survey_dates), "env_rows": len(env),
        "survey_dates_same_day_env": int(coverage["same_day_has_env"].sum()),
        "survey_dates_prior_env": int(coverage["prior_has_env"].sum()),
        "same_day_env_rate": coverage["same_day_has_env"].mean(),
        "prior_env_rate": coverage["prior_has_env"].mean(),
        "coverage_1d_mean": coverage["coverage_1d"].mean(),
        "coverage_3d_mean": coverage["coverage_3d"].mean(),
        "coverage_7d_mean": coverage["coverage_7d"].mean(),
        "usable_growth_rows_prior_env": int(growth_user["meas_date"].dt.normalize().isin(set(coverage.loc[coverage["prior_has_env"], "survey_date"])).sum()),
        "usable_growth_rows_7d_full": int(growth_user["meas_date"].dt.normalize().isin(set(coverage.loc[coverage["coverage_7d"] == 1, "survey_date"])).sum()),
        "raw_user_timestamp_duplicate_groups": int((raw_ts > 1).sum()),
        "raw_user_timestamp_duplicate_excess_rows": int((raw_ts[raw_ts > 1] - 1).sum()),
        "logical_sensor_timestamp_duplicate_groups": int((logical_dups > 1).sum()),
        "logical_sensor_timestamp_duplicate_excess_rows": int((logical_dups[logical_dups > 1] - 1).sum()),
        "stream_count": len(interval),
        "median_stream_gap_minutes": interval["median_gap_minutes"].median(),
        "median_stream_5m_continuity": interval["five_min_slot_continuity"].median(),
        "long_gap_gt_24h_count": len(long_gaps),
        "growth_item_codes": ",".join(growth_items), "env_item_codes": ",".join(env_items),
        "item_code_overlap": bool(set(growth_items) & set(env_items)),
    }
    return coverage, summary, interval, long_gaps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    connection = connect()
    try:
        growth_raw = load_growth(connection)
        growth, excluded, duplicate_audit = apply_duplicate_policy(growth_raw)
        id_membership, mapping, env_facilities, unmapped_env = id_sets(connection, growth["user_id"].unique())
        selected_mapping = mapping[mapping["user_id"].isin(growth["user_id"].unique())].copy()
        coverage_parts, summaries, interval_parts, long_gap_parts = [], [], [], []
        unmappable_rows = []
        for user_id, frame in growth.groupby("user_id", sort=True):
            facilities = selected_mapping.loc[selected_mapping["user_id"] == user_id, "facility_id"].dropna().unique()
            facilities = [f for f in facilities if f in set(env_facilities["facility_id"])]
            if len(facilities) != 1:
                unmappable_rows.append({"user_id": user_id, "growth_rows": len(frame), "reason": f"environment_facility_count={len(facilities)}", "facilities": ",".join(facilities)})
                continue
            coverage, summary, interval, long_gaps = audit_facility(connection, user_id, facilities[0], frame)
            coverage_parts.append(coverage); summaries.append(summary)
            if not interval.empty: interval_parts.append(interval)
            if not long_gaps.empty: long_gap_parts.append(long_gaps)
    finally:
        connection.close()

    coverage = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    user_summary = pd.DataFrame(summaries)
    intervals = pd.concat(interval_parts, ignore_index=True) if interval_parts else pd.DataFrame()
    long_gaps = pd.concat(long_gap_parts, ignore_index=True) if long_gap_parts else pd.DataFrame()
    unmappable = pd.DataFrame(unmappable_rows)
    if not user_summary.empty:
        user_summary["growth_rows_without_prior_env"] = user_summary["growth_rows"] - user_summary["usable_growth_rows_prior_env"]
        user_summary["growth_rows_without_full_7d"] = user_summary["growth_rows"] - user_summary["usable_growth_rows_7d_full"]

    files = {
        "growth_duplicate_audit.csv": duplicate_audit,
        "growth_conflict_date_exclusions.csv": excluded,
        "id_membership.csv": id_membership,
        "user_facility_mapping.csv": mapping,
        "environment_facilities.csv": env_facilities,
        "unmapped_environment_facilities.csv": unmapped_env,
        "survey_date_coverage.csv": coverage,
        "modeling_common_facility_summary.csv": user_summary,
        "environment_stream_interval_summary.csv": intervals,
        "environment_long_gaps_gt_24h.csv": long_gaps,
        "unmappable_growth_users.csv": unmappable,
    }
    for name, frame in files.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    manifest = {
        "read_only": True, "dataset_split_performed": False, "interpolation_performed": False,
        "feature_generation_performed": False, "growth_table": "sfkr_pvsn_grow",
        "environment_table": "sfkr_hbfm_env_con", "date_cutoff": str(DATE_CUTOFF.date()),
        "files": sorted(files),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eligible_growth_rows": len(growth), "eligible_users": growth["user_id"].nunique(),
        "analyzed_common_facilities": len(user_summary), "coverage_rows": len(coverage),
        "environment_streams": len(intervals), "long_gaps_gt_24h": len(long_gaps),
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
