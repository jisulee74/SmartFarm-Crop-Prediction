#!/usr/bin/env python3
"""Read-only analysis of all-zero strawberry fruit-count targets.

The script deliberately lives outside the existing dataset pipeline. It issues SELECT
queries only, applies the requested duplicate policy in memory, writes CSV/PNG/Markdown
artifacts, and never performs dataset splitting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
FIG = OUT / "figures"
TARGET_COLUMNS = ["first_fruits_num", "second_fruits_num", "third_fruits_num"]
OTHER_GROWTH = [
    "leaves_length",
    "leaves_num",
    "leaves_width",
    "petiole_length",
    "stem_diameter",
    "theca_diameter",
    "flower_length",
    "grow_length",
]


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
        read_timeout=180,
        cursorclass=pymysql.cursors.DictCursor,
    )


def read_sql(connection, query: str) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())


def load_source_data(connection):
    selected = [
        "sn", "user_id", "item_code", "sample_num", "meas_date",
        *TARGET_COLUMNS, *OTHER_GROWTH,
    ]
    query = f"""
        SELECT {', '.join(selected)}
        FROM sfkr_pvsn_grow
        WHERE meas_date >= '2024-05-02 00:00:00'
          AND fruit_cluster_num IS NULL
          AND first_fruits_num IS NOT NULL
          AND second_fruits_num IS NOT NULL
          AND third_fruits_num IS NOT NULL
        ORDER BY user_id, meas_date, sample_num, sn
    """
    growth = read_sql(connection, query)
    crop = read_sql(
        connection,
        """
        SELECT sn AS crop_sn, user_id, item_code, cropping_serl_no,
               cropping_season_name, cropping_date, cropping_end_date
        FROM sfkr_pvsn_crop
        ORDER BY user_id, cropping_date, cropping_serl_no
        """,
    )
    comments = read_sql(
        connection,
        """
        SELECT table_name, column_name, column_type, column_comment
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name IN ('sfkr_pvsn_grow', 'sfkr_pvsn_crop')
        ORDER BY table_name, ordinal_position
        """,
    )
    return growth, crop, comments


def numericize(growth: pd.DataFrame) -> pd.DataFrame:
    growth = growth.copy()
    growth["meas_date"] = pd.to_datetime(growth["meas_date"], errors="coerce")
    for column in TARGET_COLUMNS + OTHER_GROWTH:
        growth[column] = pd.to_numeric(growth[column], errors="coerce")
    growth["fruit_count_target"] = growth[TARGET_COLUMNS].sum(axis=1, min_count=3)
    growth["is_all_zero"] = growth[TARGET_COLUMNS].eq(0).all(axis=1)
    return growth


def apply_duplicate_policy(growth: pd.DataFrame):
    key = ["user_id", "meas_date", "sample_num"]
    grouped = growth.groupby(key, dropna=False, sort=False)
    records = []
    conflict_dates = set()
    for values, frame in grouped:
        versions = frame[TARGET_COLUMNS].drop_duplicates()
        status = "unique" if len(frame) == 1 else ("identical_duplicate" if len(versions) == 1 else "conflict_duplicate")
        records.append(
            {
                "user_id": values[0], "meas_date": values[1], "sample_num": values[2],
                "row_count": len(frame), "value_version_count": len(versions), "status": status,
                "source_sns": ",".join(map(str, frame["sn"].tolist())),
                "target_versions": ";".join(
                    versions.astype(str).agg("/".join, axis=1).tolist()
                ),
            }
        )
        if status == "conflict_duplicate":
            conflict_dates.add((values[0], pd.Timestamp(values[1])))
    audit = pd.DataFrame(records)
    deduped = growth.drop_duplicates(key + TARGET_COLUMNS, keep="first")
    excluded_mask = pd.MultiIndex.from_frame(deduped[["user_id", "meas_date"]]).isin(
        pd.MultiIndex.from_tuples(sorted(conflict_dates), names=["user_id", "meas_date"])
    )
    excluded = deduped.loc[excluded_mask].copy()
    clean = deduped.loc[~excluded_mask].copy()
    return clean, excluded, audit


def attach_crop_cycles(growth: pd.DataFrame, crop: pd.DataFrame):
    crop = crop.copy()
    crop["cropping_date"] = pd.to_datetime(crop["cropping_date"], errors="coerce")
    crop["cropping_end_date"] = pd.to_datetime(crop["cropping_end_date"], errors="coerce")
    crop = crop.dropna(subset=["user_id", "item_code", "cropping_date", "cropping_end_date"])
    rows = []
    ambiguous = []
    unmatched = []
    crop_groups = {
        key: frame for key, frame in crop.groupby(["user_id", "item_code"], dropna=False)
    }
    for row in growth.itertuples(index=False):
        candidates = crop_groups.get((row.user_id, row.item_code), crop.iloc[0:0])
        candidates = candidates[
            (candidates["cropping_date"] <= row.meas_date)
            & (candidates["cropping_end_date"] >= row.meas_date)
        ]
        base = row._asdict()
        if len(candidates) == 1:
            c = candidates.iloc[0]
            base.update(
                cropping_serl_no=c["cropping_serl_no"],
                cropping_season_name=c["cropping_season_name"],
                cropping_date=c["cropping_date"],
                cropping_end_date=c["cropping_end_date"],
            )
            rows.append(base)
        elif len(candidates) == 0:
            unmatched.append(base)
        else:
            base["candidate_cropping_serl_no"] = ",".join(candidates["cropping_serl_no"].astype(str))
            ambiguous.append(base)
    assigned = pd.DataFrame(rows)
    if not assigned.empty:
        assigned["days_since_crop_start"] = (
            assigned["meas_date"] - assigned["cropping_date"]
        ).dt.days
        assigned["crop_duration_days"] = (
            assigned["cropping_end_date"] - assigned["cropping_date"]
        ).dt.days + 1
        assigned["crop_progress"] = assigned["days_since_crop_start"] / assigned["crop_duration_days"]
        assigned["crop_phase"] = pd.cut(
            assigned["crop_progress"], [-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["early", "middle", "late"], right=False,
        )
    return assigned, pd.DataFrame(ambiguous), pd.DataFrame(unmatched), crop


def summarize_overall_and_cycles(data: pd.DataFrame):
    overall = pd.DataFrame([{
        "scope": "overall", "rows": len(data), "all_zero_rows": int(data["is_all_zero"].sum()),
        "all_zero_rate": data["is_all_zero"].mean(), "positive_rows": int((data["fruit_count_target"] > 0).sum()),
        "unique_user_dates": data[["user_id", "meas_date"]].drop_duplicates().shape[0],
    }])
    by_cycle = data.groupby(
        ["user_id", "cropping_serl_no", "cropping_season_name", "cropping_date", "cropping_end_date"],
        dropna=False,
    ).agg(
        rows=("sn", "size"), all_zero_rows=("is_all_zero", "sum"),
        unique_survey_dates=("meas_date", "nunique"), target_mean=("fruit_count_target", "mean"),
        target_median=("fruit_count_target", "median"), target_max=("fruit_count_target", "max"),
    ).reset_index()
    by_cycle["all_zero_rate"] = by_cycle["all_zero_rows"] / by_cycle["rows"]
    return overall, by_cycle


def elapsed_summaries(data: pd.DataFrame):
    max_day = int(data["days_since_crop_start"].max())
    edges = list(range(0, max_day + 31, 30))
    if edges[-1] <= max_day:
        edges.append(edges[-1] + 30)
    data = data.copy()
    data["elapsed_30d_bin"] = pd.cut(data["days_since_crop_start"], edges, right=False)
    elapsed = data.groupby("elapsed_30d_bin", observed=True).agg(
        rows=("sn", "size"), all_zero_rows=("is_all_zero", "sum"),
        target_mean=("fruit_count_target", "mean"), target_median=("fruit_count_target", "median"),
        target_q25=("fruit_count_target", lambda x: x.quantile(0.25)),
        target_q75=("fruit_count_target", lambda x: x.quantile(0.75)),
        users=("user_id", "nunique"), cycles=("cropping_serl_no", "nunique"),
    ).reset_index()
    elapsed["all_zero_rate"] = elapsed["all_zero_rows"] / elapsed["rows"]
    elapsed["elapsed_30d_bin"] = elapsed["elapsed_30d_bin"].astype(str)
    phase = data.groupby("crop_phase", observed=True).agg(
        rows=("sn", "size"), all_zero_rows=("is_all_zero", "sum"),
        target_mean=("fruit_count_target", "mean"), target_median=("fruit_count_target", "median"),
        target_q25=("fruit_count_target", lambda x: x.quantile(0.25)),
        target_q75=("fruit_count_target", lambda x: x.quantile(0.75)),
    ).reset_index()
    phase["all_zero_rate"] = phase["all_zero_rows"] / phase["rows"]
    return data, elapsed, phase


def sequence_analysis(data: pd.DataFrame):
    sequence_rows = []
    for key, frame in data.groupby(["user_id", "cropping_serl_no", "sample_num"], dropna=False):
        frame = frame.sort_values("meas_date")
        states = frame["fruit_count_target"].gt(0).astype(int).tolist()
        compact = [states[0]] if states else []
        for state in states[1:]:
            if state != compact[-1]:
                compact.append(state)
        text = "→".join("positive" if x else "zero" for x in compact)
        sequence_rows.append({
            "user_id": key[0], "cropping_serl_no": key[1], "sample_num": key[2],
            "observations": len(frame), "first_date": frame["meas_date"].min(),
            "last_date": frame["meas_date"].max(), "compact_sequence": text,
            "all_zero": bool(states and not any(states)),
            "has_zero_to_positive": any(a == 0 and b == 1 for a, b in zip(states, states[1:])),
            "has_positive_to_zero": any(a == 1 and b == 0 for a, b in zip(states, states[1:])),
            "has_positive_zero_positive": any(compact[i:i+3] == [1, 0, 1] for i in range(max(0, len(compact)-2))),
        })
    seq = pd.DataFrame(sequence_rows)
    summary = pd.DataFrame([
        {"pattern": "zero_to_positive", "sample_sequences": int(seq["has_zero_to_positive"].sum())},
        {"pattern": "positive_to_zero", "sample_sequences": int(seq["has_positive_to_zero"].sum())},
        {"pattern": "positive_zero_positive", "sample_sequences": int(seq["has_positive_zero_positive"].sum())},
        {"pattern": "all_zero", "sample_sequences": int(seq["all_zero"].sum())},
    ])
    return seq, summary


def all_sample_zero_days(data: pd.DataFrame):
    daily = data.groupby(["user_id", "cropping_serl_no", "meas_date"], dropna=False).agg(
        sample_count=("sample_num", "nunique"), rows=("sn", "size"),
        zero_rows=("is_all_zero", "sum"), target_mean=("fruit_count_target", "mean"),
    ).reset_index()
    daily["all_samples_zero"] = daily["zero_rows"] == daily["rows"]
    return daily, daily[daily["all_samples_zero"]].copy()


def other_growth_quality(data: pd.DataFrame):
    records = []
    for zero_state, frame in data.groupby("is_all_zero"):
        for column in OTHER_GROWTH:
            values = frame[column]
            records.append({
                "target_group": "all_zero" if zero_state else "positive",
                "column": column, "rows": len(frame), "non_null_count": int(values.notna().sum()),
                "non_null_rate": values.notna().mean(), "positive_count": int(values.gt(0).sum()),
                "positive_rate": values.gt(0).mean(), "median_non_null": values.median(),
            })
    return pd.DataFrame(records)


def make_figures(data, elapsed, phase, cycle, sequences, all_zero_days):
    plt.style.use("seaborn-v0_8-whitegrid")
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(elapsed))
    ax1.plot(x, elapsed["all_zero_rate"] * 100, marker="o", color="#b22222", label="All-zero rate")
    ax1.set_ylabel("All-zero rows (%)")
    ax1.set_xlabel("Days since crop start (30-day bins)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(elapsed["elapsed_30d_bin"], rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, elapsed["target_median"], marker="s", color="#1f77b4", label="Median target")
    ax2.set_ylabel("Median fruit_count_target")
    ax1.set_title("All-zero rate and target by elapsed crop days")
    fig.tight_layout()
    fig.savefig(FIG / "elapsed_zero_rate_and_target.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(phase["crop_phase"].astype(str), phase["all_zero_rate"] * 100, color=["#4c78a8", "#f2cf5b", "#e45756"])
    ax.set_ylabel("All-zero rows (%)")
    ax.set_xlabel("Crop phase (relative thirds)")
    ax.set_title("All-zero rate: early vs middle vs late")
    fig.tight_layout()
    fig.savefig(FIG / "crop_phase_zero_rate.png", dpi=180)
    plt.close(fig)

    plot_cycle = cycle.sort_values("all_zero_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(plot_cycle))))
    labels = plot_cycle["user_id"].astype(str) + "/" + plot_cycle["cropping_serl_no"].astype(str)
    ax.barh(labels, plot_cycle["all_zero_rate"] * 100, color="#72b7b2")
    ax.invert_yaxis()
    ax.set_xlabel("All-zero rows (%)")
    ax.set_title("All-zero rate by user and crop cycle")
    fig.tight_layout()
    fig.savefig(FIG / "user_cycle_zero_rate.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["zero_to_positive", "positive_to_zero", "positive_zero_positive", "all_zero"]
    counts = [int(sequences.loc[sequences["pattern"] == p, "sample_sequences"].iloc[0]) for p in order]
    ax.bar(order, counts, color="#59a14f")
    ax.set_ylabel("Sample sequences")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Within-cycle temporal target patterns")
    fig.tight_layout()
    fig.savefig(FIG / "temporal_sequence_patterns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = data.groupby("crop_phase", observed=True)["fruit_count_target"].apply(lambda x: x.clip(upper=x.quantile(.99))).reset_index()
    groups = [counts.loc[counts["crop_phase"] == p, "fruit_count_target"].values for p in ["early", "middle", "late"]]
    ax.boxplot(groups, tick_labels=["early", "middle", "late"], showfliers=False)
    ax.set_ylabel("fruit_count_target (top 1% clipped per phase)")
    ax.set_title("Target distribution by crop phase")
    fig.tight_layout()
    fig.savefig(FIG / "crop_phase_target_distribution.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns=None, max_rows=200):
    view = frame if columns is None else frame[columns]
    view = view.head(max_rows).copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].round(4)
    return view.to_markdown(index=False)


def write_report(stats):
    overall = stats["overall"].iloc[0]
    phase = stats["phase"]
    seq = stats["sequence_summary"]
    audit = stats["duplicate_audit"]
    identical = int((audit["status"] == "identical_duplicate").sum())
    conflicts = int((audit["status"] == "conflict_duplicate").sum())
    excluded_dates = stats["excluded"][["user_id", "meas_date"]].drop_duplicates().shape[0]
    all_zero_days = stats["all_zero_days"]
    other = stats["other_growth"]
    other_pivot = other.pivot(index="column", columns="target_group", values=["non_null_rate", "positive_rate", "median_non_null"]).reset_index()

    early_rate = float(phase.loc[phase["crop_phase"].astype(str) == "early", "all_zero_rate"].iloc[0]) if (phase["crop_phase"].astype(str) == "early").any() else np.nan
    middle_rate = float(phase.loc[phase["crop_phase"].astype(str) == "middle", "all_zero_rate"].iloc[0]) if (phase["crop_phase"].astype(str) == "middle").any() else np.nan
    late_rate = float(phase.loc[phase["crop_phase"].astype(str) == "late", "all_zero_rate"].iloc[0]) if (phase["crop_phase"].astype(str) == "late").any() else np.nan
    z2p = int(seq.loc[seq["pattern"] == "zero_to_positive", "sample_sequences"].iloc[0])
    p2z = int(seq.loc[seq["pattern"] == "positive_to_zero", "sample_sequences"].iloc[0])
    pzp = int(seq.loc[seq["pattern"] == "positive_zero_positive", "sample_sequences"].iloc[0])
    always = int(seq.loc[seq["pattern"] == "all_zero", "sample_sequences"].iloc[0])

    report = f"""# All-zero fruit-count semantics analysis

## Scope and provenance

- Source tables: `sfkr_pvsn_grow`, `sfkr_pvsn_crop` (read-only `SELECT` queries).
- Date filter: `meas_date >= 2024-05-02`.
- Main analysis requires all three target columns to be non-NULL.
- `fruits_num` is excluded.
- `fruit_count_target = first_fruits_num + second_fruits_num + third_fruits_num`.
- No dataset split was performed.
- Crop assignment uses matching `user_id`, `item_code`, and inclusive crop start/end dates.

## Duplicate and crop-assignment audit

- Identical duplicate keys retained once: **{identical}**.
- Conflicting duplicate keys: **{conflicts}**.
- Facility-date groups excluded because at least one conflicting duplicate existed: **{excluded_dates}**.
- Rows excluded by that date-level rule: **{len(stats['excluded'])}**.
- Ambiguous crop assignments excluded from crop-cycle analyses: **{len(stats['ambiguous'])}** rows.
- Unmatched crop assignments excluded from crop-cycle analyses: **{len(stats['unmatched'])}** rows.

See `duplicate_audit.csv`, `excluded_conflict_date_rows.csv`,
`ambiguous_crop_assignments.csv`, and `unmatched_crop_assignments.csv`.

## Main results

- Analyzed rows after duplicate/date and crop-assignment rules: **{int(overall['rows'])}**.
- All-zero rows: **{int(overall['all_zero_rows'])} ({overall['all_zero_rate']:.2%})**.
- Positive-target rows: **{int(overall['positive_rows'])}**.
- Unique user-survey dates: **{int(overall['unique_user_dates'])}**.
- All samples simultaneously zero on a facility-survey date: **{len(all_zero_days)}** dates.

### User and crop-cycle all-zero rates

{markdown_table(stats['by_cycle'], ['user_id','cropping_serl_no','cropping_season_name','cropping_date','cropping_end_date','rows','all_zero_rows','all_zero_rate','target_mean','target_median','target_max'])}

### Days since crop start

{markdown_table(stats['elapsed'])}

### Early, middle, and late crop phases

The phases are relative thirds of each crop's recorded duration, not biological stage labels.

{markdown_table(phase)}

- Early all-zero rate: **{early_rate:.2%}**
- Middle all-zero rate: **{middle_rate:.2%}**
- Late all-zero rate: **{late_rate:.2%}**

### Within-sample temporal behavior

These indicators can overlap: a sequence may contain both transitions and a positive-zero-positive subsequence.

- `0 → positive`: **{z2p}** sample-cycle sequences.
- `positive → 0`: **{p2z}** sample-cycle sequences.
- `positive → 0 → positive`: **{pzp}** sample-cycle sequences.
- Zero throughout the observed crop sequence: **{always}** sample-cycle sequences.

### Other growth measurements

`positive_rate` means a parsed numeric value greater than zero. This is a recording-completeness
diagnostic, not a biological normal-range test.

{markdown_table(other_pivot)}

## Interpretation

The statistical evidence is mixed and cannot prove data-entry intent:

1. Early-season concentration followed by `0 → positive` supports genuine pre-fruit-set zeros.
2. `positive → 0` can be biologically plausible after harvest/removal, but abrupt
   `positive → 0 → positive` patterns may also reflect inconsistent recording or changing
   interpretation of the three truss fields.
3. Facility-date-wide zeros are more suspicious than isolated plant zeros, especially when
   other growth fields remain populated, but synchronized biological state is still possible.
4. Rows that stay zero throughout an observed crop can be genuine non-setting plants, incomplete
   crop coverage, or a default-value convention. The database alone cannot distinguish them.

## Recommendation before dataset split

- **Do not globally convert all zeros to missing and do not globally discard all zeros.**
- Retain zeros provisionally as valid targets when they occur early in a crop and are followed by
  positive observations for the same sample, or when surrounding measurements support a coherent
  biological trajectory.
- Add flags rather than silently deleting questionable rows:
  `all_zero_target`, `all_samples_zero_on_date`, `positive_zero_positive`,
  `zero_throughout_observed_cycle`, `days_since_crop_start`, and `crop_phase`.
- Exclude every facility-date containing a conflicting duplicate, as done here, until an
  authoritative conflict-resolution rule exists.
- Treat mid/late all-zero rows, facility-wide all-zero dates, and positive-zero-positive sequences
  as sensitivity-analysis candidates. Train/evaluate once with them retained and once excluded.
- Confirm with the data owner whether zero is the UI/database default for unmeasured trusses and
  whether counts reset after harvest. This domain confirmation remains necessary before freezing
  the target contract.

## Figures

- `figures/elapsed_zero_rate_and_target.png`
- `figures/crop_phase_zero_rate.png`
- `figures/crop_phase_target_distribution.png`
- `figures/user_cycle_zero_rate.png`
- `figures/temporal_sequence_patterns.png`
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    connection = connect()
    try:
        growth, crop, comments = load_source_data(connection)
    finally:
        connection.close()
    growth = numericize(growth)
    clean, excluded, audit = apply_duplicate_policy(growth)
    assigned, ambiguous, unmatched, crop_clean = attach_crop_cycles(clean, crop)
    if assigned.empty:
        raise RuntimeError("No growth rows could be assigned uniquely to a crop cycle")
    assigned, elapsed, phase = elapsed_summaries(assigned)
    overall, by_cycle = summarize_overall_and_cycles(assigned)
    sequences, sequence_summary = sequence_analysis(assigned)
    daily, all_zero_days = all_sample_zero_days(assigned)
    other_growth = other_growth_quality(assigned)

    artifacts = {
        "schema_columns.csv": comments,
        "crop_cycles_source.csv": crop_clean,
        "duplicate_audit.csv": audit,
        "excluded_conflict_date_rows.csv": excluded,
        "ambiguous_crop_assignments.csv": ambiguous,
        "unmatched_crop_assignments.csv": unmatched,
        "analysis_rows.csv": assigned,
        "overall_summary.csv": overall,
        "user_crop_cycle_summary.csv": by_cycle,
        "elapsed_30d_summary.csv": elapsed,
        "crop_phase_summary.csv": phase,
        "sample_sequence_details.csv": sequences,
        "sample_sequence_summary.csv": sequence_summary,
        "facility_survey_date_summary.csv": daily,
        "all_samples_zero_dates.csv": all_zero_days,
        "other_growth_recording_quality.csv": other_growth,
    }
    for name, frame in artifacts.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    make_figures(assigned, elapsed, phase, by_cycle, sequence_summary, all_zero_days)
    stats = {
        "overall": overall, "by_cycle": by_cycle, "elapsed": elapsed, "phase": phase,
        "sequence_summary": sequence_summary, "duplicate_audit": audit, "excluded": excluded,
        "ambiguous": ambiguous, "unmatched": unmatched, "all_zero_days": all_zero_days,
        "other_growth": other_growth,
    }
    write_report(stats)
    manifest = {
        "source_tables": ["sfkr_pvsn_grow", "sfkr_pvsn_crop"],
        "read_only": True,
        "dataset_split_performed": False,
        "date_filter": "meas_date >= 2024-05-02",
        "target_columns": TARGET_COLUMNS,
        "files": sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "analysis_rows": len(assigned), "all_zero_rows": int(assigned["is_all_zero"].sum()),
        "conflict_keys": int((audit["status"] == "conflict_duplicate").sum()),
        "excluded_conflict_date_rows": len(excluded), "ambiguous_crop_rows": len(ambiguous),
        "unmatched_crop_rows": len(unmatched), "output": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
