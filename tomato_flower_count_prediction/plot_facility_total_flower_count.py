"""Plot per-sample total flower count and crop-period shading for one facility."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


def normalize_facility_id(value: str, available: pd.Series) -> str:
    candidates = sorted(set(available.dropna().astype(str)))
    if value in candidates:
        return value
    matches = [candidate for candidate in candidates if candidate.startswith(f"{value}_")]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Facility {value!r} matched {matches or 'no cached facility'}")


def build_plot_data(raw: pd.DataFrame, facility_id: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    resolved = normalize_facility_id(facility_id, raw["facility_id"])
    data = raw.loc[raw["facility_id"].astype(str).eq(resolved)].copy()
    for column in ("examin_date", "cropping_date", "cropping_end_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["truss_number"] = pd.to_numeric(data["growth_measure_code"], errors="coerce") - 10_000_233
    data["flower_count"] = pd.to_numeric(data["flower_count"], errors="coerce")
    data = data.loc[data["truss_number"].between(1, 34) & data["flower_count"].notna()]

    # Match the modelling definition: one value per sample/date/truss, then sum trusses 1–34.
    keys = ["crop_sn", "sample_num", "examin_date", "truss_number"]
    conflicts = data.groupby(keys, dropna=False)["flower_count"].nunique().gt(1)
    if conflicts.any():
        bad = conflicts[conflicts].index
        index = pd.MultiIndex.from_frame(data[keys])
        data = data.loc[~index.isin(bad)]
    data = data.drop_duplicates(keys + ["flower_count"])
    totals = (
        data.groupby(["crop_sn", "sample_num", "examin_date"], as_index=False)
        .agg(total_flower_count=("flower_count", "sum"), observed_truss_count=("truss_number", "nunique"))
        .sort_values(["sample_num", "examin_date"])
    )
    periods = (
        data[["crop_sn", "cropping_serl_no", "cropping_date", "cropping_end_date"]]
        .drop_duplicates()
        .sort_values("cropping_date")
        .reset_index(drop=True)
    )
    return totals, periods, resolved


def plot_facility(raw_path: Path, facility_id: str, output: Path) -> tuple[Path, Path]:
    totals, periods, resolved = build_plot_data(pd.read_parquet(raw_path), facility_id)
    if totals.empty:
        raise ValueError(f"No 1–34 truss flower-count observations for {resolved}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(18, 7.5))
    colors = plt.get_cmap("Pastel1")
    for idx, row in periods.iterrows():
        label = f"Crop {row.crop_sn}: {row.cropping_date:%Y-%m-%d}–{row.cropping_end_date:%Y-%m-%d}"
        ax.axvspan(row.cropping_date, row.cropping_end_date, color=colors(idx % 9), alpha=0.48,
                   label=label, zorder=0)

    for sample_num, frame in totals.groupby("sample_num", sort=True):
        ax.plot(frame.examin_date, frame.total_flower_count, color="#7f8c8d", alpha=0.32,
                linewidth=1.15, marker="o", markersize=2.5, label="Individual samples" if sample_num == totals.sample_num.iloc[0] else None)

    mean_series = totals.groupby("examin_date", as_index=False).total_flower_count.mean()
    ax.plot(mean_series.examin_date, mean_series.total_flower_count, color="#c0392b", linewidth=2.8,
            marker="o", markersize=5, label="Mean across samples", zorder=5)
    ax.set_title(f"Total flower count over time — {resolved}", pad=15)
    ax.set_xlabel("Survey date")
    ax.set_ylabel("Total flower count per sample (trusses 1–34)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    csv_path = output.with_suffix(".csv")
    totals.assign(facility_id=resolved).to_csv(csv_path, index=False)
    return output, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--facility-id", default="PF_0025108")
    parser.add_argument("--raw-path", type=Path, default=ROOT / "data/raw/growth.parquet")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/facility_visualizations/PF_0025108_total_flower_count_timeseries.png")
    args = parser.parse_args()
    image, csv = plot_facility(args.raw_path, args.facility_id, args.output)
    print(image)
    print(csv)
