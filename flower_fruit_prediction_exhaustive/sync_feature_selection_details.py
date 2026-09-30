#!/usr/bin/env python3
"""Build compact, browser-friendly validation feature-selection details.

The experiment writes one audit row per target/combination/fold.  The dashboard
only needs the frozen validation-stage selection, which is shared by all six
models for the same target and combination.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
OUTPUT = WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1"
CACHE = HERE / "cache" / "feature_selections"

REASON_LABELS = {
    "missing_over_50_percent": "결측률 50% 초과",
    "constant_or_empty": "상수 또는 유효값 없음",
    "exact_duplicate": "완전 중복",
    "pearson_abs_gte_0.95": "절대 상관계수 0.95 이상",
    "vif_over_10": "VIF 10 초과",
}

STAT_LABELS = {
    "mean": "평균",
    "std": "표준편차",
    "min": "최솟값",
    "max": "최댓값",
    "active_fraction": "작동 비율",
    "current": "현재값",
    "delta": "직전 조사 대비 변화량",
    "delta_per_day": "일평균 변화량",
    "slope": "변화 기울기",
}


def clean_number(value):
    if pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


SOURCE_LABELS = {
    "cropping_system": ("재배 시스템", ""),
    "cultivation_area": ("재배면적(평)", "평"),
    "cal_cultivation_area": ("재배면적(㎡)", "㎡"),
    "plant_num": ("전체 재식수량(주)", "주"),
    "cal_plant_num": ("㎡당 재식수량(주/㎡)", "주/㎡"),
    "plant_density": ("재식밀도", ""),
    "cropping_season_name": ("작기명", ""),
    "days_since_crop_start": ("정식 후 경과일", "일"),
    "days_since_previous": ("직전 조사 후 경과일", "일"),
    "days_since_first_measurement": ("최초 조사 후 경과일", "일"),
    "history_count": ("누적 조사 횟수", "회"),
    "week_sin": ("연중 주차 사인값", ""),
    "week_cos": ("연중 주차 코사인값", ""),
}


def load_source_labels() -> dict[str, tuple[str, str]]:
    labels = dict(SOURCE_LABELS)
    sensor_path = WORKSPACE / "strawberry_fruit_set_prediction" / "data" / "sensor_category_cache" / "sensor_metadata.csv"
    if sensor_path.exists():
        sensor = pd.read_csv(sensor_path).fillna("")
        for row in sensor.itertuples(index=False):
            labels[str(row.variable_id)] = (str(row.com_name), str(row.unit))

    semantic_path = WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e6_semantic_v2" / "e6_semantic_mapping.csv"
    if semantic_path.exists():
        semantic = pd.read_csv(semantic_path, encoding="utf-8-sig").fillna("")
        for row in semantic.itertuples(index=False):
            labels[f"CR__{row.source_code}"] = (str(row.com_name), str(row.unit))
    return labels


def feature_parts(variable_id: str, group: str, source_labels: dict[str, tuple[str, str]]) -> dict:
    """Turn a generated feature ID into a Korean, human-readable explanation."""
    result = {
        "variable_id": variable_id,
        "group": group or "-",
        "base_variable": variable_id,
        "window": "-",
        "statistic": "원값",
    }

    sensor = re.match(r"^(?P<base>.+)__(?P<days>[137])d_(?P<stat>mean|std|min|max|active_fraction)$", variable_id)
    if sensor:
        result.update(
            base_variable=sensor.group("base"),
            window=f"직전 {sensor.group('days')}일",
            statistic=STAT_LABELS[sensor.group("stat")],
        )
    else:
        growth = re.match(
            r"^(?P<base>.+)__(?P<kind>current|delta|delta_per_day|lag[123]|roll[234]_(?:mean|std|min|max|slope))$",
            variable_id,
        )
        if growth:
            kind = growth.group("kind")
            window, statistic = "조사 시점", STAT_LABELS.get(kind, kind)
            if kind.startswith("lag"):
                window, statistic = f"{kind[3:]}회 전 조사", "관측값"
            elif kind.startswith("roll"):
                match = re.match(r"roll(?P<count>[234])_(?P<stat>.+)", kind)
                window = f"최근 {match.group('count')}회 조사"
                statistic = STAT_LABELS.get(match.group("stat"), match.group("stat"))
            result.update(base_variable=growth.group("base"), window=window, statistic=statistic)

    base = result["base_variable"]
    source_name, unit = source_labels.get(base, (base, ""))
    result["source_name"] = source_name
    result["unit"] = unit
    if result["statistic"] == "작동 비율":
        result["meaning"] = f"예측 기준일 {result['window']} 동안 {source_name}가 작동(1)한 기록의 비율"
    elif result["window"] != "-":
        result["meaning"] = f"예측 기준일 {result['window']} 동안 측정된 {source_name}의 {result['statistic']}"
    elif source_name != base:
        result["meaning"] = f"예측 기준일에 연결된 {source_name} 값"
    else:
        result["meaning"] = f"{base} 코드의 원값"
    return result


def status_for(target: str, combo: str) -> tuple[str, str | None]:
    statuses = []
    reasons = []
    for path in (OUTPUT / "summaries" / target).glob(f"*/{combo}.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        statuses.append(str(data.get("Status", "unavailable")))
        if data.get("Reason"):
            reasons.append(str(data["Reason"]))
    if not statuses:
        return "unavailable", "변수 선택 기록이 아직 생성되지 않았습니다."
    status = Counter(statuses).most_common(1)[0][0]
    reason = Counter(reasons).most_common(1)[0][0] if reasons else None
    return status, reason


def main() -> None:
    catalog = json.loads((HERE / "cache" / "combination_catalog.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUTPUT / "prepared_manifest.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(OUTPUT / "selected_features.csv", encoding="utf-8-sig")
    excluded = pd.read_csv(OUTPUT / "excluded_features.csv", encoding="utf-8-sig")
    # item_code is fixed to one crop within every target, so it was never a
    # usable predictor. Keep immutable experiment audit files unchanged and
    # omit this retired E7 candidate only from the dashboard projection.
    selected = selected[selected["Variable_ID"].ne("item_code")].copy()
    excluded = excluded[excluded["Variable_ID"].ne("item_code")].copy()
    selected = selected[selected["stage"].eq("validation")].copy()
    excluded = excluded[excluded["stage"].eq("validation")].copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    source_labels = load_source_labels()

    for target, target_spec in manifest["targets"].items():
        feature_group = {
            feature: group
            for group, features in target_spec.get("groups", {}).items()
            for feature in features
        }
        target_selected = selected[selected["target"].eq(target)]
        target_excluded = excluded[excluded["target"].eq(target)]
        combinations = {}

        for combo in catalog["combinations"]:
            combo_id = combo["combination_id"]
            sel = target_selected[target_selected["combination"].eq(combo_id)]
            exc = target_excluded[target_excluded["combination"].eq(combo_id)]
            status, reason = status_for(target, combo_id)

            selected_rows = []
            for row in sel.itertuples(index=False):
                item = feature_parts(row.Variable_ID, feature_group.get(row.Variable_ID, "-"), source_labels)
                item["variable_type"] = row.Variable_Type
                selected_rows.append(item)

            excluded_rows = []
            for row in exc.itertuples(index=False):
                item = feature_parts(row.Variable_ID, feature_group.get(row.Variable_ID, "-"), source_labels)
                item.update(
                    reason=row.Reason,
                    reason_label=REASON_LABELS.get(row.Reason, row.Reason),
                    missing_rate=clean_number(row.Missing_Rate),
                    related_variable=None if pd.isna(row.Related_Variable) else str(row.Related_Variable),
                    related_variable_name=None if pd.isna(row.Related_Variable) else source_labels.get(str(row.Related_Variable), (str(row.Related_Variable), ""))[0],
                    correlation=clean_number(row.Correlation),
                    iteration=None if pd.isna(row.Iteration) else int(row.Iteration),
                    vif=clean_number(row.VIF),
                    vif_is_infinite=bool(row.VIF_Is_Infinite) if not pd.isna(row.VIF_Is_Infinite) else False,
                )
                excluded_rows.append(item)

            reason_counts = Counter(item["reason"] for item in excluded_rows)
            if selected_rows:
                status = "success"
                reason = None
            combinations[combo_id] = {
                "combination_id": combo_id,
                "label": combo["label"],
                "groups": combo["groups"],
                "status": status,
                "reason": reason,
                "stage": "validation",
                "fit_hash": str(sel.iloc[0]["fit_hash"]) if not sel.empty else None,
                "candidate_count": len(selected_rows) + len(excluded_rows),
                "selected_count": len(selected_rows),
                "excluded_count": len(excluded_rows),
                "selected": selected_rows,
                "excluded": excluded_rows,
                "exclusion_summary": [
                    {
                        "reason": key,
                        "reason_label": REASON_LABELS.get(key, key),
                        "count": count,
                    }
                    for key, count in reason_counts.most_common()
                ],
            }

        payload = {
            "campaign": "exhaustive_e1_e7_v1",
            "target": target,
            "selection_scope": "validation_train",
            "model_scope": "shared_all_models",
            "combinations": combinations,
        }
        destination = CACHE / f"{target}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{target}: {len(combinations)} combinations -> {destination}")


if __name__ == "__main__":
    main()
