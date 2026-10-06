#!/usr/bin/env python3
"""Publish the completed E1/E2 quick-screening campaign to the dashboard.

The source campaign is intentionally kept separate from the original
three-seed experiment and the semantic-E6 experiment.  Its scores are valid
for ranking combinations within this campaign only because it uses seed 42
and capped training epochs.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CAMPAIGN_KEY = "growth_history_quick_screening"
INTERNAL_CAMPAIGN_ID = "exhaustive_e1_e2_deadline_quick_v1"
MODELS = ["poisson", "random_forest", "catboost", "mlp", "tabm", "tft"]
SEEDS = [42]

TARGET_NAMES_KR = {
    "tomato_first": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "1화방 꽃수", "part": "first"},
    "tomato_second": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "2화방 꽃수", "part": "second"},
    "tomato_third": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "3화방 꽃수", "part": "third"},
    "tomato_sum123": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "1~3화방 합계 꽃수", "part": "sum123"},
    "strawberry_first": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "1화방 착과수", "part": "first"},
    "strawberry_second": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "2화방 착과수", "part": "second"},
    "strawberry_third": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "3화방 착과수", "part": "third"},
    "strawberry_sum123": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "1~3화방 합계 착과수", "part": "sum123"},
}

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

SOURCE_LABELS = {
    "count_1": ("1화방 꽃수·착과수", "개"),
    "count_2": ("2화방 꽃수·착과수", "개"),
    "count_3": ("3화방 꽃수·착과수", "개"),
    "current_max_truss_number": ("현재 조사된 최고 화방 번호", "화방"),
    "observed_truss_count": ("조사된 화방 수", "개"),
    "positive_truss_count": ("값이 0보다 큰 화방 수", "개"),
    "leaf_len": ("엽장", "cm"),
    "leaf_cnt": ("엽수", "개"),
    "leaf_wdth": ("엽폭", "cm"),
    "ptl_len": ("엽병장", "cm"),
    "stem_dmtr": ("줄기 직경", "mm"),
    "crwn_dmtr": ("관부 직경", "mm"),
    "plnt_hgt": ("초장", "cm"),
    "grw_len": ("생장 길이", "cm"),
    "flclstr_hgt": ("화방 높이", "cm"),
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


def _first_existing(configured: str | None, candidates: list[Path]) -> Path:
    paths = [Path(configured).expanduser()] if configured else []
    paths.extend(candidates)
    for path in paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("Required experiment source not found: " + ", ".join(map(str, paths)))


OUTPUT = _first_existing(
    os.environ.get("AX_E1_E2_QUICK_OUTPUTS"),
    [
        WORKSPACE / "ax_catalog_experiments" / "outputs" / INTERNAL_CAMPAIGN_ID,
        WORKSPACE / "AXData" / "ax_catalog_experiments" / "outputs" / INTERNAL_CAMPAIGN_ID,
    ],
)
PREPARED_ROOT = _first_existing(
    os.environ.get("AX_E1_E2_PREPARED_ROOT"),
    [
        WORKSPACE / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_backfill_v1",
        WORKSPACE / "AXData" / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e2_backfill_v1",
    ],
)

CACHE = HERE / "cache"
CAMPAIGN_CACHE = CACHE / "campaigns" / CAMPAIGN_KEY
INDEX_FILE = CAMPAIGN_CACHE / "experiment_index.json"
FEATURE_CACHE = CAMPAIGN_CACHE / "feature_selections"
CATALOG_FILE = CACHE / "combination_catalog.json"
CAMPAIGN_CATALOG_FILE = CACHE / "campaign_catalog.json"
BASELINE_INDEX_FILE = CACHE / "experiment_index.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_block(summary: dict, variant: str) -> dict:
    prefix = "Validation" if variant == "bounded" else None
    run = (summary.get("Runs") or [{}])[0]
    source = run.get("Bounded" if variant == "bounded" else "Raw") or {}
    result = {}
    for metric in ("rmse", "mae", "r2", "ccc"):
        if prefix:
            mean = summary.get(f"{prefix}_{metric.upper()}_Mean")
            std = summary.get(f"{prefix}_{metric.upper()}_Std")
        else:
            mean = source.get(metric)
            std = 0.0 if mean is not None else None
        mean = _finite(mean)
        std = _finite(std)
        if mean is not None:
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std or 0.0
    return result


def _source_labels() -> dict[str, tuple[str, str]]:
    labels = dict(SOURCE_LABELS)
    sensor_candidates = [
        WORKSPACE / "strawberry_fruit_set_prediction" / "data" / "sensor_category_cache" / "sensor_metadata.csv",
        WORKSPACE / "AXData" / "strawberry_fruit_set_prediction" / "data" / "sensor_category_cache" / "sensor_metadata.csv",
    ]
    for path in sensor_candidates:
        if path.exists():
            sensor = pd.read_csv(path).fillna("")
            for row in sensor.itertuples(index=False):
                labels[str(row.variable_id)] = (str(row.com_name), str(row.unit))
            break

    feature_catalog = PREPARED_ROOT / "e1_e2_feature_catalog.csv"
    if feature_catalog.exists():
        frame = pd.read_csv(feature_catalog, encoding="utf-8-sig").fillna("")
        for row in frame.itertuples(index=False):
            labels[str(row.Source_Base)] = (str(row.Source_Label_KO), labels.get(str(row.Source_Base), ("", ""))[1])
    return labels


def _feature_parts(variable_id: str, group: str, labels: dict[str, tuple[str, str]]) -> dict:
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

    source_name, unit = labels.get(result["base_variable"], (result["base_variable"], ""))
    result["source_name"] = source_name
    result["unit"] = unit
    if result["statistic"] == "작동 비율":
        result["meaning"] = f"예측 기준일 {result['window']} 동안 {source_name}가 작동한 기록의 비율"
    elif result["statistic"] == "현재값":
        result["meaning"] = f"예측 기준일에 조사된 {source_name} 현재값"
    elif result["statistic"] == "직전 조사 대비 변화량":
        result["meaning"] = f"{source_name} 현재값과 직전 조사값의 차이"
    elif result["statistic"] == "일평균 변화량":
        result["meaning"] = f"직전 조사 이후 {source_name}의 하루당 변화량"
    elif result["window"] != "-":
        result["meaning"] = f"{result['window']} {source_name}의 {result['statistic']}"
    elif source_name != result["base_variable"]:
        result["meaning"] = f"예측 기준일에 연결된 {source_name} 값"
    else:
        result["meaning"] = f"{result['base_variable']} 코드의 원값"
    return result


def _summary_records() -> list[dict]:
    records = []
    for path in sorted((OUTPUT / "summaries").glob("*/*/COMBO_*.json")):
        records.append(_read_json(path))
    expected = 127 * len(TARGET_NAMES_KR) * len(MODELS)
    if len(records) != expected:
        raise ValueError(f"Quick campaign is incomplete: {len(records)}/{expected}")
    return records


def build_index(records: list[dict]) -> None:
    catalog = _read_json(CATALOG_FILE)
    baseline = _read_json(BASELINE_INDEX_FILE)
    combo_map = {item["combination_id"]: item for item in catalog["combinations"]}
    targets = {}
    for target_id, info in TARGET_NAMES_KR.items():
        base_target = baseline["targets"][target_id]
        targets[target_id] = {
            "target_id": target_id,
            **info,
            "status": "completed",
            "frozen": {"overall": None, "winners": {}},
            "groups": deepcopy(base_target.get("groups", {})),
            "results": {},
        }

    status_counts = Counter()
    success_candidates: dict[tuple[str, str], list[tuple]] = {}
    for summary in records:
        target_id = str(summary["Target_ID"])
        combo_id = str(summary["Combination_ID"])
        model = str(summary["Model_Type"])
        status = str(summary.get("Status", "unknown"))
        status_counts[status] += 1
        result = {
            "group_id": combo_id,
            "model": model,
            "split": "validation",
            "completed_seeds": 1 if status == "success" else 0,
            "status": status,
            "reason": summary.get("Reason"),
            "params": summary.get("Params") or {},
            "best_epoch": summary.get("Fixed_Epochs"),
            "n_eval": int(summary.get("N_Eval") or 0),
            "feature_count": summary.get("Feature_Count"),
            "bounded": _metric_block(summary, "bounded") if status == "success" else {},
            "raw": _metric_block(summary, "raw") if status == "success" else {},
            "seeds": {"42": {"seed": 42, "status": "success"}} if status == "success" else {},
            "source": "e1_e2_quick_screening",
            "source_label": "E1·E2 보완 단일 시드 빠른 선별",
            "method": summary.get("Method"),
            "comparability": summary.get("Comparability"),
        }
        targets[target_id]["results"][f"{combo_id}::{model}::validation"] = result
        if status == "success" and result["bounded"].get("rmse_mean") is not None:
            success_candidates.setdefault((target_id, model), []).append(
                (
                    result["bounded"]["rmse_mean"],
                    result["bounded"].get("mae_mean", float("inf")),
                    result["feature_count"] if result["feature_count"] is not None else float("inf"),
                    combo_id,
                )
            )

    for (target_id, model), candidates in success_candidates.items():
        winner = min(candidates)[3]
        targets[target_id]["frozen"]["winners"][model] = {"group": winner}

    generated_at = datetime.now(timezone.utc).isoformat()
    index = {
        "campaign": {
            "id": INTERNAL_CAMPAIGN_ID,
            "dashboard_id": CAMPAIGN_KEY,
            "name": "E1·E2 보완 빠른 선별",
            "description": "과거 꽃수·착과수 이력(E1)과 영양생장(E2)을 정상 생성한 127개 조합 1차 선별 결과",
            "status": "completed",
            "has_real_results": True,
            "partial_results": False,
            "snapshot_generated_at": generated_at,
            "snapshot_note": "단일 시드와 제한된 학습 횟수를 사용한 조합 선별 결과입니다. 정식 3시드 결과와 직접 혼합해 순위를 산정하지 않습니다.",
            "method": "deadline_single_seed_epoch_capped_screening",
            "comparability": "within_same_campaign_only",
            "total_combinations": 127,
            "total_targets": 8,
            "total_models": 6,
            "total_seeds": 1,
            "seeds": SEEDS,
            "expected_combination_summaries": 6096,
            "processed_combination_summaries": len(records),
            "progress_percent": 100.0,
            "summary_status_counts": dict(sorted(status_counts.items())),
            "published_result_groups": len(records),
            "published_success_groups": status_counts.get("success", 0),
            "cached_prediction_runs": 0,
        },
        "targets": targets,
    }
    CAMPAIGN_CACHE.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def build_feature_selections(records: list[dict]) -> None:
    catalog = _read_json(CATALOG_FILE)
    prepared = _read_json(PREPARED_ROOT / "prepared_manifest.json")
    labels = _source_labels()
    summary_lookup = {
        (str(item["Target_ID"]), str(item["Combination_ID"])): item
        for item in records
    }
    if FEATURE_CACHE.exists():
        shutil.rmtree(FEATURE_CACHE)
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

    for target_id in TARGET_NAMES_KR:
        target_spec = prepared["targets"][target_id]
        feature_group = {
            feature: group
            for group, features in target_spec.get("groups", {}).items()
            for feature in features
        }
        target_cache = FEATURE_CACHE / target_id
        target_cache.mkdir(parents=True, exist_ok=True)
        for combo in catalog["combinations"]:
            combo_id = combo["combination_id"]
            source = _read_json(OUTPUT / "feature_selection" / target_id / f"{combo_id}.json")
            summary = summary_lookup.get((target_id, combo_id), {})
            selected_ids = [
                item for item in source.get("selected_numeric", []) + source.get("selected_categorical", [])
                if item != "item_code"
            ]
            selected_rows = []
            categorical = set(source.get("selected_categorical", []))
            for variable_id in selected_ids:
                item = _feature_parts(variable_id, feature_group.get(variable_id, "-"), labels)
                item["variable_type"] = "categorical" if variable_id in categorical else "numeric"
                selected_rows.append(item)

            excluded_rows = []
            for row in source.get("exclusions", []):
                variable_id = str(row.get("Variable_ID", ""))
                if not variable_id or variable_id == "item_code":
                    continue
                item = _feature_parts(variable_id, feature_group.get(variable_id, "-"), labels)
                related = row.get("Related_Variable")
                related_parts = _feature_parts(str(related), feature_group.get(str(related), "-"), labels) if related else None
                reason = str(row.get("Reason", "unknown"))
                item.update(
                    reason=reason,
                    reason_label=REASON_LABELS.get(reason, reason),
                    missing_rate=_finite(row.get("Missing_Rate")),
                    related_variable=str(related) if related else None,
                    related_variable_name=related_parts.get("source_name") if related_parts else None,
                    correlation=_finite(row.get("Correlation")),
                    iteration=int(row["Iteration"]) if row.get("Iteration") is not None else None,
                    vif=_finite(row.get("VIF")),
                    vif_is_infinite=bool(row.get("VIF_Is_Infinite", False)) or str(row.get("VIF", "")).lower() == "inf",
                )
                excluded_rows.append(item)

            reason_counts = Counter(item["reason"] for item in excluded_rows)
            status = "success" if selected_rows else str(summary.get("Status", "ineligible"))
            detail = {
                "combination_id": combo_id,
                "label": combo["label"],
                "groups": combo["groups"],
                "status": status,
                "reason": None if selected_rows else summary.get("Reason"),
                "stage": "validation",
                "fold": None,
                "fit_hash": None,
                "candidate_count": len(selected_rows) + len(excluded_rows),
                "selected_count": len(selected_rows),
                "excluded_count": len(excluded_rows),
                "selected": selected_rows,
                "excluded": excluded_rows,
                "exclusion_summary": [
                    {"reason": reason, "reason_label": REASON_LABELS.get(reason, reason), "count": count}
                    for reason, count in reason_counts.most_common()
                ],
                "source_label": "빠른 선별 캠페인의 검증 학습 구간 변수 선택 기록",
                "exclusion_note": "같은 타깃·조합의 6개 모델이 공통으로 사용한 변수 선택 결과입니다.",
            }
            (target_cache / f"{combo_id}.json").write_text(
                json.dumps(detail, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )


def update_campaign_catalog() -> None:
    catalog = _read_json(CAMPAIGN_CATALOG_FILE)
    quick = {
        "id": CAMPAIGN_KEY,
        "internal_campaign_id": INTERNAL_CAMPAIGN_ID,
        "name": "E1·E2 보완 빠른 선별",
        "short_name": "E1·E2 보완 선별",
        "description": "과거 꽃수·착과수 이력과 영양생장 변수를 보완한 127개 조합의 1차 선별 결과",
        "why_separate": "기존 준비 캐시에 반영되지 않았던 E1·E2 파생변수를 정상 생성한 뒤, 모든 조합의 가능성을 먼저 확인하기 위해 별도 실험으로 수행했습니다.",
        "scope_note": "8개 타깃 × 6개 모델 × 127개 조합을 Seed 42와 제한된 학습 횟수로 평가했습니다. 6,096건 중 6,048건이 성공했고 48건은 모든 후보 변수가 제외되어 실행 불가였습니다. 이 결과는 캠페인 내부의 후보 선별용이며 정식 3시드 검증이 진행 중입니다.",
        "status": "completed",
        "progress_percent": 100.0,
        "completed_results": 6096,
        "successful_results": 6048,
        "ineligible_results": 48,
        "total_results": 6096,
        "total_seeds": 1,
        "seeds": SEEDS,
        "data_path": f"./cache/campaigns/{CAMPAIGN_KEY}/experiment_index.json",
        "predictions_path": f"./cache/campaigns/{CAMPAIGN_KEY}/predictions",
        "feature_selections_path": f"./cache/campaigns/{CAMPAIGN_KEY}/feature_selections",
        "feature_selection_layout": "by_combination",
    }
    campaigns = [item for item in catalog.get("campaigns", []) if item.get("id") != CAMPAIGN_KEY]
    catalog["default_campaign"] = CAMPAIGN_KEY
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    catalog["campaigns"] = [quick, *campaigns]
    CAMPAIGN_CATALOG_FILE.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    report = _read_json(OUTPUT / "completion_report.json")
    if int(report.get("Completed", 0)) != int(report.get("Total", -1)) or int(report.get("Total", 0)) != 6096:
        raise ValueError("Quick campaign completion report is not 6096/6096")
    records = _summary_records()
    build_index(records)
    build_feature_selections(records)
    update_campaign_catalog()
    print(f"Published {len(records)} results to {CAMPAIGN_CACHE}")


if __name__ == "__main__":
    main()
