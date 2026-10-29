from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import file_hash, growth_features
from ..exhaustive.engine import temporal_folds
from ..exhaustive.io import atomic_json
from .config import TARGETS


BASE_LABELS = {
    "count_1": "1화방 꽃수·착과수",
    "count_2": "2화방 꽃수·착과수",
    "count_3": "3화방 꽃수·착과수",
    "current_max_truss_number": "현재 조사된 최고 화방 번호",
    "observed_truss_count": "조사된 화방 수",
    "positive_truss_count": "값이 0보다 큰 화방 수",
    "leaf_len": "엽장",
    "leaf_cnt": "엽수",
    "leaf_wdth": "엽폭",
    "ptl_len": "엽병장",
    "stem_dmtr": "줄기 직경",
    "crwn_dmtr": "관부 직경",
    "plnt_hgt": "초장",
    "grw_len": "생장 길이",
    "flclstr_hgt": "화방 높이",
}


def _definition(variable: str):
    base, suffix = variable.split("__", 1)
    label = BASE_LABELS.get(base, base)
    if suffix == "current":
        meaning = "예측 기준일 현재 조사값"
    elif suffix == "delta":
        meaning = "현재값과 직전 조사값의 차이"
    elif suffix == "delta_per_day":
        meaning = "직전 조사 이후 하루당 변화량"
    elif suffix.startswith("lag"):
        meaning = f"현재보다 {suffix[3:]}회 전 조사값"
    else:
        match = re.fullmatch(r"roll(\d+)_(mean|std|min|max|slope)", suffix)
        names = {"mean": "평균", "std": "표준편차", "min": "최솟값", "max": "최댓값", "slope": "일당 변화 기울기"}
        meaning = f"최근 {match.group(1)}회 조사값의 {names[match.group(2)]}" if match else suffix
    return base, label, meaning


def _synthetic_future_leakage_test():
    frame = pd.DataFrame({
        "facility_id": ["F"] * 5,
        "crop_sn": ["C"] * 5,
        "sample_num": ["1"] * 5,
        "feature_date": pd.date_range("2024-01-01", periods=5, freq="7D"),
        "count_1": [1.0, 2.0, 4.0, 7.0, 11.0],
    })
    first = growth_features(frame, ["facility_id", "crop_sn", "sample_num"], ["count_1"])
    changed = frame.copy()
    changed.loc[4, "count_1"] = 9999.0
    second = growth_features(changed, ["facility_id", "crop_sn", "sample_num"], ["count_1"])
    columns = [column for column in first if column.startswith("count_1__")]
    unchanged_past = np.allclose(first.loc[:3, columns], second.loc[:3, columns], equal_nan=True)
    return {
        "Test": "future_value_mutation_does_not_change_past_features",
        "Passed": bool(unchanged_past),
        "Checked_Past_Rows": 4,
        "Feature_Count": len(columns),
    }


def run_audit(root: Path):
    root = Path(root)
    prepared_manifest = json.loads((root / "prepared_manifest.json").read_text())
    catalog_rows, coverage_rows, preflight_rows, target_summary = [], [], [], {}
    forbidden = {"target", "target_date", "target_truss_1", "target_truss_2", "target_truss_3"}
    structural_pass = True

    for target in TARGETS:
        spec = prepared_manifest["targets"][target]
        for group in ("E1", "E2"):
            for variable in spec["groups"][group]:
                base, label, meaning = _definition(variable)
                catalog_rows.append({
                    "Target_ID": target,
                    "Variable_Group": group,
                    "Variable_ID": variable,
                    "Source_Base": base,
                    "Source_Label_KO": label,
                    "Derived_Definition_KO": meaning,
                })
        leaked = sorted((set(spec["groups"]["E1"]) | set(spec["groups"]["E2"])) & forbidden)
        structural_pass &= not leaked and bool(spec["groups"]["E1"]) and bool(spec["groups"]["E2"])
        split_frames = {}
        for split in ("train", "validation", "test"):
            path = root / "prepared" / target / f"{split}.parquet"
            if file_hash(path) != spec["dataset_hashes"][split]:
                raise ValueError(f"Prepared hash mismatch: {target}/{split}")
            frame = pd.read_parquet(path)
            split_frames[split] = frame
            date_ok = bool((pd.to_datetime(frame.target_date) > pd.to_datetime(frame.feature_date)).all())
            structural_pass &= date_ok and frame.row_id.is_unique
            for group in ("E1", "E2"):
                for variable in spec["groups"][group]:
                    values = pd.to_numeric(frame[variable], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    coverage_rows.append({
                        "Target_ID": target,
                        "Split": split,
                        "Variable_Group": group,
                        "Variable_ID": variable,
                        "Rows": len(frame),
                        "Missing_Rate": float(values.isna().mean()),
                        "Unique_Values": int(values.nunique(dropna=True)),
                    })

        fold_counts = {"E1": [], "E2": []}
        for fold, fit, _ in temporal_folds(split_frames["train"]):
            for group in ("E1", "E2"):
                survivors = 0
                for variable in spec["groups"][group]:
                    values = pd.to_numeric(fit[variable], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    missing = float(values.isna().mean())
                    unique = int(values.nunique(dropna=True))
                    eligible = missing <= 0.50 and unique >= 2
                    survivors += int(eligible)
                    preflight_rows.append({
                        "Target_ID": target,
                        "Fold_ID": fold,
                        "Variable_Group": group,
                        "Variable_ID": variable,
                        "Fit_Rows": len(fit),
                        "Missing_Rate": missing,
                        "Unique_Values": unique,
                        "Eligible_Before_Correlation_VIF": eligible,
                        "Reason": "eligible" if eligible else ("missing_over_50_percent" if missing > 0.50 else "constant_or_empty"),
                    })
                fold_counts[group].append(survivors)
        target_summary[target] = {
            "e1_feature_count": len(spec["groups"]["E1"]),
            "e2_feature_count": len(spec["groups"]["E2"]),
            "e1_survivors_before_correlation_vif_by_fold": fold_counts["E1"],
            "e2_survivors_before_correlation_vif_by_fold": fold_counts["E2"],
            "e1_usable_all_folds": all(value > 0 for value in fold_counts["E1"]),
            "e2_usable_all_folds": all(value > 0 for value in fold_counts["E2"]),
            "leaked_columns": leaked,
        }

    future_test = _synthetic_future_leakage_test()
    row_audit = pd.read_csv(root / "row_identity_audit.csv")
    row_identity_pass = bool(row_audit.Match.all() and len(row_audit) == 24)
    passed = structural_pass and row_identity_pass and future_test["Passed"]
    pd.DataFrame(catalog_rows).drop_duplicates().to_csv(root / "e1_e2_feature_catalog.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage_rows).to_csv(root / "e1_e2_coverage_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(preflight_rows).to_csv(root / "e1_e2_cv_preflight.csv", index=False, encoding="utf-8-sig")
    result = {
        "Status": "passed" if passed else "failed",
        "Structural_Checks_Passed": structural_pass,
        "V1_Row_Identity_Passed": row_identity_pass,
        "Future_Leakage_Test": future_test,
        "Target_Summary": target_summary,
        "Note": "구간별 생존 수가 0인 변수군은 생성 오류가 아니라 해당 학습 구간의 관측 부족으로 실행 시 ineligible/failed가 될 수 있습니다.",
    }
    atomic_json(root / "e1_e2_preflight_audit.json", result)
    lines = [
        "# E1·E2 사전 감사 결과",
        "",
        f"- 감사 상태: **{result['Status']}**",
        "- 기존 v1 평가 행 24개 타깃·분할 묶음의 Row ID를 모두 동일하게 유지했습니다.",
        "- 모든 입력 행에서 기준일보다 다음 조사일이 뒤에 있으며, 미래값 변경이 과거 파생변수에 영향을 주지 않는 테스트를 통과했습니다.",
        "",
        "| 타깃 | E1 후보 수 | E2 후보 수 | E1 CV 생존 수 | E2 CV 생존 수 |",
        "|---|---:|---:|---|---|",
    ]
    for target, item in target_summary.items():
        lines.append(f"| {target} | {item['e1_feature_count']} | {item['e2_feature_count']} | {item['e1_survivors_before_correlation_vif_by_fold']} | {item['e2_survivors_before_correlation_vif_by_fold']} |")
    (root / "E1_E2_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("E1/E2 preflight audit failed; training is blocked")
    return result
