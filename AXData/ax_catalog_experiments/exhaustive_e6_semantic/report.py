from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..exhaustive.config import MODELS, SEEDS, TARGETS
from ..exhaustive.io import atomic_json
from ..exhaustive.report import _flat_runs, _selection_tables
from ..report import write_xlsx
from .combinations import composite_combinations, reused_v1_combinations, semantic_combinations
from .config import CAMPAIGN_ID, V1_ROOT
from .runner import status


def _run_records(root: Path):
    return [json.loads(p.read_text()) for p in sorted((Path(root) / "runs").glob("*/result.json"))]


def _metric_summary(records, target, model, combination, stage="validation"):
    rows = [
        r for r in records
        if r.get("Target_ID") == target and r.get("Model_Type") == model
        and r.get("Combination_ID") == combination and r.get("Stage") == stage
        and r.get("Status") == "success"
    ]
    result = {"Seed_Count": len(rows), "Run_IDs": "|".join(r["Run_ID"] for r in rows)}
    for metric in ("rmse", "mae", "r2", "ccc"):
        values = [r.get("Metrics", {}).get("bounded", {}).get(metric) for r in rows]
        values = [float(x) for x in values if x is not None and np.isfinite(x)]
        result[metric.upper()] = float(np.mean(values)) if values else np.nan
        result[metric.upper() + "_Std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else np.nan)
    result["Feature_Count"] = min((r.get("Feature_Count", 0) for r in rows), default=np.nan)
    return result


def _summary(root, target, model, combination):
    path = Path(root) / "summaries" / target / model / f"{combination}.json"
    return json.loads(path.read_text()) if path.exists() else {"Status": "missing", "Reason": "summary_missing"}


def build_comparison(root: Path):
    root = Path(root)
    v1_runs, v2_runs = _run_records(V1_ROOT), _run_records(root)
    rows, transitions = [], []
    for target in TARGETS:
        for model in MODELS:
            for combo in semantic_combinations():
                cid = combo["Combination_ID"]
                old_status = _summary(V1_ROOT, target, model, cid)
                new_status = _summary(root, target, model, cid)
                old_metrics = _metric_summary(v1_runs, target, model, cid)
                new_metrics = _metric_summary(v2_runs, target, model, cid)
                row = {
                    "Target_ID": target, "Model_Type": model, "Combination_ID": cid,
                    "Variable_Groups": "|".join(combo["Groups"]),
                    "V1_Status": old_status.get("Status"), "V1_Reason": old_status.get("Reason", ""),
                    "V2_Status": new_status.get("Status"), "V2_Reason": new_status.get("Reason", ""),
                    "V1_Feature_Count": old_metrics["Feature_Count"],
                    "V2_Feature_Count": new_metrics["Feature_Count"],
                }
                for metric in ("RMSE", "MAE", "R2", "CCC"):
                    old, new = old_metrics[metric], new_metrics[metric]
                    row[f"V1_{metric}"] = old
                    row[f"V2_{metric}"] = new
                    row[f"Delta_{metric}_V2_minus_V1"] = new - old if pd.notna(old) and pd.notna(new) else np.nan
                    row[f"Improved_{metric}"] = (
                        (new < old) if metric in {"RMSE", "MAE"} else (new > old)
                    ) if pd.notna(old) and pd.notna(new) else None
                rows.append(row)
                transitions.append(
                    {
                        "Target_ID": target, "Model_Type": model, "Combination_ID": cid,
                        "Variable_Groups": "|".join(combo["Groups"]),
                        "V1_Status": old_status.get("Status"), "V2_Status": new_status.get("Status"),
                        "Transition": f"{old_status.get('Status')}→{new_status.get('Status')}",
                        "V1_Reason": old_status.get("Reason", ""), "V2_Reason": new_status.get("Reason", ""),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _composite_results(root: Path):
    v1_runs, v2_runs = _run_records(V1_ROOT), _run_records(root)
    records = []
    reused = {c["Combination_ID"] for c in reused_v1_combinations()}
    for target in TARGETS:
        for model in MODELS:
            selection_path = root / "selections" / target / f"{model}.json"
            winner = json.loads(selection_path.read_text())["winner"]["Combination_ID"] if selection_path.exists() else None
            for combo in composite_combinations():
                cid = combo["Combination_ID"]
                source = "reused_from_v1" if cid in reused else "semantic_e6_v2"
                source_root = V1_ROOT if source == "reused_from_v1" else root
                summary = _summary(source_root, target, model, cid)
                metrics = _metric_summary(v1_runs if source == "reused_from_v1" else v2_runs, target, model, cid)
                records.append(
                    {
                        "Target_ID": target, "Model_Type": model, "Combination_ID": cid,
                        "Variable_Groups": combo["Groups"], "Group_Count": combo["Group_Count"],
                        "Source": source, "Status": summary.get("Status", "missing"),
                        "Reason": summary.get("Reason", ""), "Is_Winner": cid == winner,
                        "Params": summary.get("Params", {}), "Best_Epoch": summary.get("Best_Epoch"),
                        **metrics,
                    }
                )
    return records


def _sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def report(root: Path):
    root = Path(root)
    runs, metrics, resources = _flat_runs(root)
    selected, excluded, correlations, vifs = _selection_tables(root)
    comparison, transitions = build_comparison(root)
    composite_records = _composite_results(root)
    composite = pd.DataFrame(
        [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in composite_records]
    )
    selections = pd.DataFrame(
        [
            {"Target_ID": path.parent.name, "Model_Type": path.stem, **json.loads(path.read_text())["winner"], "Winner_Source": json.loads(path.read_text()).get("winner_source")}
            for path in sorted((root / "selections").glob("*/*.json"))
        ]
    )
    exports = {
        "Runs": runs, "Metrics": metrics, "Composite_Combination_Summary": composite,
        "Best_Configurations": selections, "Selected_Features": selected,
        "Excluded_Features": excluded, "Correlation_Pairs": correlations,
        "VIF_Iterations": vifs, "Resource_Usage": resources,
        "V1_V2_Comparison": comparison, "Status_Transition": transitions,
        "E6_Coverage_Audit": pd.read_csv(root / "e6_coverage_audit.csv"),
        "Semantic_Feature_Catalog": pd.read_csv(root / "semantic_feature_catalog.csv"),
    }
    csv_dir = root / "csv"
    csv_dir.mkdir(exist_ok=True)
    for name, frame in exports.items():
        frame.to_csv(csv_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(root / "comparison_v1_v2.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(root / "status_transition.csv", index=False, encoding="utf-8-sig")
    if len(runs):
        runs.to_parquet(root / "runs.parquet", index=False)
    write_xlsx(
        root / "AX_E6_Semantic_v2_Experiment_Report.xlsx",
        {
            "README": pd.DataFrame([
                {"Item": "Scope", "Value": "16 semantic-E6 combinations newly trained; 15 non-E6 combinations reused from v1"},
                {"Item": "Selection", "Value": "Composite validation ranking over 31 combinations; test used after winner freeze only"},
                {"Item": "Status", "Value": json.dumps(status(root), ensure_ascii=False)},
            ]),
            "V1_V2_Comparison": comparison,
            "Status_Transition": transitions,
            "Composite_Summary": composite,
            "Best_Configurations": selections,
            "E6_Coverage": exports["E6_Coverage_Audit"],
            "Semantic_Features": exports["Semantic_Feature_Catalog"],
        },
    )
    progress = status(root)
    atomic_json(root / "progress.json", progress)
    (root / "Progress.md").write_text("# E6 의미 통합 v2 진행 상태\n\n```json\n" + json.dumps(progress, ensure_ascii=False, indent=2) + "\n```\n")
    if (progress.get("Scheduler") or {}).get("Status") == "finished":
        _handoff(root, composite_records, comparison, transitions)
    return progress


def _handoff(root: Path, composite_records, comparison, transitions):
    dest = root / "handoff"
    dest.mkdir(exist_ok=True)
    pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in composite_records]).to_csv(dest / "combination_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(dest / "comparison_v1_v2.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(dest / "status_transition.csv", index=False, encoding="utf-8-sig")
    bundle = _json_clean({
        "campaign": {"id": CAMPAIGN_ID, "name": "의미 통합 E6 v2", "new_combinations": 16, "composite_combinations": 31, "targets": 8, "models": 6, "seeds": list(SEEDS)},
        "results": composite_records,
        "comparison_v1_v2": comparison.to_dict("records"),
        "status_transition": transitions.to_dict("records"),
        "test_used_for_selection": False,
    })
    atomic_json(dest / "dashboard_bundle.json", bundle)
    atomic_json(dest / "dashboard_schema.json", {"version": 2, "campaign_key_required": True, "lower_is_better": ["RMSE", "MAE"], "higher_is_better": ["R2", "CCC"], "sources": ["reused_from_v1", "semantic_e6_v2"]})
    prompt = """# Google Antigravity 대시보드 업데이트 요청\n\n`handoff/dashboard_bundle.json`, `comparison_v1_v2.csv`, `status_transition.csv`만 근거로 기존 생육예측분석 페이지의 실험 결과 비교 영역에 실험 버전 선택과 E6 v1/v2 차이표를 추가하세요. RMSE·MAE 감소와 R²·CCC 증가를 개선으로 표시하고, `reused_from_v1`은 ‘v1 결과 재사용’으로 표시하세요. 테스트는 각 캠페인의 최종 우승 조합에만 표시하세요.\n\n기존 v1 수치와 관측 현황, 다른 탭, 기존 UI·스타일·문구·데이터 파일은 수정하지 마세요.\n"""
    (dest / "ANTIGRAVITY_UPDATE_PROMPT.md").write_text(prompt)
    files = [p for p in dest.iterdir() if p.is_file() and p.name != "checksums.json"]
    atomic_json(dest / "checksums.json", {p.name: _sha(p) for p in sorted(files)})
