from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..report import write_xlsx
from ..stability.policy import score
from .config import SEEDS, TARGETS
from .io import atomic_json
from .runner import status


def _flat_runs(root):
    rows = []
    metrics = []
    resources = []
    for p in sorted((root / "runs").glob("*/result.json")):
        r = json.loads(p.read_text())
        rows.append(
            {
                k: (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                )
                for k, v in r.items()
                if k != "Metrics"
            }
        )
        for variant, values in r.get("Metrics", {}).items():
            for metric, value in values.items():
                metrics.append(
                    {
                        k: r.get(k)
                        for k in [
                            "Run_ID",
                            "Target_ID",
                            "Combination_ID",
                            "Variable_Groups",
                            "Model_Type",
                            "Seed",
                            "Stage",
                            "Fold_ID",
                            "N_Eval",
                            "Eval_Row_Hash",
                        ]
                    }
                    | {
                        "Prediction_Variant": variant,
                        "Metric_Name": metric.upper(),
                        "Metric_Value": value,
                    }
                )
        resources.append(
            {
                k: r.get(k)
                for k in [
                    "Run_ID",
                    "Target_ID",
                    "Combination_ID",
                    "Model_Type",
                    "Stage",
                    "Duration_Seconds",
                    "GPU_Utilization_Percent",
                    "GPU_Memory_Used_MiB",
                    "GPU_Memory_Total_MiB",
                    "GPU_After",
                ]
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(metrics), pd.DataFrame(resources)


def _selection_tables(root):
    selected = []
    excluded = []
    corr = []
    vifs = []
    for p in sorted((root / "feature_selection").glob("*/*/*.json")):
        d = json.loads(p.read_text())
        identity = d["identity"]
        base = {
            k: identity.get(k)
            for k in ["target", "combination", "stage", "fold", "fit_hash"]
        }
        for kind in ["selected_numeric", "selected_categorical"]:
            for name in d[kind]:
                selected.append(
                    base
                    | {
                        "Variable_ID": name,
                        "Variable_Type": kind.replace("selected_", ""),
                    }
                )
        for row in d["exclusions"]:
            excluded.append(base | row)
        for row in d["correlations"]:
            corr.append(base | row)
        for row in d["vif_iterations"]:
            vifs.append(base | row)
    return map(pd.DataFrame, (selected, excluded, corr, vifs))


def _baselines(root):
    rows = []
    for target in TARGETS:
        p = root / "prepared" / target
        spec = json.loads((p / "manifest.json").read_text())
        train = pd.read_parquet(p / "train.parquet")
        for split in ("validation", "test"):
            evaluation = pd.read_parquet(p / f"{split}.parquet")
            mean = np.repeat(float(train.target.mean()), len(evaluation))
            for name, pred in [("train_mean", mean)]:
                for metric, value in score(evaluation.target, pred).items():
                    rows.append(
                        {
                            "Target_ID": target,
                            "Baseline": name,
                            "Eval_Split": split,
                            "Metric_Name": metric.upper(),
                            "Metric_Value": value,
                            "N_Eval": len(evaluation),
                        }
                    )
            component = spec["components"]
            needed = [f"count_{i}__current" for i in component]
            if all(c in evaluation for c in needed):
                pred = evaluation[needed].sum(axis=1).to_numpy(float)
                for metric, value in score(evaluation.target, pred).items():
                    rows.append(
                        {
                            "Target_ID": target,
                            "Baseline": "persistence",
                            "Eval_Split": split,
                            "Metric_Name": metric.upper(),
                            "Metric_Value": value,
                            "N_Eval": len(evaluation),
                        }
                    )
    return pd.DataFrame(rows)


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def report(root):
    root = Path(root)
    runs, metrics, resources = _flat_runs(root)
    selected, excluded, corr, vifs = _selection_tables(root)
    summaries = pd.DataFrame(
        [
            json.loads(p.read_text())
            for p in sorted((root / "summaries").glob("*/*/*.json"))
        ]
    )
    selections = []
    for p in sorted((root / "selections").glob("*/*.json")):
        d = json.loads(p.read_text())
        w = d["winner"]
        selections.append(
            {
                "Target_ID": p.parent.name,
                "Model_Type": p.stem,
                **w,
                "Frozen_Features": json.dumps(d["frozen_features"], ensure_ascii=False),
                "Test_Runs": "|".join(d["test_runs"]),
            }
        )
    best = pd.DataFrame(selections)
    finished = (root / "scheduler.json").exists() and json.loads(
        (root / "scheduler.json").read_text()
    ).get("Status") == "finished"
    baselines = _baselines(root) if finished else pd.DataFrame()
    combinations = pd.read_csv(root / "combination_catalog.csv")
    quality = pd.read_csv(root / "data_quality.csv")
    # Complete machine-readable CSV/Parquet exports stay outside Excel to avoid its 1,048,576-row limit.
    full = {
        "Runs": runs,
        "Metrics": metrics,
        "Combination_Summary": summaries,
        "Best_Configurations": best,
        "Selected_Features": selected,
        "Excluded_Features": excluded,
        "Correlation_Pairs": corr,
        "VIF_Iterations": vifs,
        "Data_Quality": quality,
        "Resource_Usage": resources,
        "Baselines": baselines,
    }
    csvdir = root / "csv"
    csvdir.mkdir(exist_ok=True)
    for name, df in full.items():
        df.to_csv(csvdir / f"{name}.csv", index=False, encoding="utf-8-sig")
    hyper = (
        runs[runs.Stage.eq("internal_cv")].copy()
        if len(runs) and "Stage" in runs
        else pd.DataFrame()
    )
    root_exports = {
        "hyperparameters.csv": hyper,
        "selected_features.csv": selected,
        "excluded_features.csv": excluded,
        "vif_iterations.csv": vifs,
        "correlation_pairs.csv": corr,
        "metrics.csv": metrics,
        "resource_usage.csv": resources,
    }
    for name, df in root_exports.items():
        df.to_csv(root / name, index=False, encoding="utf-8-sig")
    if len(runs):
        runs.to_parquet(root / "runs.parquet", index=False)
    metric_summary = (
        metrics.groupby(
            [
                "Target_ID",
                "Combination_ID",
                "Model_Type",
                "Stage",
                "Prediction_Variant",
                "Metric_Name",
            ],
            dropna=False,
        )
        .Metric_Value.agg(Mean="mean", Std="std", Count="count")
        .reset_index()
        if len(metrics)
        else pd.DataFrame()
    )
    exclusion_summary = (
        excluded.groupby(["Reason", "Variable_ID"], dropna=False)
        .size()
        .rename("Count")
        .reset_index()
        if len(excluded)
        else pd.DataFrame()
    )
    excel = {
        "README": pd.DataFrame(
            [
                {
                    "Item": "Scope",
                    "Value": "127 non-empty E1-E7 combinations × 8 targets × 6 models; C0 metadata fixed",
                },
                {
                    "Item": "Selection",
                    "Value": "fit-only missingness, Pearson |r|>=0.95, iterative VIF>10",
                },
                {
                    "Item": "Test",
                    "Value": "test accessed only after target/model winner frozen",
                },
                {
                    "Item": "Status",
                    "Value": json.dumps(status(root), ensure_ascii=False),
                },
            ]
        ),
        "Combination_Catalog": combinations,
        "Combination_Summary": summaries,
        "Best_Configurations": best,
        "Metric_Summary": metric_summary,
        "Exclusion_Summary": exclusion_summary,
        "Data_Quality": quality,
        "Resource_Usage": resources,
        "Baselines": baselines,
    }
    write_xlsx(root / "AX_Exhaustive_Experiment_Report.xlsx", excel)
    progress = status(root)
    atomic_json(root / "progress.json", progress)
    (root / "Progress.md").write_text(
        "# AX 전수실험 진행 상태\n\n```json\n"
        + json.dumps(progress, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    if (progress.get("Scheduler") or {}).get("Status") == "finished":
        _handoff(root, full)
    return progress


def _handoff(root, tables):
    dest = root / "handoff"
    dest.mkdir(exist_ok=True)
    summary = tables["Combination_Summary"]
    best = tables["Best_Configurations"]
    excluded = tables["Excluded_Features"]
    summary.to_csv(dest / "combination_summary.csv", index=False, encoding="utf-8-sig")
    best.to_csv(dest / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if len(excluded):
        excluded.groupby(["Reason", "Variable_ID"], dropna=False).size().rename(
            "Exclusion_Count"
        ).reset_index().to_csv(
            dest / "feature_exclusion_summary.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=["Reason", "Variable_ID", "Exclusion_Count"]).to_csv(
            dest / "feature_exclusion_summary.csv", index=False, encoding="utf-8-sig"
        )
    bundle = {
        "scope": {"combinations": 127, "targets": 8, "models": 6, "seeds": list(SEEDS)},
        "best_configurations": best.replace({np.nan: None}).to_dict("records"),
        "metrics": tables["Metrics"][
            tables["Metrics"].Stage.isin(["validation", "test"])
        ]
        .replace({np.nan: None})
        .to_dict("records")
        if len(tables["Metrics"])
        else [],
        "baselines": tables["Baselines"].replace({np.nan: None}).to_dict("records"),
    }
    atomic_json(dest / "dashboard_bundle.json", bundle)
    atomic_json(
        dest / "dashboard_schema.json",
        {
            "version": 1,
            "metric_units": {
                "RMSE": "target count",
                "MAE": "target count",
                "R2": "unitless",
            },
            "selection_order": [
                "validation mean RMSE",
                "validation mean MAE",
                "feature count",
                "combination ID",
            ],
            "test_used_for_selection": False,
        },
    )
    prompt = """# Google Antigravity 웹페이지 업데이트 요청\n\n`handoff/dashboard_bundle.json`과 이 폴더의 CSV만 근거로 기존 생육예측분석 페이지의 **실험 결과 및 실험 설정 표시 영역만** 갱신해 주세요. 127개 조합, 8개 타깃, 6개 모델, 3개 시드, 검증 기준 선택과 최종 테스트 결과를 구분하고, 선택·제외 변수와 상관계수/VIF 제외 사유를 조회할 수 있게 하세요. `ineligible`과 `failed`를 0점으로 표시하지 마세요. 테스트 결과로 순위를 다시 매기지 마세요.\n\n사용자가 지정한 영역 외의 다른 페이지·탭·UI·스타일·문구·데이터 파일은 절대 수정하지 마세요. 기존 관측 현황 및 분석 결론 화면도 변경하지 마세요. 변경 파일 목록과 데이터 행 수 검증 결과를 답변에 적어 주세요.\n"""
    (dest / "ANTIGRAVITY_UPDATE_PROMPT.md").write_text(prompt)
    files = [p for p in dest.iterdir() if p.is_file() and p.name != "checksums.json"]
    atomic_json(dest / "checksums.json", {p.name: _sha(p) for p in sorted(files)})
