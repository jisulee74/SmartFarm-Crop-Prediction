"""Publish a completed semantic-E6 campaign into campaign-namespaced static cache."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CAMPAIGN = "exhaustive_e6_semantic_v2"
TARGET_INFO = {
    "tomato_first": ("tomato", "토마토", "1화방 꽃수", "first"),
    "tomato_second": ("tomato", "토마토", "2화방 꽃수", "second"),
    "tomato_third": ("tomato", "토마토", "3화방 꽃수", "third"),
    "tomato_sum123": ("tomato", "토마토", "1~3화방 합계 꽃수", "sum123"),
    "strawberry_first": ("strawberry", "딸기", "1화방 착과수", "first"),
    "strawberry_second": ("strawberry", "딸기", "2화방 착과수", "second"),
    "strawberry_third": ("strawberry", "딸기", "3화방 착과수", "third"),
    "strawberry_sum123": ("strawberry", "딸기", "1~3화방 합계 착과수", "sum123"),
}


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value


def prediction_json(run_dir: Path):
    result = json.loads((run_dir / "result.json").read_text())
    frame = pd.read_parquet(run_dir / "predictions.parquet")
    for col in ("feature_date", "target_date"):
        frame[col] = pd.to_datetime(frame[col]).dt.strftime("%Y-%m-%d")
    return {"run_id": result["Run_ID"], "predictions": clean(frame.to_dict("records"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--v2-output", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    v1, v2, dashboard = args.v1_output.resolve(), args.v2_output.resolve(), args.dashboard.resolve()
    scheduler = json.loads((v2 / "scheduler.json").read_text())
    if scheduler.get("Status") != "finished": raise SystemExit("v2 campaign is not finished")
    bundle = json.loads((v2 / "handoff/dashboard_bundle.json").read_text())
    base_catalog = json.loads((dashboard / "cache/combination_catalog.json").read_text())
    dest = dashboard / "cache/campaigns" / CAMPAIGN
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "predictions").mkdir(exist_ok=True)

    semantic = pd.read_csv(v2 / "semantic_feature_catalog.csv")
    semantic_features = [
        {"feature_id": r.Variable_ID, "feature_name": r.Variable_ID, "desc": f"{r.Device_Family_KR} · {r.Measurement_Type_KR} · {int(r.Window_Days)}일 · {r.Statistic}", "type": "FLOAT", "source_table": "sfkr_hbfm_env_con", "role": "후보입력", "note": "E6 의미 통합 v2"}
        for r in semantic.itertuples(index=False)
    ]
    base_catalog["variable_groups"]["E6"] = {"code": "E6", "name": "의미 통합 시설 제어 이력", "description": "방향·창 층수를 통합하고 장치와 값 의미를 유지한 E6 v2", "feature_count": len(semantic_features), "features": semantic_features}
    combo_ids = {r["Combination_ID"] for r in bundle["results"]}
    combos = []
    for combo in base_catalog["combinations"]:
        if combo["combination_id"] not in combo_ids: continue
        features = []
        for group in combo["groups"]: features.extend(base_catalog["variable_groups"][group]["features"])
        combo = {**combo, "group_names": [base_catalog["variable_groups"][g]["name"] for g in combo["groups"]], "features": features, "feature_count": len(features)}
        combos.append(combo)
    base_catalog["combinations"] = combos
    base_catalog["total_combinations"] = len(combos)
    (dest / "combination_catalog.json").write_text(json.dumps(clean(base_catalog), ensure_ascii=False, indent=2))

    combo_lookup = {c["combination_id"]: c for c in combos}
    targets = {}
    for target, (crop, crop_kr, target_kr, part) in TARGET_INFO.items():
        groups = {cid: {"group_id": cid, "readable_name": c["label"], "display_name": c["display_name"], "feature_count": c["feature_count"], "category_names": c["group_names"], "category_ids": c["groups"], "features": c["features"]} for cid, c in combo_lookup.items()}
        targets[target] = {"target_id": target, "crop": crop, "crop_kr": crop_kr, "target_kr": target_kr, "part": part, "status": "completed", "frozen": {"overall": None, "winners": {}}, "groups": groups, "results": {}}
    run_sources = {}
    for root in (v1, v2):
        for path in (root / "runs").glob("*/result.json"):
            record = json.loads(path.read_text())
            run_sources[record.get("Run_ID")] = (root, path.parent, record)
    for row in bundle["results"]:
        target, model, cid = row["Target_ID"], row["Model_Type"], row["Combination_ID"]
        run_ids = [x for x in str(row.get("Run_IDs") or "").split("|") if x]
        entry = {"group_id": cid, "model": model, "split": "validation", "completed_seeds": int(row.get("Seed_Count") or 0), "status": row.get("Status"), "source": row.get("Source"), "params": row.get("Params") or {}, "best_epoch": row.get("Best_Epoch"), "bounded": {}, "raw": {}, "seeds": {}}
        for metric in ("rmse", "mae", "r2", "ccc"):
            entry["bounded"][metric + "_mean"] = row.get(metric.upper())
            entry["bounded"][metric + "_std"] = row.get(metric.upper() + "_Std")
        for run_id in run_ids:
            if run_id not in run_sources: continue
            _, run_dir, run = run_sources[run_id]
            entry["seeds"][str(run.get("Seed"))] = {"run_id": run_id, "seed": run.get("Seed"), "status": run.get("Status")}
            if (run_dir / "predictions.parquet").exists():
                (dest / "predictions" / f"{run_id}.json").write_text(json.dumps(prediction_json(run_dir), ensure_ascii=False))
        targets[target]["results"][f"{cid}::{model}::validation"] = clean(entry)
        if row.get("Is_Winner"):
            targets[target]["frozen"]["winners"][model] = {"group": cid}
    for source_root, run_dir, run in run_sources.values():
        if source_root != v2 or run.get("Stage") != "test" or run.get("Status") != "success": continue
        target, model, cid = run["Target_ID"], run["Model_Type"], run["Combination_ID"]
        key = f"{cid}::{model}::test"
        entry = targets[target]["results"].setdefault(key, {"group_id": cid, "model": model, "split": "test", "completed_seeds": 0, "status": "success", "bounded": {}, "raw": {}, "seeds": {}})
        entry["seeds"][str(run["Seed"])] = {"run_id": run["Run_ID"], "seed": run["Seed"], "status": "success"}
        entry["completed_seeds"] = len(entry["seeds"])
        values = entry.setdefault("_metric_values", {name: [] for name in ("rmse", "mae", "r2", "ccc")})
        for metric in values:
            value = run.get("Metrics", {}).get("bounded", {}).get(metric)
            if value is not None: values[metric].append(float(value))
        if (run_dir / "predictions.parquet").exists(): (dest / "predictions" / f"{run['Run_ID']}.json").write_text(json.dumps(prediction_json(run_dir), ensure_ascii=False))
    for target in targets.values():
        for entry in target["results"].values():
            values = entry.pop("_metric_values", None)
            if not values: continue
            for metric, metric_values in values.items():
                entry["bounded"][metric + "_mean"] = float(np.mean(metric_values)) if metric_values else None
                entry["bounded"][metric + "_std"] = float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else (0.0 if metric_values else None)
    index = {"campaign": bundle["campaign"], "targets": targets, "models": base_catalog["models"], "seeds": base_catalog["seeds"]}
    (dest / "experiment_index.json").write_text(json.dumps(clean(index), ensure_ascii=False, indent=2))
    comparison = pd.read_csv(v2 / "comparison_v1_v2.csv").replace({np.nan: None}).to_dict("records")
    (dest / "comparison_v1_v2.json").write_text(json.dumps(clean(comparison), ensure_ascii=False, indent=2))
    campaign_catalog = {"default_campaign": "exhaustive_e1_e7_v1", "campaigns": [{"id": "exhaustive_e1_e7_v1", "label": "기존 E6 · 원 코드 v1", "available": True}, {"id": CAMPAIGN, "label": "의미 통합 E6 v2", "available": True}]}
    (dashboard / "cache/campaign_catalog.json").write_text(json.dumps(campaign_catalog, ensure_ascii=False, indent=2))
    print(f"Published {CAMPAIGN}: {len(combos)} combinations, {sum(len(t['results']) for t in targets.values())} result entries")

if __name__ == "__main__": main()
