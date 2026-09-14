"""
sync_exhaustive_experiments.py - Exhaustive Experiment Index Builder & Sync
Compiles 127 Variable Group Combinations x 8 Targets x 6 Models x 3 Seeds (42, 52, 62)
outputs into a unified experiment_index.json for the web dashboard.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

LOCAL_DIR = Path(__file__).parent.resolve()
CACHE_DIR = LOCAL_DIR / "cache"
PREDICTIONS_CACHE_DIR = CACHE_DIR / "predictions"
INDEX_FILE = CACHE_DIR / "experiment_index.json"
CATALOG_FILE = CACHE_DIR / "combination_catalog.json"

# Base path to exhaustive outputs
OUTPUTS_BASE = (LOCAL_DIR.parent / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1").resolve()
HANDOFF_BASE = OUTPUTS_BASE / "handoff"
CSV_BASE = OUTPUTS_BASE / "csv"

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

MODELS = ["poisson", "random_forest", "catboost", "mlp", "tabm", "tft"]
SEEDS = [42, 52, 62]

_index_cache = None


def load_catalog():
    if CATALOG_FILE.exists():
        with open(CATALOG_FILE, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Catalog file not found: {CATALOG_FILE}")


def build_index():
    catalog = load_catalog()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    targets_dict = {}
    for t_id, t_info in TARGET_NAMES_KR.items():
        groups_dict = {}
        for c in catalog["combinations"]:
            cid = c["combination_id"]
            groups_dict[cid] = {
                "group_id": cid,
                "readable_name": c["label"],
                "display_name": c["display_name"],
                "feature_count": c["feature_count"],
                "category_names": c["group_names"],
                "category_ids": c["groups"],
                "features": c["features"],
            }
        targets_dict[t_id] = {
            "target_id": t_id,
            "crop": t_info["crop"],
            "crop_kr": t_info["crop_kr"],
            "target_kr": t_info["target_kr"],
            "part": t_info["part"],
            "status": "pending_experiment",
            "frozen": {"overall": None, "winners": {}},
            "groups": groups_dict,
            "results": {},
        }

    # Check if actual outputs exist in OUTPUTS_BASE
    runs_file = CSV_BASE / "Runs.csv"
    metrics_file = CSV_BASE / "Metrics.csv"
    summary_file = CSV_BASE / "Combination_Summary.csv"
    best_file = CSV_BASE / "Best_Configurations.csv"
    handoff_bundle_file = HANDOFF_BASE / "dashboard_bundle.json"

    has_real_results = False
    campaign_status = "pending"

    if handoff_bundle_file.exists():
        try:
            with open(handoff_bundle_file, encoding="utf-8") as f:
                bundle = json.load(f)
            has_real_results = True
            campaign_status = "completed"
            print(f"[Sync] Loaded handoff bundle from {handoff_bundle_file}")
        except Exception as e:
            print(f"[Sync] Warning: Failed to read handoff bundle: {e}")

    elif runs_file.exists() and metrics_file.exists():
        try:
            runs_df = pd.read_csv(runs_file)
            metrics_df = pd.read_csv(metrics_file)
            summary_df = pd.read_csv(summary_file) if summary_file.exists() else None
            best_df = pd.read_csv(best_file) if best_file.exists() else None

            has_real_results = True
            campaign_status = "running" if len(runs_df[runs_df.Status == "running"]) > 0 else "completed"

            # Parse results by (target, group, model, split)
            # Group metrics by Run_ID and aggregate
            for _, r in runs_df.iterrows():
                run_id = str(r["Run_ID"])
                target_id = str(r["Target_ID"])
                combo_id = str(r["Combination_ID"])
                model_type = str(r["Model_Type"])
                stage = str(r["Stage"])  # validation or test
                seed = int(r["Seed"]) if pd.notna(r.get("Seed")) else None

                if target_id not in targets_dict:
                    continue

                res_key = f"{combo_id}::{model_type}::{stage}"
                if res_key not in targets_dict[target_id]["results"]:
                    targets_dict[target_id]["results"][res_key] = {
                        "group_id": combo_id,
                        "model": model_type,
                        "split": stage,
                        "completed_seeds": 0,
                        "status": "success",
                        "params": json.loads(r["Hyperparameters"]) if "Hyperparameters" in r and pd.notna(r["Hyperparameters"]) else {},
                        "best_epoch": int(r["Best_Epoch"]) if "Best_Epoch" in r and pd.notna(r["Best_Epoch"]) else None,
                        "bounded": {},
                        "raw": {},
                        "seeds": {},
                    }

                entry = targets_dict[target_id]["results"][res_key]
                if seed is not None:
                    entry["seeds"][str(seed)] = {
                        "run_id": run_id,
                        "seed": seed,
                        "status": str(r.get("Status", "success")),
                    }
                    entry["completed_seeds"] = len(entry["seeds"])

            # Compute summary means from metrics_df
            for res_key_full, res_entry in targets_dict[target_id]["results"].items():
                cid, mtype, stg = res_key_full.split("::")
                sub_m = metrics_df[
                    (metrics_df.Target_ID == target_id) &
                    (metrics_df.Combination_ID == cid) &
                    (metrics_df.Model_Type == mtype) &
                    (metrics_df.Stage == stg)
                ]
                for variant in ["bounded", "raw"]:
                    var_sub = sub_m[sub_m.Prediction_Variant == variant]
                    if len(var_sub) > 0:
                        for mname in ["RMSE", "MAE", "R2", "CCC"]:
                            vals = var_sub[var_sub.Metric_Name == mname]["Metric_Value"].dropna()
                            if len(vals) > 0:
                                res_entry[variant][f"{mname.lower()}_mean"] = float(vals.mean())
                                res_entry[variant][f"{mname.lower()}_std"] = float(vals.std()) if len(vals) > 1 else 0.0

            # Fill winners if best_df exists
            if best_df is not None:
                for _, b in best_df.iterrows():
                    tid = str(b["Target_ID"])
                    mtype = str(b["Model_Type"])
                    win_grp = str(b["Combination_ID"])
                    if tid in targets_dict:
                        targets_dict[tid]["frozen"]["winners"][mtype] = {"group": win_grp}

            print(f"[Sync] Processed real results from {runs_file}")
        except Exception as e:
            print(f"[Sync] Warning: Failed to parse CSV results: {e}")

    index_data = {
        "campaign": {
            "id": "exhaustive_e1_e7_v1",
            "name": "8개 타깃·6개 모델·127개 변수군 조합 전수실험",
            "description": "E1~E7 비공집합 127개 조합, 8개 타깃, 6개 모델, 3개 시드(42, 52, 62) 전수실험",
            "status": campaign_status,
            "has_real_results": has_real_results,
            "total_combinations": len(catalog["combinations"]),
            "total_targets": len(TARGET_NAMES_KR),
            "total_models": len(MODELS),
            "total_seeds": len(SEEDS),
            "seeds": SEEDS,
        },
        "targets": targets_dict,
    }

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    global _index_cache
    _index_cache = index_data
    print(f"[Sync] Saved index to {INDEX_FILE} (has_real_results={has_real_results})")
    return index_data


def load_index(force=False):
    global _index_cache
    if not force and _index_cache is not None:
        return _index_cache
    if not force and INDEX_FILE.exists():
        with open(INDEX_FILE, encoding="utf-8") as f:
            _index_cache = json.load(f)
            return _index_cache
    return build_index()


def get_comparison_data(target, model, split):
    index = load_index()
    t_data = index.get("targets", {}).get(target, {})
    groups_dict = t_data.get("groups", {})
    results_dict = t_data.get("results", {})
    frozen_winners = t_data.get("frozen", {}).get("winners", {})
    winner_group = frozen_winners.get(model, {}).get("group") or t_data.get("frozen", {}).get("overall")

    rows = []
    for gid, ginfo in groups_dict.items():
        key = f"{gid}::{model}::{split}"
        res_item = results_dict.get(key)
        if not res_item:
            rows.append({
                "group_id": gid,
                "readable_name": ginfo.get("readable_name", ""),
                "display_name": ginfo.get("display_name", ""),
                "feature_count": ginfo.get("feature_count", 0),
                "category_names": ginfo.get("category_names", []),
                "category_ids": ginfo.get("category_ids", []),
                "completed_seeds": 0,
                "status": "unstarted",
                "is_winner": (gid == winner_group),
                "params": {},
                "bounded": {},
                "raw": {},
            })
        else:
            rows.append({
                "group_id": gid,
                "readable_name": ginfo.get("readable_name", ""),
                "display_name": ginfo.get("display_name", ""),
                "feature_count": ginfo.get("feature_count", 0),
                "category_names": ginfo.get("category_names", []),
                "category_ids": ginfo.get("category_ids", []),
                "completed_seeds": res_item.get("completed_seeds", 0),
                "status": res_item.get("status", "unstarted"),
                "is_winner": (gid == winner_group),
                "params": res_item.get("params", {}),
                "best_epoch": res_item.get("best_epoch"),
                "bounded": res_item.get("bounded", {}),
                "raw": res_item.get("raw", {}),
            })

    # Sort: completed results with valid RMSE first (ascending), unstarted last
    rows.sort(key=lambda r: (
        0 if r.get("bounded", {}).get("rmse_mean") is not None else 1,
        r.get("bounded", {}).get("rmse_mean") if r.get("bounded", {}).get("rmse_mean") is not None else 999999,
        r.get("feature_count", 999)
    ))

    # Assign rank only to completed rows
    rank_idx = 1
    for r in rows:
        if r.get("bounded", {}).get("rmse_mean") is not None:
            r["rank"] = rank_idx
            rank_idx += 1
        else:
            r["rank"] = None

    provisional_winner = rows[0]["group_id"] if rows and rows[0].get("bounded", {}).get("rmse_mean") is not None else None

    return {
        "target": target,
        "model": model,
        "split": split,
        "target_status": t_data.get("status", "pending_experiment"),
        "frozen_winner": winner_group,
        "provisional_winner": provisional_winner,
        "total_candidates": len(rows),
        "rows": rows,
    }


def get_predictions_data(target, group, model, split, seed="all"):
    index = load_index()
    t_data = index.get("targets", {}).get(target, {})
    key = f"{group}::{model}::{split}"
    res_entry = t_data.get("results", {}).get(key)
    if not res_entry:
        return {
            "status": "no_results",
            "group_id": group,
            "model": model,
            "split": split,
            "target": target,
            "predictions": [],
            "metadata": {},
        }

    seeds_dict = res_entry.get("seeds", {})
    available_seeds = sorted([int(s) for s in seeds_dict.keys()])

    if str(seed).lower() in ("all", "ensemble"):
        all_pred_runs = []
        for s in available_seeds:
            run_id = seeds_dict.get(str(s), {}).get("run_id")
            if not run_id:
                continue
            pred_file = PREDICTIONS_CACHE_DIR / f"{run_id}.json"
            if pred_file.exists():
                with open(pred_file, encoding="utf-8") as pf:
                    pj = json.load(pf)
                    if pj.get("predictions"):
                        all_pred_runs.append(pj["predictions"])

        if not all_pred_runs:
            return {"status": "no_predictions", "predictions": [], "metadata": res_entry}

        n_rows = len(all_pred_runs[0])
        ensemble_preds = []
        for i in range(n_rows):
            row0 = all_pred_runs[0][i]
            raw_vals = [run[i]["prediction_raw"] for run in all_pred_runs if i < len(run) and run[i].get("prediction_raw") is not None]
            bnd_vals = [run[i]["prediction_bounded"] for run in all_pred_runs if i < len(run) and run[i].get("prediction_bounded") is not None]
            ensemble_preds.append({
                "facility_id": row0.get("facility_id"),
                "crop_sn": str(row0.get("crop_sn")),
                "sample_num": str(row0.get("sample_num")),
                "row_id": row0.get("row_id"),
                "feature_date": row0.get("feature_date"),
                "target_date": row0.get("target_date"),
                "target": row0.get("target"),
                "prediction_raw": float(np.mean(raw_vals)) if raw_vals else None,
                "prediction_bounded": float(np.mean(bnd_vals)) if bnd_vals else None,
                "upper_bound": row0.get("upper_bound"),
                "seeds_averaged": len(raw_vals),
            })
        return {
            "status": "success",
            "mode": "ensemble",
            "seeds_count": len(all_pred_runs),
            "available_seeds": available_seeds,
            "target": target,
            "group_id": group,
            "model": model,
            "split": split,
            "metadata": res_entry,
            "predictions": ensemble_preds,
        }
    else:
        s_int = int(seed)
        seed_item = seeds_dict.get(str(s_int))
        if not seed_item or not seed_item.get("run_id"):
            return {"status": "seed_not_found", "available_seeds": available_seeds, "predictions": [], "metadata": res_entry}
        run_id = seed_item["run_id"]
        pred_file = PREDICTIONS_CACHE_DIR / f"{run_id}.json"
        if not pred_file.exists():
            return {"status": "prediction_file_missing", "available_seeds": available_seeds, "predictions": [], "metadata": res_entry}
        with open(pred_file, encoding="utf-8") as pf:
            pj = json.load(pf)
        preds = [{**r, "crop_sn": str(r.get("crop_sn")), "sample_num": str(r.get("sample_num"))} for r in pj.get("predictions", [])]
        return {
            "status": "success",
            "mode": "single_seed",
            "seed": s_int,
            "run_id": run_id,
            "available_seeds": available_seeds,
            "target": target,
            "group_id": group,
            "model": model,
            "split": split,
            "metadata": res_entry,
            "predictions": preds,
        }


if __name__ == "__main__":
    print("Building Exhaustive Experiment Index...")
    index = build_index()
    print("Index successfully built with", len(index["targets"]), "targets.")
