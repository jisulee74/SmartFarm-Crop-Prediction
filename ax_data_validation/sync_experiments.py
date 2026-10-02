"""
sync_experiments.py - High Performance Experiment Index Builder & Sync
Compiles stability_v2 experiment outputs into a unified experiment_index.json.
Supports on-demand prediction retrieval, ensemble averaging, and local caching.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np

REMOTE_HOST = "NHN-GPU"
REMOTE_BASE = "/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/AXData"
REMOTE_STABILITY = f"{REMOTE_BASE}/ax_catalog_experiments/outputs/stability_v2"
REMOTE_CSV = f"{REMOTE_BASE}/ax_catalog_experiments/outputs/csv"

LOCAL_DIR = Path(__file__).parent.resolve()
CACHE_DIR = LOCAL_DIR / "cache"
PREDICTIONS_CACHE_DIR = CACHE_DIR / "predictions"
INDEX_FILE = CACHE_DIR / "experiment_index.json"

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
SEEDS = [42, 52, 62, 72, 82]

_index_cache = None

def run_remote_python(code: str) -> str:
    """Executes Python code on the remote server via SSH."""
    proc = subprocess.run(
        ["ssh", REMOTE_HOST, "python3"],
        input=code,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Remote command failed: {proc.stderr}")
    return proc.stdout

def build_index_script():
    return r'''
import json, os, time
from pathlib import Path
import pandas as pd
import numpy as np

base = Path("/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/AXData/ax_catalog_experiments/outputs/stability_v2")
csv_base = Path("/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/AXData/ax_catalog_experiments/outputs/csv")

# Load Catalogs
gm = pd.read_csv(base / "Group_Mapping.csv")
gd = pd.read_csv(base / "Group_Detail.csv")
cat_cat = pd.read_csv(csv_base / "Category_Catalog.csv")
cat_name_map = dict(zip(cat_cat['Category_ID'], cat_cat['Variable_Category']))
cat_domain_map = dict(zip(cat_cat['Category_ID'], cat_cat['Variable_Domain']))

# Load Runs and Results
runs_csv = base / "csv/Runs.csv"
runs_df = pd.read_csv(runs_csv)
val_test_runs = runs_df[runs_df.Stage.isin(['validation', 'test']) & (runs_df.Status == 'success')].copy()

# Load Results for individual seed metrics
res_csv = base / "csv/Results.csv"
res_df = pd.read_csv(res_csv)
val_test_res = res_df[res_df.Eval_Split.isin(['validation', 'test']) & (res_df.Status == 'success')].copy()

# Load Results_Summary for mean and std
sum_csv = base / "csv/Results_Summary.csv"
sum_df = pd.read_csv(sum_csv)

# Load Scheduler
scheduler_file = base / "parallel/scheduler.json"
scheduler = json.load(open(scheduler_file)) if scheduler_file.exists() else {}

targets = [
    "tomato_first", "tomato_second", "tomato_third", "tomato_sum123",
    "strawberry_first", "strawberry_second", "strawberry_third", "strawberry_sum123"
]

target_names_kr = {
    "tomato_first": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "1화방 꽃수", "part": "first"},
    "tomato_second": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "2화방 꽃수", "part": "second"},
    "tomato_third": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "3화방 꽃수", "part": "third"},
    "tomato_sum123": {"crop": "tomato", "crop_kr": "토마토", "target_kr": "1~3화방 합계 꽃수", "part": "sum123"},
    "strawberry_first": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "1화방 착과수", "part": "first"},
    "strawberry_second": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "2화방 착과수", "part": "second"},
    "strawberry_third": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "3화방 착과수", "part": "third"},
    "strawberry_sum123": {"crop": "strawberry", "crop_kr": "딸기", "target_kr": "1~3화방 합계 착과수", "part": "sum123"},
}

out_index = {
    "campaign": scheduler,
    "targets": {}
}

# Pre-index Results by (Run_ID, Prediction_Variant, Metric_Name) -> Metric_Value
res_metric_map = {}
for _, r in val_test_res.iterrows():
    k = (r["Run_ID"], r["Prediction_Variant"], r["Metric_Name"].upper())
    res_metric_map[k] = r["Metric_Value"]

# Pre-index Results_Summary by (Target_ID, Variable_Group_ID, Model_Type, Eval_Split, Prediction_Variant, Metric_Name) -> (Mean, Std, Seed_Count)
sum_map = {}
for _, r in sum_df.iterrows():
    k = (r["Target_ID"], r["Variable_Group_ID"], r["Model_Type"], r["Eval_Split"], r["Prediction_Variant"], r["Metric_Name"].upper())
    sum_map[k] = (float(r["Mean"]) if pd.notnull(r["Mean"]) else None,
                  float(r["Std"]) if pd.notnull(r["Std"]) else 0.0,
                  int(r["Seed_Count"]) if pd.notnull(r["Seed_Count"]) else 0)

for t in targets:
    t_info = target_names_kr[t]
    t_dir = base / "experiments" / t
    status_file = t_dir / "status.json"
    frozen_file = t_dir / "selection_frozen.json"
    manifest_file = base / "prepared" / t / "manifest.json"
    
    st_info = {"status": "unstarted"}
    if status_file.exists():
        try: st_info = json.load(open(status_file))
        except: pass
    elif (t_dir / "runs").exists():
        st_info = {"status": "in_progress"}
        
    frozen_info = {}
    if frozen_file.exists():
        try: frozen_info = json.load(open(frozen_file))
        except: pass
        
    groups_meta = {}
    if manifest_file.exists():
        m_data = json.load(open(manifest_file))
        raw_groups = m_data.get("groups", {})
        for gid, gdata in raw_groups.items():
            old_gid_vals = gm[gm.Variable_Group_ID == gid]["Old_Variable_Group_ID"].values
            old_str = str(old_gid_vals[0]) if len(old_gid_vals) else ""
            
            cats_in_gd = gd[gd.Variable_Group_ID == gid]["Old_Category_ID"].dropna().unique().tolist()
            cat_names = []
            seen_names = set()
            for c in cats_in_gd:
                name = cat_name_map.get(c, c)
                if name not in seen_names:
                    seen_names.add(name)
                    cat_names.append(name)
                    
            if not cat_names:
                readable = old_str if old_str else gid
            else:
                readable = " + ".join(cat_names)
                
            features = gdata.get("features", [])
            groups_meta[gid] = {
                "group_id": gid,
                "old_group_id": old_str,
                "readable_name": readable,
                "category_names": cat_names,
                "category_ids": cats_in_gd,
                "feature_count": len(features),
                "features": features
            }

    t_runs = val_test_runs[val_test_runs.Target_ID == t]
    target_results = {}
    
    if not t_runs.empty:
        for (gid, model, split), g_runs in t_runs.groupby(["Variable_Group_ID", "Model_Type", "Stage"]):
            key = f"{gid}::{model}::{split}"
            is_win = (frozen_info.get("winners", {}).get(model, {}).get("group") == gid) if frozen_info else False
            
            seeds_dict = {}
            for _, run_row in g_runs.iterrows():
                rid = run_row["Run_ID"]
                s = int(run_row["Seed"])
                
                b_metrics = {
                    "rmse": res_metric_map.get((rid, "bounded", "RMSE")),
                    "mae": res_metric_map.get((rid, "bounded", "MAE")),
                    "r2": res_metric_map.get((rid, "bounded", "R2")),
                    "ccc": res_metric_map.get((rid, "bounded", "CCC")),
                }
                r_metrics = {
                    "rmse": res_metric_map.get((rid, "raw", "RMSE")),
                    "mae": res_metric_map.get((rid, "raw", "MAE")),
                    "r2": res_metric_map.get((rid, "raw", "R2")),
                    "ccc": res_metric_map.get((rid, "raw", "CCC")),
                }
                
                params = {}
                if pd.notnull(run_row["Params"]):
                    try: params = json.loads(run_row["Params"])
                    except: pass
                    
                seeds_dict[s] = {
                    "run_id": rid,
                    "seed": s,
                    "n_eval": int(run_row["N_Eval"]) if pd.notnull(run_row["N_Eval"]) else None,
                    "bounded": b_metrics,
                    "raw": r_metrics,
                    "params": params,
                    "upper_bound": float(run_row["Upper_Bound"]) if pd.notnull(run_row["Upper_Bound"]) else None,
                    "output_clip_rate": float(run_row["Output_Clip_Rate"]) if pd.notnull(run_row["Output_Clip_Rate"]) else None,
                    "best_epoch": int(run_row["Best_Epoch"]) if pd.notnull(run_row["Best_Epoch"]) else None,
                    "duration_seconds": float(run_row["Duration_Seconds"]) if pd.notnull(run_row["Duration_Seconds"]) else None,
                }
                
            def get_stat(var, metric):
                return sum_map.get((t, gid, model, split, var, metric), (None, None, 0))
                
            b_rmse_m, b_rmse_s, s_cnt = get_stat("bounded", "RMSE")
            b_mae_m, b_mae_s, _ = get_stat("bounded", "MAE")
            b_r2_m, b_r2_s, _ = get_stat("bounded", "R2")
            b_ccc_m, b_ccc_s, _ = get_stat("bounded", "CCC")

            r_rmse_m, r_rmse_s, _ = get_stat("raw", "RMSE")
            r_mae_m, r_mae_s, _ = get_stat("raw", "MAE")
            r_r2_m, r_r2_s, _ = get_stat("raw", "R2")
            r_ccc_m, r_ccc_s, _ = get_stat("raw", "CCC")
            
            first_run = next(iter(seeds_dict.values())) if seeds_dict else {}
            status_str = "complete" if len(seeds_dict) >= 5 else ("partial" if len(seeds_dict) > 0 else "unstarted")

            target_results[key] = {
                "group_id": gid,
                "model": model,
                "split": split,
                "completed_seeds": len(seeds_dict),
                "status": status_str,
                "is_winner": is_win,
                "params": first_run.get("params", {}),
                "upper_bound": first_run.get("upper_bound"),
                "output_clip_rate": first_run.get("output_clip_rate"),
                "best_epoch": first_run.get("best_epoch"),
                "n_eval": first_run.get("n_eval"),
                "bounded": {
                    "rmse_mean": b_rmse_m, "rmse_std": b_rmse_s,
                    "mae_mean": b_mae_m, "mae_std": b_mae_s,
                    "r2_mean": b_r2_m, "r2_std": b_r2_s,
                    "ccc_mean": b_ccc_m, "ccc_std": b_ccc_s,
                },
                "raw": {
                    "rmse_mean": r_rmse_m, "rmse_std": r_rmse_s,
                    "mae_mean": r_mae_m, "mae_std": r_mae_s,
                    "r2_mean": r_r2_m, "r2_std": r_r2_s,
                    "ccc_mean": r_ccc_m, "ccc_std": r_ccc_s,
                },
                "seeds": seeds_dict
            }

    out_index["targets"][t] = {
        "target_id": t,
        "crop": t_info["crop"],
        "crop_kr": t_info["crop_kr"],
        "target_kr": t_info["target_kr"],
        "part": t_info["part"],
        "status": st_info.get("status", "unstarted"),
        "status_details": st_info,
        "frozen": frozen_info,
        "groups": groups_meta,
        "results": target_results
    }

print(json.dumps(out_index, ensure_ascii=False))
'''

def sync_index():
    """Generates index on remote server and saves locally to cache/experiment_index.json."""
    global _index_cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    script = build_index_script()
    print("[sync_experiments] Executing remote index builder...")
    out_str = run_remote_python(script)
    
    data = json.loads(out_str)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[sync_experiments] Index written successfully to {INDEX_FILE} ({os.path.getsize(INDEX_FILE)/1024:.1f} KB)")
    _index_cache = data
    return data

def load_index(force_refresh: bool = False):
    """Loads cached index or syncs if missing or force_refresh is True."""
    global _index_cache
    if not force_refresh and _index_cache is not None:
        return _index_cache
    if not force_refresh and INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                _index_cache = json.load(f)
                return _index_cache
        except Exception:
            pass
    return sync_index()

def fetch_prediction(target: str, run_id: str) -> dict:
    """Fetches a specific run's predictions and metadata."""
    PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PREDICTIONS_CACHE_DIR / f"{run_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    fetch_script = f'''
import json, os
import pandas as pd

base = "/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/AXData/ax_catalog_experiments/outputs/stability_v2/experiments/{target}/runs/{run_id}"
res_file = f"{{base}}/result.json"
pred_file = f"{{base}}/predictions.parquet"

meta = {{}}
if os.path.exists(res_file):
    meta = json.load(open(res_file))

preds = []
if os.path.exists(pred_file):
    df = pd.read_parquet(pred_file)
    if 'target_date' in df.columns:
        df['target_date'] = pd.to_datetime(df['target_date']).dt.strftime('%Y-%m-%d')
    if 'feature_date' in df.columns:
        df['feature_date'] = pd.to_datetime(df['feature_date']).dt.strftime('%Y-%m-%d')
    df = df.where(pd.notnull(df), None)
    preds = df.to_dict(orient="records")

out = {{
    "run_id": "{run_id}",
    "target": "{target}",
    "metadata": meta,
    "predictions": preds
}}
print(json.dumps(out, ensure_ascii=False))
'''
    out_str = run_remote_python(fetch_script)
    data = json.loads(out_str)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data

def get_prediction_data(target: str, group_id: str, model: str, split: str, seed: str = "all") -> dict:
    """
    Returns prediction time-series and metadata for a given target, group, model, split, and seed.
    If seed == 'all', computes ensemble average across available seeds.
    """
    index = load_index()
    t_data = index.get("targets", {}).get(target)
    if not t_data:
        return {"status": "target_not_found", "predictions": [], "metadata": {}}

    key = f"{group_id}::{model}::{split}"
    res_entry = t_data.get("results", {}).get(key)
    if not res_entry or not res_entry.get("seeds"):
        return {
            "status": "no_results",
            "group_id": group_id,
            "model": model,
            "split": split,
            "target": target,
            "predictions": [],
            "metadata": {}
        }

    seeds_dict = res_entry["seeds"]
    available_seeds = [int(s) for s in seeds_dict.keys()]

    if str(seed).lower() in ["all", "mean", "ensemble"]:
        # Fetch all available seeds
        all_pred_runs = []
        for s in sorted(available_seeds):
            run_id = seeds_dict[str(s)]["run_id"]
            p_data = fetch_prediction(target, run_id)
            if p_data.get("predictions"):
                all_pred_runs.append(p_data["predictions"])

        if not all_pred_runs:
            return {"status": "no_predictions", "predictions": [], "metadata": res_entry}

        # Align rows and average predictions
        n_rows = len(all_pred_runs[0])
        ensemble_preds = []
        for i in range(n_rows):
            row0 = all_pred_runs[0][i]
            raw_vals = [all_pred_runs[j][i]["prediction_raw"] for j in range(len(all_pred_runs)) if all_pred_runs[j][i].get("prediction_raw") is not None]
            bnd_vals = [all_pred_runs[j][i]["prediction_bounded"] for j in range(len(all_pred_runs)) if all_pred_runs[j][i].get("prediction_bounded") is not None]
            
            ensemble_preds.append({
                "facility_id": row0["facility_id"],
                "crop_sn": str(row0["crop_sn"]),
                "sample_num": str(row0["sample_num"]),
                "row_id": row0.get("row_id"),
                "feature_date": row0.get("feature_date"),
                "target_date": row0.get("target_date"),
                "target": row0.get("target"),
                "prediction_raw": float(np.mean(raw_vals)) if raw_vals else None,
                "prediction_bounded": float(np.mean(bnd_vals)) if bnd_vals else None,
                "upper_bound": row0.get("upper_bound"),
                "seeds_averaged": len(raw_vals)
            })

        return {
            "status": "success",
            "mode": "ensemble",
            "seeds_count": len(all_pred_runs),
            "available_seeds": available_seeds,
            "target": target,
            "group_id": group_id,
            "model": model,
            "split": split,
            "metadata": res_entry,
            "predictions": ensemble_preds
        }
    else:
        s_int = int(seed)
        seed_item = seeds_dict.get(str(s_int))
        if not seed_item:
            return {
                "status": "seed_not_found",
                "available_seeds": available_seeds,
                "predictions": [],
                "metadata": res_entry
            }
        run_id = seed_item["run_id"]
        p_data = fetch_prediction(target, run_id)
        return {
            "status": "success",
            "mode": "single_seed",
            "seed": s_int,
            "run_id": run_id,
            "available_seeds": available_seeds,
            "target": target,
            "group_id": group_id,
            "model": model,
            "split": split,
            "metadata": {
                **res_entry,
                "single_seed_metrics": seed_item
            },
            "predictions": p_data.get("predictions", [])
        }

if __name__ == "__main__":
    print("Testing get_prediction_data...")
    res = get_prediction_data("tomato_first", "VG_Flower_first_0af25803_V2", "catboost", "validation", "all")
    print("Status:", res["status"])
    print("Predictions count:", len(res.get("predictions", [])))
    print("Metadata bounded RMSE mean:", res.get("metadata", {}).get("bounded", {}).get("rmse_mean"))
