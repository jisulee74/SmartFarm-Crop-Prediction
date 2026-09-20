"""Build the static exhaustive-experiment dashboard cache.

The builder supports both a finished campaign and a live campaign.  During a
live campaign it publishes only validation/test groups whose three configured
seeds all completed successfully.  This prevents half-written runs from being
presented as comparable results.
"""

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_DIR = Path(__file__).parent.resolve()
CACHE_DIR = LOCAL_DIR / "cache"
PREDICTIONS_CACHE_DIR = CACHE_DIR / "predictions"
INDEX_FILE = CACHE_DIR / "experiment_index.json"
CATALOG_FILE = CACHE_DIR / "combination_catalog.json"

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


def _resolve_outputs_base():
    configured = os.environ.get("AX_EXHAUSTIVE_OUTPUTS")
    candidates = [
        Path(configured).expanduser() if configured else None,
        LOCAL_DIR.parent / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1",
        LOCAL_DIR.parent / "AXData" / "ax_catalog_experiments" / "outputs" / "exhaustive_e1_e7_v1",
    ]
    for candidate in candidates:
        if candidate and candidate.resolve().exists():
            return candidate.resolve()
    return candidates[1].resolve()


OUTPUTS_BASE = _resolve_outputs_base()


def load_catalog():
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(f"Catalog file not found: {CATALOG_FILE}")
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Sync] Skipping unreadable JSON {path}: {exc}")
        return None


def _metric_stats(runs, variant):
    result = {}
    for metric in ("rmse", "mae", "r2", "ccc"):
        values = []
        for run in runs:
            value = run.get("Metrics", {}).get(variant, {}).get(metric)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def _write_prediction_cache(run):
    run_id = run["Run_ID"]
    source = OUTPUTS_BASE / "runs" / run_id / "predictions.parquet"
    if not source.exists():
        return False
    frame = pd.read_parquet(source)
    predictions = json.loads(frame.to_json(orient="records", date_format="iso"))
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
    }
    (PREDICTIONS_CACHE_DIR / f"{run_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return True


def build_index():
    catalog = load_catalog()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for stale in PREDICTIONS_CACHE_DIR.glob("*.json"):
        stale.unlink()

    targets_dict = {}
    for target_id, info in TARGET_NAMES_KR.items():
        groups = {}
        for combo in catalog["combinations"]:
            cid = combo["combination_id"]
            groups[cid] = {
                "group_id": cid,
                "readable_name": combo["label"],
                "display_name": combo["display_name"],
                "feature_count": combo["feature_count"],
                "category_names": combo["group_names"],
                "category_ids": combo["groups"],
                "features": combo["features"],
            }
        targets_dict[target_id] = {
            "target_id": target_id,
            **info,
            "status": "pending_experiment",
            "frozen": {"overall": None, "winners": {}},
            "groups": groups,
            "results": {},
        }

    summary_by_key = {}
    status_counts = Counter()
    for path in OUTPUTS_BASE.glob("summaries/*/*/*.json"):
        item = _read_json(path)
        if not item:
            continue
        status_counts[str(item.get("Status", "unknown"))] += 1
        key = (str(item.get("Target_ID")), str(item.get("Combination_ID")), str(item.get("Model_Type")))
        summary_by_key[key] = item

    runs_by_key = defaultdict(dict)
    for path in OUTPUTS_BASE.glob("runs/*/result.json"):
        run = _read_json(path)
        if not run or run.get("Status") != "success" or run.get("Stage") not in {"validation", "test"}:
            continue
        try:
            seed = int(run["Seed"])
        except (KeyError, TypeError, ValueError):
            continue
        if seed not in SEEDS:
            continue
        key = (str(run.get("Target_ID")), str(run.get("Combination_ID")), str(run.get("Model_Type")), str(run.get("Stage")))
        runs_by_key[key][seed] = run

    complete_entries = {}
    for (target_id, combo_id, model, stage), seeded in runs_by_key.items():
        if target_id not in targets_dict or model not in MODELS or set(seeded) != set(SEEDS):
            continue
        summary = summary_by_key.get((target_id, combo_id, model), {})
        if stage == "validation" and summary.get("Status") != "success":
            continue
        runs = [seeded[seed] for seed in SEEDS]
        result_key = f"{combo_id}::{model}::{stage}"
        entry = {
            "group_id": combo_id,
            "model": model,
            "split": stage,
            "completed_seeds": len(SEEDS),
            "status": "success",
            "params": summary.get("Params") or runs[0].get("Params") or {},
            "best_epoch": summary.get("Best_Epoch") if summary else runs[0].get("Best_Epoch"),
            "n_eval": int(round(np.mean([r.get("N_Eval", 0) or 0 for r in runs]))),
            "bounded": _metric_stats(runs, "bounded"),
            "raw": _metric_stats(runs, "raw"),
            "seeds": {
                str(seed): {"run_id": seeded[seed]["Run_ID"], "seed": seed, "status": "success"}
                for seed in SEEDS
            },
        }
        if entry["bounded"].get("rmse_mean") is None:
            continue
        targets_dict[target_id]["results"][result_key] = entry
        targets_dict[target_id]["status"] = "running"
        complete_entries[(target_id, combo_id, model, stage)] = (entry, runs)

    # Keep chart data compact: cache the current best validation combination for
    # each target/model plus the catalog's default (first) combination when it is complete.
    chart_keys = set()
    default_combo = catalog["combinations"][0]["combination_id"] if catalog.get("combinations") else None
    for target_id in TARGET_NAMES_KR:
        for model in MODELS:
            choices = [
                (key, value)
                for key, value in complete_entries.items()
                if key[0] == target_id and key[2] == model and key[3] == "validation"
            ]
            if choices:
                best_key, _ = min(
                    choices,
                    key=lambda pair: (
                        pair[1][0]["bounded"]["rmse_mean"],
                        pair[1][0]["bounded"].get("mae_mean", float("inf")),
                        pair[0][1],
                    ),
                )
                chart_keys.add(best_key)
            default_key = (target_id, default_combo, model, "validation")
            if default_key in complete_entries:
                chart_keys.add(default_key)

    cached_prediction_runs = 0
    for key in sorted(chart_keys):
        _, runs = complete_entries[key]
        for run in runs:
            cached_prediction_runs += int(_write_prediction_cache(run))

    expected_summaries = len(TARGET_NAMES_KR) * len(MODELS) * len(catalog["combinations"])
    processed_summaries = sum(status_counts.values())
    campaign_complete = processed_summaries >= expected_summaries
    if campaign_complete:
        for target_id in TARGET_NAMES_KR:
            targets_dict[target_id]["status"] = "completed"
            for model in MODELS:
                candidates = [
                    (key, value[0]) for key, value in complete_entries.items()
                    if key[0] == target_id and key[2] == model and key[3] == "validation"
                ]
                if candidates:
                    best_key, _ = min(candidates, key=lambda pair: (
                        pair[1]["bounded"]["rmse_mean"],
                        pair[1]["bounded"].get("mae_mean", float("inf")),
                        pair[0][1],
                    ))
                    targets_dict[target_id]["frozen"]["winners"][model] = {"group": best_key[1]}

    generated_at = datetime.now(timezone.utc).isoformat()
    index_data = {
        "campaign": {
            "id": "exhaustive_e1_e7_v1",
            "name": "8개 타깃·6개 모델·127개 변수군 조합 전수실험",
            "description": "E1~E7 비공집합 127개 조합, 8개 타깃, 6개 모델, 3개 시드(42, 52, 62) 전수실험",
            "status": "completed" if campaign_complete else "running",
            "has_real_results": bool(complete_entries),
            "partial_results": not campaign_complete,
            "snapshot_generated_at": generated_at,
            "snapshot_note": "전체 실험 진행 중에 생성한 중간 결과입니다. 성공한 3개 시드 검증 결과만 반영했습니다." if not campaign_complete else "전체 실험 완료 결과입니다.",
            "total_combinations": len(catalog["combinations"]),
            "total_targets": len(TARGET_NAMES_KR),
            "total_models": len(MODELS),
            "total_seeds": len(SEEDS),
            "seeds": SEEDS,
            "expected_combination_summaries": expected_summaries,
            "processed_combination_summaries": processed_summaries,
            "progress_percent": round(processed_summaries / expected_summaries * 100, 1),
            "summary_status_counts": dict(sorted(status_counts.items())),
            "published_result_groups": len(complete_entries),
            "cached_prediction_runs": cached_prediction_runs,
        },
        "targets": targets_dict,
    }

    INDEX_FILE.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")
    global _index_cache
    _index_cache = index_data
    print(
        f"[Sync] Saved {INDEX_FILE}: {len(complete_entries)} completed result groups, "
        f"{processed_summaries}/{expected_summaries} summaries, {cached_prediction_runs} prediction files"
    )
    return index_data


def load_index(force=False):
    global _index_cache
    if not force and _index_cache is not None:
        return _index_cache
    if not force and INDEX_FILE.exists():
        _index_cache = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        return _index_cache
    return build_index()


def get_comparison_data(target, model, split):
    index = load_index()
    target_data = index.get("targets", {}).get(target, {})
    results = target_data.get("results", {})
    winner = target_data.get("frozen", {}).get("winners", {}).get(model, {}).get("group") or target_data.get("frozen", {}).get("overall")
    rows = []
    for group_id, info in target_data.get("groups", {}).items():
        result = results.get(f"{group_id}::{model}::{split}")
        row = {
            "group_id": group_id,
            "readable_name": info.get("readable_name", ""),
            "display_name": info.get("display_name", ""),
            "feature_count": info.get("feature_count", 0),
            "category_names": info.get("category_names", []),
            "category_ids": info.get("category_ids", []),
            "completed_seeds": result.get("completed_seeds", 0) if result else 0,
            "status": result.get("status", "unstarted") if result else "unstarted",
            "is_winner": group_id == winner,
            "params": result.get("params", {}) if result else {},
            "bounded": result.get("bounded", {}) if result else {},
            "raw": result.get("raw", {}) if result else {},
        }
        if result:
            row["best_epoch"] = result.get("best_epoch")
        rows.append(row)
    rows.sort(key=lambda row: (
        0 if row.get("bounded", {}).get("rmse_mean") is not None else 1,
        row.get("bounded", {}).get("rmse_mean", float("inf")),
        row.get("bounded", {}).get("mae_mean", float("inf")),
        row["group_id"],
    ))
    rank = 1
    for row in rows:
        if row.get("bounded", {}).get("rmse_mean") is not None:
            row["rank"] = rank
            rank += 1
        else:
            row["rank"] = None
    return {
        "target": target,
        "model": model,
        "split": split,
        "target_status": target_data.get("status", "pending_experiment"),
        "frozen_winner": winner,
        "provisional_winner": rows[0]["group_id"] if rows and rows[0].get("rank") else None,
        "total_candidates": len(rows),
        "rows": rows,
    }


def get_predictions_data(target, group, model, split, seed="all"):
    index = load_index()
    target_data = index.get("targets", {}).get(target, {})
    entry = target_data.get("results", {}).get(f"{group}::{model}::{split}")
    if not entry:
        return {"status": "no_results", "predictions": [], "metadata": {}}
    available = sorted(int(value) for value in entry.get("seeds", {}))
    if str(seed).lower() in {"all", "ensemble"}:
        runs = []
        for value in available:
            run_id = entry["seeds"][str(value)].get("run_id")
            path = PREDICTIONS_CACHE_DIR / f"{run_id}.json"
            if path.exists():
                predictions = _read_json(path).get("predictions", [])
                if predictions:
                    runs.append(predictions)
        if not runs:
            return {"status": "no_predictions", "predictions": [], "metadata": entry}
        output = []
        for index_row, first in enumerate(runs[0]):
            raw = [rows[index_row].get("prediction_raw") for rows in runs if index_row < len(rows) and rows[index_row].get("prediction_raw") is not None]
            bounded = [rows[index_row].get("prediction_bounded") for rows in runs if index_row < len(rows) and rows[index_row].get("prediction_bounded") is not None]
            output.append({
                **first,
                "crop_sn": str(first.get("crop_sn")),
                "sample_num": str(first.get("sample_num")),
                "prediction_raw": float(np.mean(raw)) if raw else None,
                "prediction_bounded": float(np.mean(bounded)) if bounded else None,
                "seeds_averaged": len(raw),
            })
        return {"status": "success", "mode": "ensemble", "available_seeds": available, "metadata": entry, "predictions": output}
    selected = int(seed)
    seed_data = entry.get("seeds", {}).get(str(selected), {})
    path = PREDICTIONS_CACHE_DIR / f"{seed_data.get('run_id')}.json"
    if not path.exists():
        return {"status": "prediction_file_missing", "available_seeds": available, "predictions": [], "metadata": entry}
    predictions = _read_json(path).get("predictions", [])
    return {"status": "success", "mode": "single_seed", "seed": selected, "available_seeds": available, "metadata": entry, "predictions": predictions}


if __name__ == "__main__":
    print(f"Building exhaustive experiment index from {OUTPUTS_BASE}")
    built = build_index()
    print(f"Index built with {len(built['targets'])} targets")
