#!/usr/bin/env python3
"""Flower-only +1 increment-event suite; fruit-set remains direct regression."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "increment_event_backend.py"

def implementation():
    spec = importlib.util.spec_from_file_location("flower_increment_impl", SOURCE)
    if spec is None or spec.loader is None: raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RF_ROOT = ROOT
    return module

def run(run_id: str, features: list[str] | None = None):
    m = implementation()
    if features is not None:
        m.FEATURES = list(features)
    out = ROOT / "artifacts" / "increment_event_model_suite_v1_3" / run_id
    out.mkdir(parents=True, exist_ok=False)
    train, val = m.load("train"), m.load("val")
    rows, settings, searches = [], {}, []
    for kind in m.MODELS:
        threshold, cv_mae, epochs, search = m.cv_model(train, kind)
        searches.append(search); settings[kind] = {"threshold": threshold, "cv_mae": cv_mae, "fixed_epochs": epochs}
        seeds = m.FINAL_SEEDS if kind in m.DL_MODELS else (42,)
        _, probability = m.fit_predict(train, val, kind, epochs, seeds)
        event = probability >= threshold
        pred = val.fruit_cluster_num.to_numpy(float) + event.astype(float)
        truth = val.increase_target.to_numpy(bool)
        tn, fp, fn, tp = confusion_matrix(truth, event, labels=[False, True]).ravel()
        rows.append({"model": kind, "is_learning_model": True, "threshold": threshold,
                     "cv_mae": cv_mae, "seed_count": len(seeds),
                     **m.metrics(val.next_fruit_cluster_num, pred),
                     "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    persistence = val.fruit_cluster_num.to_numpy(float)
    truth = val.increase_target.to_numpy(bool)
    tn, fp, fn, tp = confusion_matrix(truth, np.zeros(len(truth), bool), labels=[False, True]).ravel()
    rows.append({"model": "persistence", "is_learning_model": False, "threshold": np.nan,
                 "cv_mae": np.nan, "seed_count": 0,
                 **m.metrics(val.next_fruit_cluster_num, persistence),
                 "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    table = pd.DataFrame(rows); table.to_csv(out / "validation_model_comparison.csv", index=False)
    pd.concat(searches).to_csv(out / "internal_cv_threshold_search.csv", index=False)
    learning = table[table.is_learning_model]; best = learning.rmse.min()
    selected = str(learning[learning.rmse <= best * 1.01].sort_values(["mae", "model"], kind="stable").iloc[0].model)
    frozen = {"prediction_contract": "current_count + binary_increment_event",
              "allowed_increment": [0, 1], "features": m.FEATURES,
              "feature_count": len(m.FEATURES), "selected_learning_model": selected,
              "settings": settings[selected], "selection_uses_test": False,
              "fruit_set_pipeline_changed": False}
    (out / "selected_configuration.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    combined = pd.concat([train, val], ignore_index=True)
    seeds = m.FINAL_SEEDS if selected in m.DL_MODELS else (42,)
    models, _ = m.fit_predict(combined, combined.iloc[:1], selected, settings[selected]["fixed_epochs"], seeds)
    model_dir = out / "models"; model_dir.mkdir()
    for model, seed in zip(models, seeds): model.save(model_dir / f"{selected}_seed_{seed}.bin")
    test = m.load("test")
    probability = np.mean([model.predict_proba(test) for model in models], axis=0)
    event = probability >= settings[selected]["threshold"]
    prediction = test.fruit_cluster_num.to_numpy(float) + event.astype(float)
    baseline = test.fruit_cluster_num.to_numpy(float)
    test_table = pd.DataFrame([{"model": selected, **m.metrics(test.next_fruit_cluster_num, prediction)},
                               {"model": "persistence", **m.metrics(test.next_fruit_cluster_num, baseline)}])
    test_table.to_csv(out / "test_model_comparison.csv", index=False)
    pred = test[["sn", "user_id", "crop_cycle_id", "sample_num", "target_date", "fruit_cluster_num", "next_fruit_cluster_num"]].copy()
    pred["increase_probability"] = probability; pred["predicted_increase"] = event.astype(int)
    pred["prediction"] = prediction; pred["persistence_prediction"] = baseline
    pred.to_parquet(out / "test_predictions.parquet", index=False)
    manifest = {"status": "complete", "stage": "test_once", "selection_uses_test": False,
                "test_is_historical_reused_holdout": True, "selected": frozen,
                "test": test_table.set_index("model").to_dict(orient="index")}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2)); return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True)
    run(parser.parse_args().run_id)
