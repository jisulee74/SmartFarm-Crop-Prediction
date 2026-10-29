from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...common_regression.temporal_cv import build_expanding_folds
from ...common_regression.selection import parameter_grid
from ..catalog import AX
from ..core import digest, file_hash
from ..stability.policy import constrain, score, upper_bound
from .combinations import all_combinations
from .config import SEEDS, SELECTION_POLICY, TFT_GRID
from .io import atomic_json
from .models import make_exhaustive_adapter
from .selection import cached_selection


def model_config():
    return yaml.safe_load(
        (AX / "strawberry_fruit_set_prediction/config/model_suite.yaml").read_text()
    )


def grid(model):
    values = (
        TFT_GRID
        if model == "tft"
        else parameter_grid(model_config()["tuning_grid"][model])
    )
    if model == "poisson":
        unique = {}
        for value in values:
            value = dict(value)
            value["alpha"] = max(0.1, float(value["alpha"]))
            unique[json.dumps(value, sort_keys=True)] = value
        values = list(unique.values())
    return values


def temporal_folds(train):
    for number, (a, b) in enumerate(
        build_expanding_folds(train, "feature_date", n_folds=3), 1
    ):
        validation = train.iloc[b]
        fit = train.iloc[a]
        fit = fit[fit.target_date < validation.feature_date.min()]
        if fit.empty:
            raise ValueError("Purged temporal fold is empty")
        yield number, fit, validation


def _gpu_state():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        util, used, total = [float(x.strip()) for x in out.splitlines()[0].split(",")]
        return {
            "GPU_Utilization_Percent": util,
            "GPU_Memory_Used_MiB": used,
            "GPU_Memory_Total_MiB": total,
        }
    except Exception as exc:
        return {"GPU_Monitor_Error": f"{type(exc).__name__}: {exc}"}


class TargetCampaign:
    def __init__(self, root: Path, target: str):
        self.root = Path(root)
        self.target = target
        self.prepared = self.root / "prepared" / target
        self.spec = json.loads((self.prepared / "manifest.json").read_text())
        self.train = self.load("train")
        self.validation = self.load("validation")
        if set(self.train.facility_id) & set(self.validation.facility_id):
            raise ValueError("Train/validation facility overlap")

    def load(self, split):
        path = self.prepared / f"{split}.parquet"
        if file_hash(path) != self.spec["dataset_hashes"][split]:
            raise ValueError(f"Prepared {split} changed")
        return pd.read_parquet(path)

    def candidates(self, combination):
        features = sorted(
            {
                f
                for group in combination["Groups"]
                for f in self.spec["groups"].get(group, [])
            }
        )
        categorical = [f for f in features if f in self.spec["categorical"]]
        numeric = [f for f in features if f not in categorical]
        return numeric, categorical

    def evaluate(
        self,
        combination,
        model,
        params,
        seed,
        stage,
        fit,
        evaluation,
        fold=None,
        epochs=None,
        frozen_selection=None,
    ):
        numeric, categorical = self.candidates(combination)
        selection = frozen_selection or cached_selection(
            self.root,
            self.target,
            combination["Combination_ID"],
            fit,
            numeric,
            categorical,
            stage,
            fold,
        )
        identity = {
            "target": self.target,
            "combination": combination["Combination_ID"],
            "groups": combination["Groups"],
            "model": model,
            "params": params,
            "seed": seed,
            "stage": stage,
            "fold": fold,
            "epochs": epochs,
            "fit_hash": digest(sorted(fit.row_id.astype(str).tolist())),
            "eval_hash": digest(sorted(evaluation.row_id.astype(str).tolist())),
            "selected": selection.selected,
            "dataset_hashes": self.spec["dataset_hashes"],
            "selection_policy": SELECTION_POLICY,
        }
        run_id = "RUN_EXH_" + digest(identity)[:24]
        dest = self.root / "runs" / run_id
        result_path = dest / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        dest.mkdir(parents=True, exist_ok=True)
        atomic_json(dest / "identity.json", identity)
        start = time.monotonic()
        record = {
            "Run_ID": run_id,
            "Target_ID": self.target,
            "Combination_ID": combination["Combination_ID"],
            "Variable_Groups": combination["Groups"],
            "Model_Type": model,
            "Seed": seed,
            "Stage": stage,
            "Fold_ID": fold,
            "Params": params,
            "Status": "running",
            "Selected_Features": selection.selected,
            "Selected_Numeric": selection.selected_numeric,
            "Selected_Categorical": selection.selected_categorical,
            "Feature_Count": len(selection.selected),
            "N_Eval": len(evaluation),
            "Eval_Row_Hash": identity["eval_hash"],
            "Started_At": time.time(),
            **_gpu_state(),
        }
        atomic_json(dest / "running.json", record)
        try:
            if not selection.selected:
                raise ValueError("All candidate features were excluded")
            model_params = dict(params)
            if model == "random_forest":
                model_params.setdefault(
                    "n_jobs", max(1, int(os.getenv("AX_RF_THREADS", "24")))
                )
            if model == "catboost":
                model_params.setdefault(
                    "thread_count", int(os.getenv("AX_GPU_HOST_THREADS", "4"))
                )
            adapter = make_exhaustive_adapter(
                model,
                selection,
                model_params,
                seed,
                self.spec["entity"],
                dest / "checkpoint",
            )
            adapter.fit(
                fit,
                "target",
                validation=evaluation if stage == "internal_cv" else None,
                fixed_epochs=epochs,
            )
            raw = adapter.predict(evaluation)
            bounded, unstable = constrain(
                raw, upper_bound(fit, self.spec["components"])
            )
            if unstable:
                raise FloatingPointError("Raw prediction exceeded stability multiplier")
            metrics = {
                "raw": score(evaluation.target, raw),
                "bounded": score(evaluation.target, bounded),
            }
            predictions = evaluation[
                [
                    "row_id",
                    "facility_id",
                    "crop_sn",
                    "sample_num",
                    "feature_date",
                    "target_date",
                    "target",
                ]
            ].copy()
            predictions["prediction_raw"] = raw
            predictions["prediction_bounded"] = bounded
            predictions["Run_ID"] = run_id
            predictions.to_parquet(dest / "predictions.parquet", index=False)
            if stage == "test":
                adapter.save(dest / "model.bin")
            record.update(
                Status="success", Metrics=metrics, Best_Epoch=adapter.best_epoch
            )
        except Exception as exc:
            record.update(Status="failed", Error=f"{type(exc).__name__}: {exc}")
        record.update(
            Duration_Seconds=time.monotonic() - start,
            Completed_At=time.time(),
            **{"GPU_After": _gpu_state()},
        )
        atomic_json(result_path, record)
        (dest / "running.json").unlink(missing_ok=True)
        return record

    def tune_and_validate(self, combination, model):
        summary = (
            self.root
            / "summaries"
            / self.target
            / model
            / f"{combination['Combination_ID']}.json"
        )
        if summary.exists():
            return json.loads(summary.read_text())
        unavailable = sorted(
            set(combination["Groups"]) & set(self.spec["unavailable_groups"])
        )
        if unavailable:
            result = {
                "Target_ID": self.target,
                "Combination_ID": combination["Combination_ID"],
                "Variable_Groups": combination["Groups"],
                "Model_Type": model,
                "Status": "ineligible",
                "Reason": "unavailable_groups:" + ",".join(unavailable),
            }
            atomic_json(summary, result)
            return result
        choices = []
        for params in grid(model):
            runs = [
                self.evaluate(
                    combination, model, params, 42, "internal_cv", fit, val, fold=fold
                )
                for fold, fit, val in temporal_folds(self.train)
            ]
            if all(r["Status"] == "success" for r in runs):
                epochs = [r.get("Best_Epoch") for r in runs if r.get("Best_Epoch")]
                choices.append(
                    (
                        float(np.mean([r["Metrics"]["bounded"]["rmse"] for r in runs])),
                        float(np.mean([r["Metrics"]["bounded"]["mae"] for r in runs])),
                        json.dumps(params, sort_keys=True),
                        params,
                        int(statistics.median(epochs)) if epochs else None,
                    )
                )
        if not choices:
            result = {
                "Target_ID": self.target,
                "Combination_ID": combination["Combination_ID"],
                "Variable_Groups": combination["Groups"],
                "Model_Type": model,
                "Status": "failed",
                "Reason": "no_successful_complete_cv_configuration",
            }
            atomic_json(summary, result)
            return result
        winner = min(choices, key=lambda x: x[:3])
        params, epochs = winner[3], winner[4]
        validation_runs = [
            self.evaluate(
                combination,
                model,
                params,
                seed,
                "validation",
                self.train,
                self.validation,
                epochs=epochs,
            )
            for seed in SEEDS
        ]
        successful = [r for r in validation_runs if r["Status"] == "success"]
        result = {
            "Target_ID": self.target,
            "Combination_ID": combination["Combination_ID"],
            "Variable_Groups": combination["Groups"],
            "Model_Type": model,
            "Status": "success" if len(successful) == len(SEEDS) else "failed",
            "Params": params,
            "Best_Epoch": epochs,
            "Seed_Count": len(successful),
        }
        if successful:
            result.update(
                Validation_RMSE=float(
                    np.mean([r["Metrics"]["bounded"]["rmse"] for r in successful])
                ),
                Validation_MAE=float(
                    np.mean([r["Metrics"]["bounded"]["mae"] for r in successful])
                ),
                Feature_Count=min(r["Feature_Count"] for r in successful),
            )
        if len(successful) != len(SEEDS):
            result["Reason"] = "incomplete_validation_seeds"
        atomic_json(summary, result)
        return result

    def finalize_model(self, model):
        selections = []
        for combination in all_combinations():
            path = (
                self.root
                / "summaries"
                / self.target
                / model
                / f"{combination['Combination_ID']}.json"
            )
            if path.exists():
                value = json.loads(path.read_text())
                if value.get("Status") == "success":
                    selections.append(value)
        if not selections:
            return {
                "Target_ID": self.target,
                "Model_Type": model,
                "Status": "failed",
                "Reason": "no_eligible_combination",
            }
        winner = min(
            selections,
            key=lambda x: (
                x["Validation_RMSE"],
                x["Validation_MAE"],
                x["Feature_Count"],
                x["Combination_ID"],
            ),
        )
        combination = next(
            x
            for x in all_combinations()
            if x["Combination_ID"] == winner["Combination_ID"]
        )
        numeric, categorical = self.candidates(combination)
        frozen = cached_selection(
            self.root,
            self.target,
            combination["Combination_ID"],
            self.train,
            numeric,
            categorical,
            "validation",
            None,
        )
        test = self.load("test")
        combined = pd.concat([self.train, self.validation], ignore_index=True)
        runs = [
            self.evaluate(
                combination,
                model,
                winner["Params"],
                seed,
                "test",
                combined,
                test,
                epochs=winner.get("Best_Epoch"),
                frozen_selection=frozen,
            )
            for seed in SEEDS
        ]
        frozen_record = {
            "winner": winner,
            "test_runs": [r["Run_ID"] for r in runs],
            "test_accessed_after_selection": True,
            "frozen_features": frozen.selected,
            "Status": "success"
            if all(r["Status"] == "success" for r in runs)
            else "failed",
        }
        atomic_json(
            self.root / "selections" / self.target / f"{model}.json", frozen_record
        )
        return frozen_record


def run_lane(root: Path, target: str, model: str):
    campaign = TargetCampaign(root, target)
    results = []
    for combo in all_combinations():
        if (Path(root) / "STOP").exists():
            break
        results.append(campaign.tune_and_validate(combo, model))
    return {"target": target, "model": model, "completed": len(results)}
