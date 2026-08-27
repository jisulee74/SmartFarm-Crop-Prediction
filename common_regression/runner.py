from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import SplitBundle, TargetSpec
from .evaluation import evaluate_final_and_persistence, sha256, write_json
from .feature_blocks import flower_feature_sets, forward_select_blocks
from .selection import (cross_validate_model, select_feature_set_with_fixed_configs,
                        tune_models_on_selected_features, validate_learning_models)


class Stage(str, Enum):
    PREPARED = "prepared"
    INTERNAL_FEATURE_SELECTION = "internal_feature_selection"
    INTERNAL_TUNING = "internal_tuning"
    EXTERNAL_VALIDATION = "external_validation"
    FROZEN = "frozen"
    FINAL_FIT = "final_fit"
    TEST_ONCE = "test_once"


class CommonRegressionRunner:
    """Stateful orchestrator that prevents Test access before configuration freeze."""

    def __init__(self, bundle: SplitBundle, blocks: dict[str, list[str]], categorical: list[str],
                 config: dict[str, Any], artifact_dir: Path):
        self.bundle, self.blocks, self.categorical = bundle, blocks, categorical
        self.config, self.artifact_dir = config, artifact_dir
        self.stage = Stage.PREPARED
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _require(self, stage: Stage) -> None:
        if self.stage != stage:
            raise RuntimeError(f"Expected stage {stage.value}, got {self.stage.value}")

    def _write_manifest(self, **extra) -> None:
        write_json(self.artifact_dir / "run_manifest.json", {
            "target": self.bundle.spec.name, "stage": self.stage.value,
            "train_rows": len(self.bundle.train), "validation_rows": len(self.bundle.validation),
            "test_path": str(self.bundle.test_path), "test_accessed": self.stage == Stage.TEST_ONCE,
            **extra,
        })

    def _fixed_evaluator(self, feature_set: str, features: list[str]) -> pd.DataFrame:
        parts = []
        for model, params in self.config["feature_selection_baseline_params"].items():
            values = cross_validate_model(
                self.bundle.train, self.bundle.spec.target, self.bundle.spec.target_date,
                feature_set, features, self.categorical, model, params,
                self.config["feature_selection_seeds"],
            )
            parts.append(values.groupby(["feature_set", "model", "feature_count"], as_index=False).agg(
                rmse=("rmse", "mean"), mae=("mae", "mean")))
        return pd.concat(parts, ignore_index=True)

    def select_features(self) -> str:
        self._require(Stage.PREPARED)
        if self.bundle.spec.name == "flower_cluster":
            sets, aliases = flower_feature_sets(self.blocks)
            selected, trials, summary = select_feature_set_with_fixed_configs(
                self.bundle.train, self.bundle.spec.target, self.bundle.spec.target_date,
                sets, self.categorical, self.config["feature_selection_baseline_params"],
                self.config["feature_selection_seeds"])
            write_json(self.artifact_dir / "feature_sets.json", {"sets": sets, "aliases": aliases})
        else:
            sets, history = forward_select_blocks(self.blocks, self._fixed_evaluator)
            parts = [self._fixed_evaluator(name, features) for name, features in sets.items()]
            model_metrics = pd.concat(parts, ignore_index=True)
            from .feature_blocks import choose_final_feature_set, summarize_feature_sets
            summary = summarize_feature_sets(model_metrics)
            selected = str(choose_final_feature_set(summary).feature_set)
            trials = model_metrics
            history.to_csv(self.artifact_dir / "internal_cv_feature_selection_history.csv", index=False)
            write_json(self.artifact_dir / "feature_sets.json", {"sets": sets})
        self.feature_sets, self.selected_feature_set = sets, selected
        self.features = sets[selected]
        self.bundle.spec.validate_features(self.features)
        out = self.artifact_dir / "internal_cv" / "feature_set_selection"; out.mkdir(parents=True, exist_ok=True)
        trials.to_csv(out / "trials.csv", index=False); summary.to_csv(out / "summary.csv", index=False)
        self.stage = Stage.INTERNAL_FEATURE_SELECTION; self._write_manifest(selected_feature_set=selected)
        return selected

    def tune(self) -> dict[str, dict[str, Any]]:
        self._require(Stage.INTERNAL_FEATURE_SELECTION)
        params, epochs, trials = tune_models_on_selected_features(
            self.bundle.train, self.bundle.spec.target, self.bundle.spec.target_date,
            self.selected_feature_set, self.features, self.categorical,
            self.config["tuning_grid"], self.config["feature_selection_seeds"])
        out = self.artifact_dir / "internal_cv" / "hyperparameter_tuning"; out.mkdir(parents=True, exist_ok=True)
        trials.to_csv(out / "trials.csv", index=False)
        write_json(out / "selected_params.json", {"params": params, "fixed_epochs": epochs})
        self.params, self.epochs = params, epochs
        self.stage = Stage.INTERNAL_TUNING; self._write_manifest(selected_feature_set=self.selected_feature_set)
        return params

    def validate(self) -> str:
        self._require(Stage.INTERNAL_TUNING)
        model, table, fitted = validate_learning_models(
            self.bundle.train, self.bundle.validation, self.bundle.spec.target, self.features,
            self.categorical, self.params, self.epochs, self.config["validation_seeds"],
            self.bundle.spec.current_value)
        out = self.artifact_dir / "validation"; out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "model_metrics.csv", index=False)
        learning = table[table.is_learning_model]; persistence = table[~table.is_learning_model].iloc[0]
        winner = learning[learning.model.eq(model)].iloc[0]
        self.selected_model, self.validation_table = model, table
        self.stage = Stage.EXTERNAL_VALIDATION
        self._write_manifest(selected_feature_set=self.selected_feature_set, selected_learning_model=model,
                             persistence_baseline="persistence",
                             baseline_beaten_on_validation=bool(winner.rmse < persistence.rmse))
        return model

    def freeze(self) -> None:
        self._require(Stage.EXTERNAL_VALIDATION)
        frozen = {"selected_feature_set": self.selected_feature_set, "features": self.features,
                  "selected_learning_model": self.selected_model, "params": self.params[self.selected_model],
                  "fixed_epochs": self.epochs[self.selected_model], "seeds": self.config["validation_seeds"],
                  "test_sha256_before_access": sha256(self.bundle.test_path)}
        write_json(self.artifact_dir / "selected_configuration.json", frozen)
        self.frozen = frozen; self.stage = Stage.FROZEN; self._write_manifest(**frozen)

    def fit_final(self) -> None:
        self._require(Stage.FROZEN)
        from .models import make_adapter
        combined = pd.concat([self.bundle.train, self.bundle.validation], ignore_index=True)
        seeds = self.config["validation_seeds"] if self.selected_model in {"mlp", "tabm"} else [self.config["validation_seeds"][0]]
        self.final_adapters = []
        model_dir = self.artifact_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            adapter = make_adapter(self.selected_model, self.features, self.categorical,
                                   self.params[self.selected_model], seed)
            adapter.fit(combined, self.bundle.spec.target, fixed_epochs=self.epochs[self.selected_model])
            adapter.save(model_dir / f"{self.selected_model}_seed_{seed}.bin")
            self.final_adapters.append(adapter)
        self.stage = Stage.FINAL_FIT; self._write_manifest(**self.frozen)

    def test_once(self) -> dict:
        self._require(Stage.FINAL_FIT)
        test = self.bundle.load_test()
        self.bundle.spec.validate_frame(test, "test")
        import numpy as np
        prediction = np.mean([model.predict(test) for model in self.final_adapters], axis=0)
        result = evaluate_final_and_persistence(test, self.bundle.spec.row_key, self.bundle.spec.target,
                                                self.bundle.spec.current_value, prediction,
                                                self.artifact_dir, self.config.get("rounded_output", True))
        if sha256(self.bundle.test_path) != self.frozen["test_sha256_before_access"]:
            raise RuntimeError("Test dataset changed after freeze")
        self.stage = Stage.TEST_ONCE
        self._write_manifest(**self.frozen, persistence_baseline="persistence", **result)
        write_json(self.artifact_dir / "test_access_log.json", {"access_count": 1, "stage": self.stage.value})
        return result
