from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..stability.adapters import adapter as stable_adapter


class FoldEncoder:
    def __init__(self, numeric, categorical, levels):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.levels = levels

    def _categories(self, frame):
        result = pd.DataFrame(index=frame.index)
        for name in self.categorical:
            definition = self.levels[name]
            known = set(definition["known"])
            rare = set(definition["rare"])
            values = frame[name].fillna("__MISSING__").astype(str)
            result[name] = values.map(
                lambda x: (
                    "__RARE__" if x in rare else x if x in known else "__UNKNOWN__"
                )
            )
        return result

    def fit(self, frame):
        self.medians = (
            frame[self.numeric]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .median()
            if self.numeric
            else pd.Series(dtype=float)
        )
        if self.categorical:
            kwargs = dict(handle_unknown="ignore", drop="first", dtype=np.float32)
            try:
                self.onehot = OneHotEncoder(sparse_output=False, **kwargs)
            except TypeError:
                self.onehot = OneHotEncoder(sparse=False, **kwargs)
            self.onehot.fit(self._categories(frame))
            self.encoded_categorical = self.onehot.get_feature_names_out(
                self.categorical
            ).tolist()
        else:
            self.onehot = None
            self.encoded_categorical = []
        self.feature_names = self.numeric + self.encoded_categorical
        return self

    def transform(self, frame):
        numeric = (
            frame[self.numeric]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
            .astype(np.float32)
            if self.numeric
            else pd.DataFrame(index=frame.index)
        )
        if self.onehot is None:
            return numeric
        encoded = pd.DataFrame(
            self.onehot.transform(self._categories(frame)),
            columns=self.encoded_categorical,
            index=frame.index,
        )
        return pd.concat([numeric, encoded], axis=1)


class EncodedAdapter:
    def __init__(self, model, selection, params, seed, entity, checkpoint_dir=None):
        self.model_name = model
        self.selection = selection
        self.params = dict(params)
        self.seed = seed
        self.entity = entity
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.best_epoch = None

    def _frame(self, source):
        encoded = self.encoder.transform(source)
        metadata = [
            c
            for c in [*self.entity, "feature_date", "target_date", "row_id", "target"]
            if c in source
        ]
        return pd.concat(
            [source[metadata].reset_index(drop=True), encoded.reset_index(drop=True)],
            axis=1,
        )

    def fit(self, train, target, validation=None, fixed_epochs=None):
        self.encoder = FoldEncoder(
            self.selection.selected_numeric,
            self.selection.selected_categorical,
            self.selection.categorical_levels,
        ).fit(train)
        if not self.encoder.feature_names:
            raise ValueError("All candidate features were excluded")
        params = dict(self.params)
        if self.checkpoint_dir and self.model_name in {"mlp", "tabm", "tft"}:
            params["_checkpoint_dir"] = str(self.checkpoint_dir)
        self.inner = stable_adapter(
            self.model_name,
            self.encoder.feature_names,
            params,
            self.seed,
            self.entity,
            self.encoder.feature_names[0],
        )
        fit = self._frame(train)
        val = self._frame(validation) if validation is not None else None
        self.inner.fit(fit, target, validation=val, fixed_epochs=fixed_epochs)
        self.best_epoch = self.inner.best_epoch
        return self

    def predict(self, frame):
        return self.inner.predict(self._frame(frame))

    def save(self, path):
        self.inner.save(path)


class NativeCatBoost:
    def __init__(self, selection, params, seed, checkpoint_dir=None):
        self.selection = selection
        self.params = dict(params)
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.best_epoch = None

    def _prepare(self, frame, fit=False):
        result = pd.DataFrame(index=frame.index)
        if fit:
            self.medians = (
                frame[self.selection.selected_numeric]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .median()
            )
        for c in self.selection.selected_numeric:
            result[c] = (
                pd.to_numeric(frame[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.medians[c])
                .astype(float)
            )
        for c in self.selection.selected_categorical:
            definition = self.selection.categorical_levels[c]
            known = set(definition["known"])
            rare = set(definition["rare"])
            values = frame[c].fillna("__MISSING__").astype(str)
            result[c] = values.map(
                lambda x: (
                    "__RARE__" if x in rare else x if x in known else "__UNKNOWN__"
                )
            )
        return result

    def fit(self, train, target, validation=None, fixed_epochs=None):
        from catboost import CatBoostRegressor

        x = self._prepare(train, fit=True)
        params = {
            **self.params,
            "random_seed": self.seed,
            "allow_writing_files": bool(self.checkpoint_dir),
            "verbose": False,
            "task_type": "GPU",
            "devices": "0",
        }
        stopping = int(params.pop("early_stopping_rounds", 150))
        if fixed_epochs is not None:
            params["iterations"] = int(fixed_epochs)
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            params.update(
                save_snapshot=True,
                snapshot_file=str(self.checkpoint_dir / "catboost.snapshot"),
                snapshot_interval=60,
                train_dir=str(self.checkpoint_dir),
            )
        self.model = CatBoostRegressor(**params)
        cats = [x.columns.get_loc(c) for c in self.selection.selected_categorical]
        kwargs = {"cat_features": cats, "verbose": False}
        if validation is not None and fixed_epochs is None:
            kwargs.update(
                eval_set=(self._prepare(validation), validation[target]),
                early_stopping_rounds=stopping,
            )
        self.model.fit(x, train[target], **kwargs)
        best = self.model.get_best_iteration()
        self.best_epoch = (
            max(1, best + 1)
            if best is not None and best >= 0
            else int(params.get("iterations", 1))
        )
        return self

    def predict(self, frame):
        return np.asarray(self.model.predict(self._prepare(frame)), float)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)


def make_exhaustive_adapter(
    model, selection, params, seed, entity, checkpoint_dir=None
):
    if model == "catboost":
        return NativeCatBoost(selection, params, seed, checkpoint_dir)
    return EncodedAdapter(model, selection, params, seed, entity, checkpoint_dir)
