from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor

from .preprocessing import TabularPreprocessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class RegressionAdapter(ABC):
    def __init__(self, features: list[str], categorical: list[str], params: dict[str, Any], seed: int):
        self.features, self.categorical = list(features), list(categorical)
        self.params, self.seed = dict(params), int(seed)
        self.best_epoch: int | None = None

    @abstractmethod
    def fit(self, train: pd.DataFrame, target: str, validation: pd.DataFrame | None = None,
            fixed_epochs: int | None = None) -> "RegressionAdapter": ...

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class PersistenceAdapter(RegressionAdapter):
    def __init__(self, current_value: str):
        super().__init__([current_value], [], {}, 0)
        self.current_value = current_value

    def fit(self, train, target, validation=None, fixed_epochs=None):
        return self

    def predict(self, frame):
        return frame[self.current_value].to_numpy(float)


class SklearnAdapter(RegressionAdapter):
    model_class = None
    scale = False

    def fit(self, train, target, validation=None, fixed_epochs=None):
        set_seed(self.seed)
        self.preprocessor = TabularPreprocessor(self.features, self.categorical, self.scale).fit(train)
        params = {**self.params}
        if self.model_class is RandomForestRegressor:
            params.setdefault("random_state", self.seed)
            params.setdefault("n_jobs", -1)
        self.model = self.model_class(**params)
        self.model.fit(self.preprocessor.transform(train), train[target].to_numpy(float))
        return self

    def predict(self, frame):
        return np.asarray(self.model.predict(self.preprocessor.transform(frame)), dtype=float)


class PoissonAdapter(SklearnAdapter):
    model_class = PoissonRegressor
    scale = True

    def predict(self, frame):
        # Evaluate the log-link explicitly so temporal extrapolation cannot overflow exp.
        values = self.preprocessor.transform(frame)
        linear = values @ self.model.coef_ + self.model.intercept_
        return np.exp(np.clip(linear, -30.0, 30.0)).astype(float)


class RandomForestAdapter(SklearnAdapter):
    model_class = RandomForestRegressor


class CatBoostAdapter(RegressionAdapter):
    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[self.features].copy()
        for feature in self.categorical:
            result[feature] = result[feature].fillna("__MISSING__").astype(str)
        for feature in set(self.features) - set(self.categorical):
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
        return result

    def fit(self, train, target, validation=None, fixed_epochs=None):
        from catboost import CatBoostRegressor
        set_seed(self.seed)
        params = {**self.params, "random_seed": self.seed, "allow_writing_files": False, "verbose": False}
        stopping = int(params.pop("early_stopping_rounds", 150))
        if fixed_epochs is not None:
            params["iterations"] = int(fixed_epochs)
        self.model = CatBoostRegressor(**params)
        cat = [self.features.index(x) for x in self.categorical]
        kwargs = {"cat_features": cat, "verbose": False}
        if validation is not None and fixed_epochs is None:
            kwargs.update(eval_set=(self._prepare(validation), validation[target]),
                          early_stopping_rounds=stopping)
        self.model.fit(self._prepare(train), train[target], **kwargs)
        iteration = self.model.get_best_iteration()
        self.best_epoch = max(1, int(iteration) + 1) if iteration is not None and iteration >= 0 else int(params.get("iterations", 1))
        return self

    def predict(self, frame):
        return np.asarray(self.model.predict(self._prepare(frame)), dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)


class _TorchAdapter(RegressionAdapter):
    model_kind = "mlp"

    def _make_model(self, input_dim: int):
        import torch
        from torch import nn
        if self.model_kind == "mlp":
            hidden = self.params.get("hidden", [128, 64])
            layers: list[nn.Module] = []
            width = input_dim
            for next_width in hidden:
                layers.extend([nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(self.params.get("dropout", 0.1))])
                width = next_width
            layers.append(nn.Linear(width, 1))
            return nn.Sequential(*layers)
        from tabm import TabM
        return TabM.make(n_num_features=input_dim, cat_cardinalities=[], d_out=1,
                         arch_type="tabm", k=int(self.params.get("k", 16)),
                         n_blocks=int(self.params.get("n_blocks", 2)),
                         dropout=float(self.params.get("dropout", 0.0)))

    def _forward(self, x):
        output = self.model(x)
        if self.model_kind == "tabm":
            output = output.mean(dim=1)
        return output.squeeze(-1)

    def fit(self, train, target, validation=None, fixed_epochs=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        set_seed(self.seed)
        self.preprocessor = TabularPreprocessor(self.features, self.categorical, scale=True).fit(train)
        x = torch.tensor(self.preprocessor.transform(train)); y = torch.tensor(train[target].to_numpy(np.float32))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._make_model(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.params.get("learning_rate", 1e-3)),
                                      weight_decay=float(self.params.get("weight_decay", 3e-4)))
        loader = DataLoader(TensorDataset(x, y), batch_size=min(int(self.params.get("batch_size", 64)), len(x)),
                            shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        max_epochs = int(fixed_epochs or self.params.get("max_epochs", 500))
        patience = int(self.params.get("early_stopping_rounds", 50))
        best_loss, best_state, stale = float("inf"), None, 0
        val_xy = None
        if validation is not None:
            val_xy = (torch.tensor(self.preprocessor.transform(validation)).to(device),
                      torch.tensor(validation[target].to_numpy(np.float32)).to(device))
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(); loss = torch.mean((self._forward(bx) - by) ** 2)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); optimizer.step()
            if val_xy is None:
                self.best_epoch = epoch
                continue
            self.model.eval()
            with torch.no_grad():
                value = float(torch.sqrt(torch.mean((self._forward(val_xy[0]) - val_xy[1]) ** 2)).cpu())
            if value < best_loss - 1e-8:
                best_loss, best_state, stale, self.best_epoch = value, copy.deepcopy(self.model.state_dict()), 0, epoch
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        return self

    def predict(self, frame):
        import torch
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(self.preprocessor.transform(frame))
            return self._forward(x).cpu().numpy().astype(float)

    def save(self, path: Path) -> None:
        import torch
        torch.save({"state_dict": self.model.state_dict(), "features": self.features,
                    "categorical": self.categorical, "params": self.params, "seed": self.seed,
                    "best_epoch": self.best_epoch, "preprocessor": self.preprocessor}, path)


class MLPAdapter(_TorchAdapter):
    model_kind = "mlp"


class TabMAdapter(_TorchAdapter):
    model_kind = "tabm"


ADAPTERS = {"poisson": PoissonAdapter, "random_forest": RandomForestAdapter,
            "catboost": CatBoostAdapter, "mlp": MLPAdapter, "tabm": TabMAdapter}


def make_adapter(name: str, features: list[str], categorical: list[str], params: dict, seed: int):
    if name not in ADAPTERS:
        raise KeyError(f"Unknown learning model: {name}")
    return ADAPTERS[name](features, categorical, params, seed)
