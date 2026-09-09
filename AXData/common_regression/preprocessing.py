from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TabularPreprocessor:
    def __init__(self, features: list[str], categorical: list[str], scale: bool):
        self.features = list(features)
        self.categorical = [x for x in categorical if x in features]
        self.numeric = [x for x in features if x not in self.categorical]
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scaler", StandardScaler()))
        self.transformer = ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), self.numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.categorical),
        ], verbose_feature_names_out=False)
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> "TabularPreprocessor":
        self.transformer.fit(train[self.features])
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit on Train first")
        values = np.asarray(self.transformer.transform(frame[self.features]), dtype=np.float32)
        if len(values) != len(frame) or not np.isfinite(values).all():
            raise ValueError("Preprocessing changed rows or produced invalid values")
        return values

    def save(self, path) -> None:
        joblib.dump(self, path)
