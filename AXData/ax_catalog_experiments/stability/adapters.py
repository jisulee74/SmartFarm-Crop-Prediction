from __future__ import annotations
import numpy as np
from ...common_regression.models import make_adapter, CatBoostAdapter, PoissonAdapter
from .policy import StablePreprocessor, POLICY

class StableCatBoost(CatBoostAdapter):
    def fit(self, train, target, validation=None, fixed_epochs=None):
        self.preprocessor=StablePreprocessor(self.features).fit(train)
        return super().fit(train,target,validation,fixed_epochs)
    def _prepare(self, frame):
        import pandas as pd
        return pd.DataFrame(self.preprocessor.transform(frame),columns=self.features,index=frame.index)

class RawPoisson(PoissonAdapter):
    def predict(self, frame):
        # Preserve genuine pre-constraint log-link outputs, including overflow failures.
        with np.errstate(over='ignore'):
            return np.exp(self.preprocessor.transform(frame)@self.model.coef_+self.model.intercept_)

def adapter(model, features, params, seed, entity, current):
    if model=='tft':
        from .tft import StableTFT
        return StableTFT(features,params,seed,entity,current)
    if model=='catboost': result=StableCatBoost(features,[],params,seed)
    elif model=='poisson': result=RawPoisson(features,[],params,seed)
    else: result=make_adapter(model,features,[],params,seed)
    if model=="poisson":
        result.params["alpha"]=max(float(result.params.get("alpha",1.0)),POLICY["poisson_alpha_floor"])
    result.preprocessor_factory=StablePreprocessor
    result.stability_checks=True
    return result
