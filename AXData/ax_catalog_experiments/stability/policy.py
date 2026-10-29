from __future__ import annotations
import numpy as np
import pandas as pd
from ...common_regression.metrics import regression_metrics

POLICY = dict(version="2.1", poisson_alpha_floor=0.1, scale_floor=1.0, input_clip=10.0, gradient_norm=1.0,
              instability_multiplier=100.0, prediction='raw_and_train_bounded_no_rounding',
              test_notice='Previously inspected holdout; not a new independent test')

class StablePreprocessor:
    """Fixed-width train-only median/mean/std transform for registered numeric inputs."""
    def __init__(self, features, categorical=(), scale=True):
        if categorical: raise ValueError('Categorical inputs require an explicitly versioned encoder')
        self.features=list(features)
    def values(self, frame):
        return frame[self.features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    def fit(self, train):
        x=self.values(train)
        missing=x.columns[x.isna().all()].tolist()
        if missing: raise ValueError(f'Frozen inputs entirely missing in fit fold: {missing}')
        self.medians=x.median(); x=x.fillna(self.medians)
        self.mean=x.mean(); self.std=x.std(ddof=0); self.scale=self.std.clip(lower=1.0)
        self.lo=x.min(); self.hi=x.max()
        return self
    def unbounded(self, frame): return (self.values(frame).fillna(self.medians)-self.mean)/self.scale
    def transform(self, frame):
        out=self.unbounded(frame).clip(-10,10).to_numpy(np.float32)
        if not np.isfinite(out).all(): raise ValueError('Nonfinite transformed input')
        return out
    def diagnostics(self, frame):
        x=self.values(frame); z=self.unbounded(frame)
        return pd.DataFrame({'Variable_ID':self.features,'Missing_Rate':x.isna().mean().values,
          'Train_Mean':self.mean.values,'Train_Std':self.std.values,'Train_Min':self.lo.values,
          'Train_Max':self.hi.values,'Eval_Min':x.min().values,'Eval_Max':x.max().values,
          'Unbounded_Abs_Max':z.abs().max().values,'Input_Clip_Rate':z.abs().gt(10).mean().values,
          'Outside_Train_Range_Rate':(x.lt(self.lo)|x.gt(self.hi)).mean().values})

def upper_bound(train, components):
    cols=[f'target_truss_{i}' for i in components]
    x=train[cols].apply(pd.to_numeric,errors='coerce')
    maxima=x.max()
    for i,c in zip(components,cols):
        observed=f'count_{i}'
        if observed in train:
            maxima[c]=max(maxima[c],pd.to_numeric(train[observed],errors='coerce').max())
    if maxima.isna().any() or not np.isfinite(maxima).all() or (maxima<0).any():
        raise ValueError('Cannot establish training-only component upper bounds')
    return float(maxima.sum())

def score(y, prediction):
    y=np.asarray(y,float); p=np.asarray(prediction,float)
    if len(y)!=len(p) or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite or empty metric inputs')
    result=regression_metrics(y,p)
    if len(y)<2 or np.ptp(y)==0: result['r2']=None
    return {k:float(v) if v is not None and np.isfinite(v) else None for k,v in result.items()}

def constrain(raw, upper):
    raw=np.asarray(raw,float)
    if not np.isfinite(raw).all(): raise FloatingPointError('Nonfinite raw predictions')
    unstable=bool(np.any(np.abs(raw)>100*max(upper,1)))
    return np.clip(raw,0,upper),unstable
