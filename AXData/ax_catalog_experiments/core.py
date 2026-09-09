from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def freeze(path: Path, value):
    """An existing identity can only be resumed with identical content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen definition changed: {path}')
    else:
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        temp.replace(path)


def union(categories, selected):
    missing = set(selected) - set(categories)
    if missing: raise ValueError(f'Unknown categories: {sorted(missing)}')
    return sorted({f for c in selected for f in categories[c]})


def qualify(train, candidates):
    """Train-only screening, including affine dependence; retain explicit audit."""
    kept, audit, basis = [], [], []
    for f in candidates:
        s = pd.to_numeric(train[f], errors='coerce').replace([np.inf, -np.inf], np.nan) if f in train else pd.Series(np.nan,index=train.index)
        reason = 'eligible'
        if s.isna().mean() > .5: reason = 'missing_over_50_percent'
        elif s.nunique() < 2: reason = 'constant_or_empty'
        else:
            x = s.fillna(s.median()).to_numpy(float)
            x = (x-x.mean()) / x.std()
            residual = x.copy()
            # Modified Gram-Schmidt avoids repeatedly solving large systems.
            for q in basis: residual -= q * np.dot(q, residual)
            for q in basis: residual -= q * np.dot(q, residual)
            norm = np.linalg.norm(residual)
            if norm < 1e-9*np.sqrt(len(x)): reason = 'affine_dependent_on_prior_members'
            else: basis.append(residual/norm); kept.append(f)
        audit.append({'Variable_ID':f,'Missing_Rate':float(s.isna().mean()),'Status':reason})
    return kept, audit


def growth_features(frame, entity, bases):
    d = frame.sort_values([*entity,'feature_date'],kind='stable').copy()
    if d.duplicated([*entity,'feature_date']).any(): raise ValueError('Conflicting observation keys')
    groups = d.groupby(entity,sort=False,dropna=False)
    elapsed = groups.feature_date.diff().dt.total_seconds()/86400
    values = {}
    for base in bases:
        s = pd.to_numeric(d[base],errors='coerce')
        g = s.groupby([d[k] for k in entity],sort=False,dropna=False)
        values[base+'__current'] = s
        values[base+'__delta'] = s-g.shift(1)
        values[base+'__delta_per_day'] = values[base+'__delta']/elapsed.replace(0,np.nan)
        for lag in (1,2,3): values[f'{base}__lag{lag}'] = g.shift(lag)
        for w in (2,3,4):
            for stat in ('mean','std','min','max'):
                values[f'{base}__roll{w}_{stat}'] = g.transform(lambda x: getattr(x.rolling(w,min_periods=w),stat)(**({'ddof':0} if stat=='std' else {})))
            slopes = pd.Series(np.nan,index=d.index)
            for _, part in groups:
                x = (part.feature_date-part.feature_date.iloc[0]).dt.total_seconds().to_numpy()/86400
                y = pd.to_numeric(part[base],errors='coerce').to_numpy(float)
                for i in range(w-1,len(part)):
                    xx, yy = x[i-w+1:i+1], y[i-w+1:i+1]
                    if np.isfinite(yy).all() and np.ptp(xx)>0:
                        slopes.loc[part.index[i]] = np.polyfit(xx-xx[0],yy,1)[0]
            values[f'{base}__roll{w}_slope'] = slopes
    values['days_since_previous'] = elapsed
    values['days_since_first_measurement'] = (d.feature_date-groups.feature_date.transform('min')).dt.days
    values['history_count'] = groups.cumcount()+1
    week = d.feature_date.dt.isocalendar().week.astype(float)
    values['week_sin'] = np.sin(2*np.pi*week/52.1775)
    values['week_cos'] = np.cos(2*np.pi*week/52.1775)
    return pd.concat([d.drop(columns=list(set(d)&set(values))),pd.DataFrame(values,index=d.index)],axis=1)


def sensor_features(rows, sensors, definitions):
    """Only [feature_date-days, feature_date); never select on test values."""
    sensors = sensors.copy()
    sensors['meas_date'] = pd.to_datetime(sensors.meas_date)
    streams = {}
    for k,v in sensors.groupby(['facility_id','variable_id']):
        v=v.sort_values('meas_date')
        streams[k]=(v.meas_date.to_numpy('datetime64[ns]'),pd.to_numeric(v.sensor_value,errors='coerce').to_numpy(float))
    records=[]
    for r in rows.itertuples():
        result={}
        for variable, binary in definitions.items():
            stream=streams.get((r.facility_id,variable))
            for w in (1,3,7):
                stats=('active_fraction',) if binary else ('mean','std','min','max')
                if stream is None: x=np.array([])
                else:
                    times,raw_values=stream
                    left=np.searchsorted(times,np.datetime64(r.feature_date-pd.Timedelta(days=w)))
                    right=np.searchsorted(times,np.datetime64(r.feature_date))
                    x=raw_values[left:right]
                    x=x[np.isfinite(x)]
                    if binary: x=x[np.isin(x,[0,1])]
                for stat in stats:
                    result[f'{variable}__{w}d_{stat}'] = float(np.mean(x==1) if stat=='active_fraction' else getattr(np,stat)(x)) if len(x) else np.nan
        records.append(result)
    return pd.concat([rows.reset_index(drop=True),pd.DataFrame(records)],axis=1)


def facility_split(frame, seed=42):
    facilities=sorted(frame.facility_id.unique())
    if len(facilities)<6: raise ValueError('At least six facilities required; no temporal fallback')
    counts=frame.groupby('facility_id').size()
    rng=np.random.default_rng(seed); best=None
    n=max(1,round(len(facilities)*.15))
    for _ in range(5000):
        order=list(rng.permutation(facilities))
        val,test=sorted(order[:n]),sorted(order[n:2*n]); train=sorted(set(facilities)-set(val)-set(test))
        ratio=np.array([counts[train].sum(),counts[val].sum(),counts[test].sum()])/len(frame)
        key=(float(np.abs(ratio-[.7,.15,.15]).max()),tuple(val),tuple(test))
        if best is None or key<best[0]: best=(key,{'train':train,'validation':val,'test':test})
    return best[1]


def initial_groups(base, available):
    optional=sorted(set(available)-{base})
    return sorted({(base,),tuple(sorted(available)),*((x,) for x in optional),*(tuple(sorted((base,x))) for x in optional)})


def neighbours(selected, available, base):
    return {tuple(sorted(set(selected)^{c})) for c in available if c!=base}


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
