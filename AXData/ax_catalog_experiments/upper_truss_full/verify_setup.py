"""Preflight invariants and small, real-adapter probes before long campaigns."""
import json
import numpy as np
import pandas as pd
from AXData.ax_catalog_experiments.core import sensor_features
from AXData.ax_catalog_experiments.exhaustive.engine import temporal_folds
from AXData.ax_catalog_experiments.exhaustive.io import atomic_json
from AXData.ax_catalog_experiments.upper_truss_screening.run import PARAMS
from .prepare import OUT, prepare, windows
from .run import fit_run, frame, COMBINATIONS, MODELS


def verify():
    m=prepare()
    for crop,spec in m['crops'].items():
        fs={s:frame(crop,s) for s in ['train','validation','test']}
        for s,d in fs.items():
            assert set(d.facility_id)==set(spec['split'][s])
            assert d.row_id.is_unique
            assert d.target.notna().all() and d.current.notna().all()
            assert (d.feature_date<d.target_date).all() and d.gap_days.between(1,14).all()
            assert set(d.truss)==set(range(1,spec['limit']+1))
            assert d.series_truss.eq(d.truss.astype(str)).all()
            original=pd.read_parquet(OUT/'base'/crop/f'{s}.parquet')
            for c in ['row_id','truss','current','target','feature_date','target_date','weighted_position']:
                pd.testing.assert_series_equal(d[c],original[c])
            if crop=='strawberry':
                assert d.loc[d.truss==5,'upper_neighbor'].isna().all()
                assert d.positive_max.dropna().le(5).all()
            assert not {'target','target_date','gap_days','row_id','facility_id','crop_sn',
                        'sample_num','series_truss'} & set(sum(spec['groups'].values(),[]))
        assert not set(fs['train'].facility_id)&set(fs['test'].facility_id)
        assert not set(fs['validation'].facility_id)&set(fs['test'].facility_id)
        for _,a,b in temporal_folds(fs['train']):
            assert a.target_date.max()<b.feature_date.min()
    # Daily sufficient statistics reproduce raw [D-w,D) windows, including
    # population standard deviation and exclusion of observations at D.
    raw=pd.DataFrame({'facility_id':['f']*5,'variable_id':['EI__x']*5,
        'sensor_value':[1.,3.,8.,10.,1000.],
        'meas_date':pd.to_datetime(['2026-01-01 01:00','2026-01-01 12:00',
                                    '2026-01-02 03:00','2026-01-02 06:00','2026-01-03 00:00'])})
    d=raw.assign(day=raw.meas_date.dt.normalize(),sq=raw.sensor_value**2,
                 ones=raw.sensor_value.eq(1).astype(int),nonbinary=(~raw.sensor_value.isin([0,1])).astype(int))
    daily=d.groupby(['facility_id','variable_id','day']).agg(count=('sensor_value','size'),
        total=('sensor_value','sum'),sq=('sq','sum'),lo=('sensor_value','min'),
        hi=('sensor_value','max'),ones=('ones','sum'),nonbinary=('nonbinary','sum')).reset_index().rename(columns={'variable_id':'variable'})
    keys=pd.DataFrame({'facility_id':['f'],'feature_date':[pd.Timestamp('2026-01-03')]})
    actual,_=windows(keys,daily);expected=sensor_features(keys,raw,{'EI__x':False})
    for c in expected.columns.difference(keys.columns):
        np.testing.assert_allclose(actual[c],expected[c],rtol=1e-12,atol=1e-12)
    combo=next(c for c in COMBINATIONS if c['Groups']==['E1','E7'])
    a=frame('tomato','train').groupby('truss').head(20).reset_index(drop=True)
    b=frame('tomato','validation').groupby('truss').head(5).reset_index(drop=True)
    for model in MODELS:
        p=dict(PARAMS[model])
        if model=='random_forest':p['n_estimators']=5
        if model=='catboost':p['iterations']=5
        if model in ['mlp','tabm','tft']:p['max_epochs']=1
        r=fit_run('smoke','tomato',combo,model,p,42,'cv',a,b,fold='adapter_probe')
        print(model,r['status'],flush=True)
        assert r['status']=='success',r.get('error')
    atomic_json(OUT/'setup_verification.json',{'status':'passed','models':MODELS,
        'checks':['source target alignment','20/5 truss scope','facility split','purged temporal folds',
                  'feature allowlist','raw vs daily window equivalence','six adapter probes']})


if __name__=='__main__':
    verify()
