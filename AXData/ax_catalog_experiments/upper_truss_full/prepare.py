"""Frozen 20/5-truss cohort, with sensor windows rebuilt from frozen raw data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from AXData.ax_catalog_experiments.core import file_hash, growth_features
from AXData.ax_catalog_experiments.exhaustive.io import atomic_json
from AXData.ax_catalog_experiments.exhaustive.prepare import _tomato, _strawberry, CONTEXT, GROWTH_MAP
from AXData.ax_catalog_experiments.exhaustive_e6_semantic.features import build_all_features
from AXData.ax_catalog_experiments.upper_truss_screening.prepare import prepare as base_prepare, ROOT, SNAP, KEY

OUT = ROOT / "ax_catalog_experiments/outputs/upper_truss_full_20261020"
SEMANTIC = ROOT / "ax_catalog_experiments/outputs/exhaustive_e6_semantic_v2"
BASE_FEATURES = ['current','lag1','lag2','lag3','delta','roll3_mean','past_max',
                 'past_observations','days_since_first_positive','days_since_previous',
                 'lower_neighbor','upper_neighbor','observed_trusses','positive_trusses',
                 'positive_min','positive_max','observed_total','weighted_position',
                 'position_previous','position_movement','relative_position','relative_max']


def daily_sensors(facilities):
    path = OUT / "sensor_daily.parquet"
    if path.exists():
        return pd.read_parquet(path)
    parts = []
    source = pq.ParquetFile(SNAP / "sfkr_hbfm_env_con.parquet")
    for i, batch in enumerate(source.iter_batches(
            columns=['facility_id','sect_code','fatr_code','sen_val','meas_date'],
            batch_size=1_000_000)):
        d = batch.to_pandas()
        d = d[d.facility_id.isin(facilities) & d.sect_code.isin(['EI','EO','NT'])].copy()
        if len(d):
            d['v'] = pd.to_numeric(d.sen_val, errors='coerce')
            d['day'] = pd.to_datetime(d.meas_date, errors='coerce').dt.normalize()
            d = d[np.isfinite(d.v) & d.day.notna()].copy()
            d['variable'] = d.sect_code.astype(str) + '__' + d.fatr_code.astype(str)
            d['sq'] = d.v ** 2
            d['ones'] = d.v.eq(1).astype(int)
            d['nonbinary'] = (~d.v.isin([0, 1])).astype(int)
            parts.append(d.groupby(['facility_id','variable','day']).agg(
                count=('v','size'), total=('v','sum'), sq=('sq','sum'),
                lo=('v','min'), hi=('v','max'), ones=('ones','sum'),
                nonbinary=('nonbinary','sum')).reset_index())
        if i % 20 == 0:
            print("sensor batches", i, "/", source.metadata.num_rows // 1_000_000 + 1, flush=True)
    d = pd.concat(parts).groupby(['facility_id','variable','day']).agg(
        count=('count','sum'),total=('total','sum'),sq=('sq','sum'),
        lo=('lo','min'),hi=('hi','max'),ones=('ones','sum'),
        nonbinary=('nonbinary','sum')).reset_index()
    d.to_parquet(path, index=False)
    return d


def windows(keys, daily):
    variables = daily.groupby('variable').nonbinary.sum().eq(0).to_dict()
    streams = {(f,v):g.sort_values('day') for (f,v),g in daily.groupby(['facility_id','variable'])}
    records = []
    groups = {g:[] for g in ['E3','E4','E5']}
    for v, binary in variables.items():
        group = {'EI':'E3','EO':'E4','NT':'E5'}[v.split('__')[0]]
        for w in [1,3,7]:
            groups[group] += [f'{v}__{w}d_{s}' for s in (
                ['active_fraction'] if binary else ['mean','std','min','max'])]
    for r in keys.itertuples():
        row = {'facility_id':r.facility_id,'feature_date':r.feature_date}
        for v,binary in variables.items():
            g = streams.get((r.facility_id,v))
            for w in [1,3,7]:
                x = None if g is None else g[(g.day >= r.feature_date-pd.Timedelta(days=w)) & (g.day < r.feature_date)]
                n = 0 if x is None else x['count'].sum()
                if binary:
                    row[f'{v}__{w}d_active_fraction'] = x.ones.sum()/n if n else np.nan
                else:
                    mean = x.total.sum()/n if n else np.nan
                    vals = [mean, np.sqrt(max(0,x.sq.sum()/n-mean**2)) if n else np.nan,
                            x.lo.min() if n else np.nan,x.hi.max() if n else np.nan]
                    for s,value in zip(['mean','std','min','max'],vals):
                        row[f'{v}__{w}d_{s}'] = value
        records.append(row)
    return pd.DataFrame(records), groups


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        for crop,spec in m['crops'].items():
            for split,h in spec['hashes'].items():
                assert file_hash(OUT/crop/f'{split}.parquet') == h
        return m
    base = base_prepare(OUT/'base', strawberry_limit=5)
    frames = {c:pd.concat([pd.read_parquet(OUT/'base'/c/f'{s}.parquet')
              for s in ['train','validation','test']],ignore_index=True) for c in base['crops']}
    keys = pd.concat([d[['facility_id','feature_date']] for d in frames.values()]).drop_duplicates()
    assert keys.feature_date.eq(keys.feature_date.dt.normalize()).all()
    daily = daily_sensors(set(keys.facility_id))
    sensor, sensor_groups = windows(keys, daily)
    semantic, coverage = build_all_features(keys, SEMANTIC/'semantic_streams')
    coverage.to_csv(OUT/'semantic_coverage.csv', index=False)
    e6 = [c for c in semantic if c.startswith('E6S__')]
    environmental = sensor.merge(semantic, on=['facility_id','feature_date'], validate='one_to_one')
    environmental.to_parquet(OUT/'environmental_features.parquet', index=False)
    result = {'crops':{}, 'limits':{'tomato':20,'strawberry':5},
              'common_features':['truss'], 'test_notice':'Previously inspected facility holdout; exploratory range decision; not new external validation',
              'sensor_policy':'raw frozen snapshot, [date-window,date), 1/3/7 days; E6 semantic streams',
              'source_hashes':{'sensors':file_hash(SNAP/'sfkr_hbfm_env_con.parquet'),
                               'base':file_hash(OUT/'base/manifest.json')}}
    for crop, d in frames.items():
        source, audit = (_tomato if crop=='tomato' else _strawberry)(SNAP.parent)
        source.crop_sn = source.crop_sn.astype(str)
        source.sample_num = source.sample_num.astype(str)
        veg = [v for v in GROWTH_MAP.values() if v in source]
        source = growth_features(source, KEY[:-1], veg)
        e2 = [c for c in source if any(c.startswith(v+'__') for v in veg)]
        e7 = [c for c in CONTEXT if c in source]
        context = source[KEY+e2+e7].copy()
        # Remove screening-cache columns; attach source-derived variables afresh.
        core = list(dict.fromkeys(KEY+['row_id','target_date','target','truss','split',
            'gap_days','days_since_crop_start']+BASE_FEATURES))
        d = d[core].merge(context,on=KEY,how='left',validate='many_to_one')
        d = d.merge(environmental,on=['facility_id','feature_date'],how='left',validate='many_to_one')
        e7 += ['days_since_crop_start']
        categorical = [c for c in ['cropping_system','cropping_season_name'] if c in d]
        for c in set(e2+e7+e6+sum(sensor_groups.values(),[]))-set(categorical):
            d[c] = pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        d['series_truss'] = d.truss.astype(str)
        groups = {'E1':BASE_FEATURES,'E2':e2,**sensor_groups,'E6':e6,'E7':e7}
        hashes, counts = {}, {}
        for split in ['train','validation','test']:
            x = d[d.split==split].reset_index(drop=True)
            assert x.row_id.is_unique and x.target.notna().all()
            assert x.truss.between(1,result['limits'][crop]).all()
            path = OUT/crop/f'{split}.parquet';path.parent.mkdir(exist_ok=True)
            x.to_parquet(path,index=False);hashes[split] = file_hash(path)
            counts[split] = {'rows':len(x),'positive':int(x.target.gt(0).sum()),
                             'facilities':int(x.facility_id.nunique())}
        result['crops'][crop] = {'groups':groups,'categorical':categorical,
            'entity':KEY[:-1]+['series_truss'],'hashes':hashes,'counts':counts,
            'split':base['crops'][crop]['split'],'limit':result['limits'][crop],
            'source_assignment_audit':audit}
        print(crop,counts,{g:len(v) for g,v in groups.items()},flush=True)
    atomic_json(manifest_path,result)
    return result


if __name__ == '__main__':
    prepare()
