"""Validate saved predictions and export a self-contained publication bundle."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from AXData.ax_catalog_experiments.core import file_hash
from AXData.ax_catalog_experiments.exhaustive.io import atomic_json
from AXData.ax_catalog_experiments.upper_truss_screening.run import metrics
from .prepare import OUT
from .run import SEEDS, MODELS, COMBINATIONS, frame

METRICS=['rmse','mae','r2','ccc']


def aggregate(rows):
    result={}
    for k in METRICS:
        vals=[r[k] for r in rows if r[k] is not None]
        result[k]={'mean':float(np.mean(vals)) if vals else None,
                   'std':float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                   'valid_count':len(vals)}
    return result


def total_metrics(p,limit):
    keys=['facility_id','crop_sn','sample_num','feature_date','target_date']
    grouped=p.groupby(keys)
    full=grouped.filter(lambda x:len(x)==limit and x.truss.nunique()==limit)
    if not len(full):return None
    sums=full.groupby(keys)[['target','current','prediction']].sum()
    return {'n':len(sums),'metrics':metrics(sums,sums.prediction.to_numpy()),
            'persistence':metrics(sums,sums.current.to_numpy())}


def export(mode):
    root=OUT/mode
    progress=json.loads((root/'progress.json').read_text())
    assert progress['phase']=='complete' and progress['failed']==0
    manifest=json.loads((OUT/'manifest.json').read_text())
    settings=json.loads((root/'settings.json').read_text())
    assert settings['manifest_hash']==file_hash(OUT/'manifest.json')
    for crop,spec in manifest['crops'].items():
        for split,h in spec['hashes'].items():
            assert file_hash(OUT/crop/f'{split}.parquet')==h
    frozen=json.loads((root/'frozen_winners.json').read_text())
    summaries=[json.loads(p.read_text()) for p in (root/'summaries').glob('*/*/*.json')]
    expected={(c,m,x['Combination_ID']) for c in ['tomato','strawberry'] for m in MODELS for x in COMBINATIONS}
    assert {(r['crop'],r['model'],r['combo']) for r in summaries}==expected
    assert len(summaries)==1524 and all(r['status']=='success' for r in summaries)
    tests=[];validation=[];audited_rows=0
    for path in (root/'runs').glob('*/result.json'):
        r=json.loads(path.read_text())
        assert r['status']=='success',str(path)
        if r['stage']=='cv':continue
        p=pd.read_parquet(path.parent/'predictions.parquet')
        d=frame(r['crop'],r['stage'])
        for col in ['row_id','truss','target','current','feature_date','target_date']:
            pd.testing.assert_series_equal(p[col],d[col])
        assert np.isfinite(p[['prediction','prediction_raw']]).all().all()
        np.testing.assert_array_equal(p.prediction,np.maximum(p.prediction_raw,0))
        recalculated=metrics(p,p.prediction.to_numpy())
        for k,v in recalculated.items():
            if v is None:assert r['metrics'][k] is None
            else:np.testing.assert_allclose(v,r['metrics'][k],rtol=1e-10,atol=1e-10)
        audited_rows+=len(p)
        (tests if r['stage']=='test' else validation).append(r)
    assert len(validation)==1524*len(SEEDS[mode])
    assert len(tests)==12*len(SEEDS[mode])
    bundle={'schema':2,'campaign':OUT.name,'mode':mode,'seed_count':len(SEEDS[mode]),'seeds':SEEDS[mode],
            'limits':manifest['limits'],'comparison_count':1524,'selection':'validation pooled RMSE',
            'aggregation':'seed mean, then equal-weight truss metric mean; pooled and complete-case sum shown separately',
            'notice':manifest['test_notice'],'crops':{},'comparison':[]}
    for s in summaries:
        runs=[r for r in validation if r['crop']==s['crop'] and r['model']==s['model'] and r['combo']==s['combo']]
        assert {r['seed'] for r in runs}==set(SEEDS[mode])
        np.testing.assert_allclose(s['rmse'],np.mean([r['metrics']['rmse'] for r in runs]))
        bundle['comparison'].append({'crop':s['crop'],'model':s['model'],
            'groups':'+'.join(s['groups']),'features':s['features'],'metrics':aggregate([r['metrics'] for r in runs])})
    for crop,spec in manifest['crops'].items():
        data=frame(crop,'test')
        baselines={name:{'pooled':metrics(data,p),'per_truss':[
            {'truss':int(t),**metrics(g,p[data.index.get_indexer(g.index)])} for t,g in data.groupby('truss')]}
            for name,p in [('persistence',data.current.to_numpy()),('zero',np.zeros(len(data)))]}
        models=[]
        for model in MODELS:
            winner=next(w for w in frozen['winners'] if w['crop']==crop and w['model']==model)
            best=min([s for s in summaries if s['crop']==crop and s['model']==model],
                     key=lambda s:(s['rmse'],s['mae'],s['features'],s['combo']))
            assert winner==best
            runs=[r for r in tests if r['crop']==crop and r['model']==model]
            assert {r['seed'] for r in runs}==set(SEEDS[mode])
            assert all(r['combo']==winner['combo'] and r['best_epoch']==winner['epochs'] for r in runs)
            trusses=[]
            for t in range(1,spec['limit']+1):
                rows=[next(g for g in r['per_truss'] if g['truss']==t) for r in runs]
                trusses.append({'truss':t,'n':rows[0]['n'],'positive_n':rows[0]['positive_n'],
                                'metrics':aggregate(rows)})
            macro=[]
            totals=[]
            for r in runs:
                macro.append({k:float(np.mean([x[k] for x in r['per_truss'] if x[k] is not None]))
                              if any(x[k] is not None for x in r['per_truss']) else None for k in METRICS})
                p=pd.read_parquet(root/'runs'/r['run_id']/'predictions.parquet')
                total=total_metrics(p,spec['limit'])
                if total:totals.append(total)
            models.append({'model':model,'groups':'+'.join(winner['groups']),
                           'validation_rmse':winner['rmse'],'macro':aggregate(macro),
                           'pooled':aggregate([r['metrics'] for r in runs]),
                           'per_truss':trusses,
                           'sum':{'n':totals[0]['n'],'metrics':aggregate([r['metrics'] for r in totals])} if totals else None})
        selected=frozen['overall'][crop]
        assert selected==min([w for w in frozen['winners'] if w['crop']==crop],
            key=lambda w:(w['rmse'],w['mae'],w['features'],w['model']))['model']
        bundle['crops'][crop]={'selected':selected,'models':models,'baselines':baselines,
                               'counts':spec['counts'],'limit':spec['limit']}
    bundle['provenance']={n:file_hash(root/n) for n in ['settings.json','frozen_winners.json','progress.json']}
    atomic_json(root/'summary.json',bundle)
    atomic_json(root/'verification.json',{'status':'passed','audited_prediction_rows':audited_rows,
        'validation_runs':len(validation),'test_runs':len(tests),'summary_hash':file_hash(root/'summary.json')})
    pd.DataFrame([{'crop':r['crop'],'model':r['model'],'groups':r['groups'],'features':r['features'],
        **{k:r['metrics'][k]['mean'] for k in METRICS}} for r in bundle['comparison']]).to_csv(root/'validation_comparison.csv',index=False)
    print('VERIFIED',mode,audited_rows,flush=True)
    return bundle


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=list(SEEDS));export(p.parse_args().mode)
