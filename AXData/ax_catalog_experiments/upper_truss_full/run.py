"""Resumable 127-combination screening and three-seed temporal-CV campaigns."""
import argparse
import concurrent.futures as futures
import contextlib
import fcntl
from functools import lru_cache
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time
import traceback
import numpy as np
import pandas as pd
from AXData.ax_catalog_experiments.core import digest, file_hash
from AXData.ax_catalog_experiments.exhaustive.combinations import all_combinations
from AXData.ax_catalog_experiments.exhaustive.engine import grid, temporal_folds
from AXData.ax_catalog_experiments.exhaustive.io import atomic_json, frozen_json
from AXData.ax_catalog_experiments.exhaustive.models import EncodedAdapter
from AXData.ax_catalog_experiments.exhaustive.selection import select_features, SelectionResult
from AXData.ax_catalog_experiments.upper_truss_screening.run import PARAMS, MODELS, metrics
from .prepare import OUT, prepare

SEEDS = {'quick':[42], 'formal':[42,52,62]}
COMBINATIONS = all_combinations()


@lru_cache(maxsize=8)
def frame(crop, split):
    return pd.read_parquet(OUT/crop/f'{split}.parquet')


@lru_cache(maxsize=1)
def manifest():
    return json.loads((OUT/'manifest.json').read_text())


def selection(crop, combo, fit):
    spec = manifest()['crops'][crop]
    key = digest([crop,combo['Combination_ID'], sorted(fit.row_id.tolist())])[:24]
    path = OUT/'selection'/f'{key}.json';path.parent.mkdir(exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            data = json.loads(path.read_text());data.pop('policy',None)
            return SelectionResult(**data)
        features = sorted({f for g in combo['Groups'] for f in spec['groups'][g]})
        cat = [f for f in features if f in spec['categorical']]
        s = select_features(fit, [f for f in features if f not in cat],cat)
        # Target truss identity is common metadata, not an optional E1 history.
        s.selected_numeric.insert(0,'truss')
        atomic_json(path,s.as_dict())
        return s


def fit_run(mode,crop,combo,model,params,seed,stage,fit,evaluation,fold=None,epochs=None,frozen=None):
    root = OUT/mode
    identity = {'crop':crop,'combo':combo['Combination_ID'],'model':model,
                'params':params,'seed':seed,'stage':stage,'fold':fold,'epochs':epochs}
    key = digest(identity)[:24]; dest = root/'runs'/key;dest.mkdir(parents=True,exist_ok=True)
    result_path = dest/'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    atomic_json(dest/'running.json',{'identity':identity,'started':time.time()})
    start = time.time()
    record = {**identity,'run_id':key}
    with (dest/'log.txt').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            torch.set_num_threads(4)
            s = frozen or selection(crop,combo,fit)
            p = dict(params)
            if model=='random_forest':p['n_jobs']=4
            if model=='catboost':
                p.update(task_type='GPU',devices='0',thread_count=4,gpu_ram_part=.12)
            adapter = EncodedAdapter(model,s,p,seed,manifest()['crops'][crop]['entity'],
                dest/'checkpoint' if model in ['mlp','tabm','tft'] else None)
            adapter.fit(fit,'target',validation=evaluation if stage=='cv' or (mode=='quick' and stage=='validation') else None,
                        fixed_epochs=epochs)
            raw = adapter.predict(evaluation)
            assert np.isfinite(raw).all()
            pred = np.maximum(raw,0)
            record.update(status='success',best_epoch=adapter.best_epoch,
                          features=len(s.selected),selected=s.selected,metrics=metrics(evaluation,pred))
            if stage != 'cv':
                output = evaluation[['row_id','facility_id','crop_sn','sample_num','truss',
                                     'feature_date','target_date','current','target']].copy()
                output['prediction_raw']=raw;output['prediction']=pred
                output.to_parquet(dest/'predictions.parquet',index=False)
                record['per_truss']=[{'truss':int(t),**metrics(g,g.prediction.to_numpy())}
                                    for t,g in output.groupby('truss')]
            if stage=='test':
                adapter.save(dest/'model.bin')
            del adapter
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except Exception:
            record.update(status='failed',error=traceback.format_exc())
            traceback.print_exc()
    record['seconds']=time.time()-start
    atomic_json(result_path,record);(dest/'running.json').unlink(missing_ok=True)
    return record


def job(task):
    mode,crop,combo,model = task
    dest=OUT/mode/'summaries'/crop/model/f'{combo["Combination_ID"]}.json'
    if dest.exists():return json.loads(dest.read_text())
    train,val=frame(crop,'train'),frame(crop,'validation')
    base={'crop':crop,'model':model,'combo':combo['Combination_ID'],'groups':combo['Groups']}
    try:
        if mode=='quick':
            params=dict(PARAMS[model]);epochs=None
        else:
            choices=[]
            for params in grid(model):
                runs=[fit_run(mode,crop,combo,model,params,42,'cv',a,b,fold=n)
                      for n,a,b in temporal_folds(train)]
                if not all(r['status']=='success' for r in runs):
                    raise RuntimeError('CV candidate failed; inspect run logs; no silent candidate omission')
                es=[r['best_epoch'] for r in runs if r['best_epoch']]
                choices.append((np.mean([r['metrics']['rmse'] for r in runs]),
                                np.mean([r['metrics']['mae'] for r in runs]),
                                json.dumps(params,sort_keys=True),params,
                                int(statistics.median(es)) if es else None))
            winner=min(choices,key=lambda x:x[:3]);params,epochs=winner[3:]
        runs=[fit_run(mode,crop,combo,model,params,seed,'validation',train,val,epochs=epochs)
              for seed in SEEDS[mode]]
        assert all(r['status']=='success' for r in runs),'validation failure'
        if mode=='quick':epochs=runs[0]['best_epoch']
        base.update(status='success',params=params,epochs=epochs,run_ids=[r['run_id'] for r in runs],
            rmse=float(np.mean([r['metrics']['rmse'] for r in runs])),
            mae=float(np.mean([r['metrics']['mae'] for r in runs])),
            features=min(r['features'] for r in runs))
    except Exception:
        base.update(status='failed',error=traceback.format_exc())
    atomic_json(dest,base)
    return base


def test_job(task):
    mode,winner=task;crop,model=winner['crop'],winner['model']
    combo=next(c for c in COMBINATIONS if c['Combination_ID']==winner['combo'])
    train=frame(crop,'train');fit=pd.concat([train,frame(crop,'validation')],ignore_index=True)
    s=selection(crop,combo,train)
    return [fit_run(mode,crop,combo,model,winner['params'],seed,'test',fit,frame(crop,'test'),
                    epochs=winner['epochs'],frozen=s) for seed in SEEDS[mode]]


def run(mode):
    prepare()
    root=OUT/mode;root.mkdir(exist_ok=True)
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code=[*Path(__file__).parent.glob('*.py')]
        settings={'mode':mode,'seeds':SEEDS[mode],'models':MODELS,'combination_count':127,
                  'scope':{'tomato':20,'strawberry':5},'cv':'3 purged chronological folds, tuning seed 42' if mode=='formal' else 'none',
                  'grid':{m:grid(m) for m in MODELS} if mode=='formal' else PARAMS,
                  'selection':'Validation pooled RMSE, MAE, feature count, combination ID',
                  'common_input':'truss identity; E1 history optional',
                  'feature_selection':'fit-only missing/constant/correlation/VIF; protected truss',
                  'bounds':'nonnegative only; no upper cap','manifest_hash':file_hash(OUT/'manifest.json'),
                  'implementation':{p.name:file_hash(p) for p in code}}
        frozen_json(root/'settings.json',settings)
        context=mp.get_context('spawn')
        completed=[]
        # CPU and GPU pools keep long TFT jobs from starving CPU models.
        with futures.ProcessPoolExecutor(max_workers=1,mp_context=context) as cpu, \
             futures.ProcessPoolExecutor(max_workers=2,mp_context=context) as gpu:
            tasks={}
            for combo in COMBINATIONS:
                for crop in ['tomato','strawberry']:
                    for model in MODELS:
                        pool=cpu if model in ['poisson','random_forest'] else gpu
                        tasks[pool.submit(job,(mode,crop,combo,model))]=(crop,model,combo['Label'])
            for future in futures.as_completed(tasks):
                r=future.result();completed.append(r)
                atomic_json(root/'progress.json',{'phase':'validation','completed':len(completed),
                    'total':1524,'failed':sum(x['status']!='success' for x in completed),'last':tasks[future],
                    'updated':time.time()})
                print(mode,len(completed),'/1524',tasks[future],r['status'],flush=True)
        if any(r['status']!='success' for r in completed):
            raise RuntimeError('Incomplete comparison grid; publication blocked')
        winners=[]
        for crop in ['tomato','strawberry']:
            for model in MODELS:
                winners.append(min([r for r in completed if r['crop']==crop and r['model']==model],
                                   key=lambda r:(r['rmse'],r['mae'],r['features'],r['combo'])))
        overall={c:min([w for w in winners if w['crop']==c],
                       key=lambda w:(w['rmse'],w['mae'],w['features'],w['model']))['model']
                 for c in ['tomato','strawberry']}
        frozen_json(root/'frozen_winners.json',{'overall':overall,'winners':winners})
        results=[]
        with futures.ProcessPoolExecutor(max_workers=2,mp_context=context) as pool:
            for r in pool.map(test_job,[(mode,w) for w in winners]):
                results.extend(r)
                atomic_json(root/'progress.json',{'phase':'test','completed':len(results),
                    'total':12*len(SEEDS[mode]),'failed':sum(x['status']!='success' for x in results)})
        assert all(r['status']=='success' for r in results),'Test failure; publication blocked'
        atomic_json(root/'progress.json',{'phase':'complete','validation_combinations':1524,
                    'test_runs':len(results),'failed':0,'overall':overall,'completed_at':time.time()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['quick','formal'])
    run(parser.parse_args().mode)
