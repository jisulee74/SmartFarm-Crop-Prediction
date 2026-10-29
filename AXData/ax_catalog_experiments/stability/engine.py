from __future__ import annotations
import json,time,warnings,statistics,os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from ..core import digest,freeze,file_hash
from ..engine import MODELS,SEEDS,TFT_GRID
from ...common_regression.temporal_cv import build_expanding_folds
from ...common_regression.selection import parameter_grid
from .adapters import adapter
from .policy import POLICY,upper_bound,constrain,score,StablePreprocessor
from .data import AX,TARGETS


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str));tmp.replace(path)

def config():return yaml.safe_load((AX/'strawberry_fruit_set_prediction/config/model_suite.yaml').read_text())
def grid(model):
    values=TFT_GRID if model=='tft' else parameter_grid(config()['tuning_grid'][model])
    if model=='poisson':
        unique={}
        for params in values:
            params=dict(params);params['alpha']=max(float(params['alpha']),POLICY['poisson_alpha_floor'])
            unique[json.dumps(params,sort_keys=True)]=params
        values=list(unique.values())
    return values

def implementation_hash():
    paths=[Path(__file__).parent/n for n in ['policy.py','adapters.py','tft.py','data.py','engine.py']]
    paths.append(AX/'common_regression/models.py')
    return digest({str(p.relative_to(AX)):file_hash(p) for p in paths})
def folds(train):
    for number,(a,b) in enumerate(build_expanding_folds(train,'feature_date',n_folds=3),1):
        val=train.iloc[b];fit=train.iloc[a];fit=fit[fit.target_date<val.feature_date.min()]
        if fit.empty:raise ValueError('Purged temporal fold empty')
        yield number,fit,val

class Campaign:
    def __init__(self,root,target):
        self.root=Path(root);self.target=target;self.prepared=self.root/'prepared'/target
        self.spec=json.loads((self.prepared/'manifest.json').read_text());self.out=self.root/'experiments'/target
        self.out.mkdir(parents=True,exist_ok=True)
        self.train=self.load('train');self.val=self.load('validation')
        if set(self.train.facility_id)&set(self.val.facility_id):raise ValueError('Facility overlap')
        self.policy_hash=digest(POLICY)
    def load(self,split):
        p=self.prepared/f'{split}.parquet'
        if file_hash(p)!=self.spec['dataset_hashes'][split]:raise ValueError('Prepared data changed')
        return pd.read_parquet(p)
    def evaluate(self,gid,model,params,seed,stage,fit=None,evaluation=None,epochs=None,fold=None):
        fit=self.train if fit is None else fit;evaluation=self.val if evaluation is None else evaluation
        features=[self.spec['current']] if model=='persistence' else self.spec['groups'][gid]['features']
        if model!='persistence':
            union=sorted({f for fs in self.spec['groups'][gid]['categories'].values() for f in fs})
            if union!=features:raise ValueError('Atomic category union mismatch')
        identity=dict(target=self.target,group=gid,model=model,params=params,seed=seed,stage=stage,fold=fold,epochs=epochs,
          fit_hash=digest(sorted(fit.row_id.tolist())),eval_hash=digest(sorted(evaluation.row_id.tolist())),
          data=self.spec['dataset_hashes'],policy=self.policy_hash)
        rid='RUN_V2_'+digest(identity)[:24];dest=self.out/'runs'/rid
        if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
        dest.mkdir(parents=True,exist_ok=True);freeze(dest/'identity.json',identity)
        record=dict(Run_ID=rid,Target_ID=self.target,Variable_Group_ID=gid,Model_Type=model,Seed=seed,Stage=stage,Fold_ID=fold,
          Eval_Row_Hash=identity['eval_hash'],Feature_Count=len(features),N_Eval=len(evaluation),Params=params,Status='running',Started_At=time.time())
        atomic(dest/'running.json',record);atomic(self.root/'heartbeat.json',dict(**record,PID=os.getpid()))
        start=time.monotonic()
        try:
            upper=upper_bound(fit,self.spec['components'])
            prep=StablePreprocessor(features).fit(fit)
            prep.diagnostics(evaluation).to_csv(dest/'input_diagnostics.csv',index=False)
            if model=='persistence': raw=evaluation[self.spec['current']].to_numpy(float);best=None;caught=[]
            else:
                pp=dict(params)
                if model=='random_forest':pp.setdefault('n_jobs',4)
                if model=='catboost':pp.setdefault('thread_count',4)
                if model in ['mlp','tabm','tft']:
                    import torch
                    torch.set_num_threads(4)
                a=adapter(model,features,pp,seed,self.spec['entity'],self.spec['current'])
                record['Effective_Params']=a.params
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    a.fit(fit,'target',validation=evaluation if stage=='internal_cv' else None,fixed_epochs=epochs)
                    raw=np.asarray(a.predict(evaluation),float)
                best=a.best_epoch
                if a.preprocessor.transform(fit.iloc[:1]).shape[1]!=len(features):raise ValueError('Input width changed')
            if not np.isfinite(raw).all():
                evidence=evaluation[[*self.spec['entity'],'row_id','feature_date','target_date','target']].copy()
                evidence['prediction_raw']=raw;evidence.to_parquet(dest/'failed_predictions.parquet',index=False)
            bounded,unstable=constrain(raw,upper)
            nonconverged=any(issubclass(w.category,ConvergenceWarning) for w in caught)
            record.update(Status='unstable' if unstable else 'nonconverged' if nonconverged else 'success',
              Upper_Bound=upper,Raw_Abs_Max=float(np.abs(raw).max()),Output_Clip_Rate=float(np.mean(raw!=bounded)),
              Best_Epoch=best,Warnings=[str(w.message) for w in caught],
              Metrics={'raw':score(evaluation.target,raw),'bounded':score(evaluation.target,bounded)})
            predictions=evaluation[[*self.spec['entity'],'row_id','feature_date','target_date','target']].copy()
            predictions=predictions.assign(prediction_raw=raw,prediction_bounded=bounded,upper_bound=upper,
              was_constrained=raw!=bounded,Run_ID=rid,Eval_Split=stage)
            predictions.to_parquet(dest/'predictions.parquet',index=False)
            if stage=='test' and model!='persistence' and record['Status']=='success':a.save(dest/'model.bin')
        except Exception as e:record.update(Status='failed',Reason=f'{type(e).__name__}: {e}')
        record.update(Duration_Seconds=time.monotonic()-start,Completed_At=time.time())
        atomic(dest/'result.json',record);(dest/'running.json').unlink(missing_ok=True)
        print(self.target,stage,model,rid,record['Status'],flush=True)
        return record
    def tune(self,gid,model):
        choices=[]
        for params in grid(model):
            trials=[self.evaluate(gid,model,params,42,'internal_cv',fit,val,fold=number) for number,fit,val in folds(self.train)]
            if all(t['Status']=='success' for t in trials):
                epochs=[t['Best_Epoch'] for t in trials if t.get('Best_Epoch')]
                choices.append((np.mean([t['Metrics']['bounded']['rmse'] for t in trials]),
                  np.mean([t['Metrics']['bounded']['mae'] for t in trials]),json.dumps(params,sort_keys=True),params,
                  int(statistics.median(epochs)) if epochs else None))
        if not choices:raise ValueError('No stable converged complete CV configuration')
        best=min(choices,key=lambda x:x[:3]);return best[3],best[4]
    def run(self):
        winners={};failures=[]
        # Every available group is tuned for EVERY model; no staged group pruning.
        for model in MODELS:
            candidates=[]
            for gid in sorted(self.spec['groups']):
                selection_path=self.out/'tuning'/f'{gid}_{model}.json'
                try:
                    if selection_path.exists():chosen=json.loads(selection_path.read_text())
                    else:
                        params,epochs=self.tune(gid,model);chosen=dict(params=params,epochs=epochs)
                        freeze(selection_path,chosen)
                    trials=[self.evaluate(gid,model,chosen['params'],seed,'validation',epochs=chosen['epochs']) for seed in SEEDS]
                    if not all(t['Status']=='success' for t in trials):raise ValueError('Incomplete stable validation seeds')
                    rmse=float(np.mean([t['Metrics']['bounded']['rmse'] for t in trials]));mae=float(np.mean([t['Metrics']['bounded']['mae'] for t in trials]))
                    candidates.append(((rmse,mae,len(self.spec['groups'][gid]['features']),gid),dict(group=gid,**chosen,validation_rmse=rmse,validation_mae=mae)))
                except Exception as e:
                    failure=dict(Target_ID=self.target,Variable_Group_ID=gid,Model_Type=model,Status='ineligible',Reason=f'{type(e).__name__}: {e}')
                    failures.append(failure);atomic(self.out/'ineligible'/f'{gid}_{model}.json',failure)
                atomic(self.out/'progress.json',dict(target=self.target,model=model,last_group=gid,failures=failures))
            if candidates:winners[model]=min(candidates,key=lambda x:x[0])[1]
            from .report import report
            report(self.root,include_sums=False)
        baseline=self.evaluate('PERSISTENCE','persistence',{},42,'validation')
        if not winners:
            atomic(self.out/'status.json',dict(status='blocked',reason='No eligible model',failures=failures));return
        overall=min(winners,key=lambda m:(winners[m]['validation_rmse'],winners[m]['validation_mae'],len(self.spec['groups'][winners[m]['group']]['features']),winners[m]['group'],m))
        frozen=dict(winners=winners,overall=overall,policy_hash=self.policy_hash,dataset_hashes=self.spec['dataset_hashes'])
        freeze(self.out/'selection_frozen.json',frozen)
        # Test data are loaded only after all group/model validation selections are frozen.
        test=self.load('test')
        if set(test.facility_id)&(set(self.train.facility_id)|set(self.val.facility_id)):raise ValueError('Test facility overlap')
        tests=[]
        for model,w in winners.items():
            for seed in SEEDS:tests.append(self.evaluate(w['group'],model,w['params'],seed,'test',evaluation=test,epochs=w['epochs']))
        tests.append(self.evaluate('PERSISTENCE','persistence',{},42,'test',evaluation=test))
        atomic(self.out/'status.json',dict(status='complete' if len(winners)==len(MODELS) and not failures and all(r['Status']=='success' for r in tests) else 'complete_with_exclusions',failures=failures,test_failures=[r['Run_ID'] for r in tests if r['Status']!='success'],baseline_status=baseline['Status']))

def run_all(root):
    root=Path(root)
    gate=json.loads((root/'diagnostics/gate.json').read_text())
    if gate.get('Status')!='passed' or gate.get('Policy_Hash')!=digest(POLICY):raise ValueError('Diagnostic gate has not passed current policy')
    freeze(root/'policy.json',POLICY)
    freeze(root/'execution_config.json',dict(models=MODELS,seeds=SEEDS,config=config(),tft_grid=TFT_GRID,implementation_hash=implementation_hash(),effective_poisson_grid=grid('poisson'),prepared_manifest_hashes={t:file_hash(root/'prepared'/t/'manifest.json') for t in TARGETS}))
    import fcntl
    with (root/'campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=[]
        for target in TARGETS:
            try:
                Campaign(root,target).run()
                record=dict(Target_ID=target,**json.loads((root/'experiments'/target/'status.json').read_text()))
            except Exception as e:record=dict(Target_ID=target,status='blocked',reason=f'{type(e).__name__}: {e}')
            campaign.append(record);atomic(root/'campaign_status.json',campaign)
            from .report import report
            report(root)
        atomic(root/'heartbeat.json',dict(Status='finished',Completed_At=time.time()))
    return campaign
