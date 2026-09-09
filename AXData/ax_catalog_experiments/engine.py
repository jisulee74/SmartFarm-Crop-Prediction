from __future__ import annotations
import json
import sys
import statistics
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .catalog import AX
from .core import digest,freeze,union,initial_groups,neighbours,file_hash
sys.path.insert(0,str(AX))
from common_regression.models import make_adapter
from common_regression.metrics import regression_metrics
from common_regression.temporal_cv import build_expanding_folds
from common_regression.selection import parameter_grid

MODELS=['poisson','random_forest','catboost','mlp','tabm','tft']
SEEDS=[42,52,62,72,82]
TFT_GRID=[{'hidden_size':h,'attention_head_size':2,'hidden_continuous_size':8,'dropout':.1,'learning_rate':.001,'batch_size':64,'max_epochs':100} for h in (16,32)]


class ExperimentEngine:
    def __init__(self,prepared,output,profile='full'):
        self.prepared,self.output=Path(prepared),Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.spec=json.loads((self.prepared/'manifest.json').read_text())
        hashes=self.prepared/'data_hashes.json'
        self.data_hashes=json.loads(hashes.read_text()) if hashes.exists() else {}
        for split in ('train','validation'):
            if self.data_hashes and file_hash(self.prepared/f'{split}.parquet')!=self.data_hashes[split]:raise ValueError(f'Prepared {split} data changed')
        self.train=pd.read_parquet(self.prepared/'train.parquet');self.val=pd.read_parquet(self.prepared/'validation.parquet')
        self.categories=self.spec['categories'];self.base=next(k for k,v in self.categories.items() if self.spec['current'] in v)
        self.profile=profile;self.cache={}
        self.config=yaml.safe_load((AX/'strawberry_fruit_set_prediction/config/model_suite.yaml').read_text())
        freeze(self.output/'execution.json',{'prepared_manifest':self.spec,'profile':profile,'engine_version':1,'seed_list':SEEDS,'config':self.config,'tft_grid':TFT_GRID})
        freeze(self.output/'runtime_limits.json',{'cpu_threads':4})
        if set(self.train.facility_id)&set(self.val.facility_id):raise ValueError('Facility overlap')
        self.eval_hash=digest(sorted(self.val.row_id.tolist()))
    def adapter(self,model,features,params,seed):
        params=dict(params)
        if model=='random_forest':params.setdefault('n_jobs',4)
        if model=='catboost':params.setdefault('thread_count',4)
        if model in {'mlp','tabm','tft'}:
            import torch
            torch.set_num_threads(4)
        if model=='tft':
            from .tft import TFTAdapter
            return TFTAdapter(features,params,seed,self.spec['entity'],self.spec['current'])
        return make_adapter(model,features,[],params,seed)
    def group_id(self,group):return 'VG_'+digest({'categories':sorted(group),'features':union(self.categories,group)})[:16]
    def evaluate(self,group,model,params,seed,stage,fit=None,evaluation=None,epochs=None,fold=None):
        features=union(self.categories,group)
        fit=self.train if fit is None else fit;evaluation=self.val if evaluation is None else evaluation
        # No evaluation-time feature filtering or row deletion.
        identity={'profile':self.profile,'dataset':self.spec['dataset_id'],'group':self.group_id(group),'model':model,'params':params,'seed':seed,'stage':stage,'fold':fold,'epochs':epochs,'fit_rows':digest(sorted(fit.row_id.tolist())),'eval_rows':digest(sorted(evaluation.row_id.tolist()))}
        rid='RUN_'+digest(identity)[:24];dest=self.output/'runs'/rid;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
        freeze(dest/'identity.json',identity)
        record={'Run_ID':rid,'Target_ID':self.spec['target_id'],'Variable_Group_ID':self.group_id(group),'Categories':sorted(group),'Features':features,'Feature_Count':len(features),'Model_Type':model,'Seed':seed,'Stage':stage,'Fold_ID':fold,'Eval_Row_Hash':identity['eval_rows'],'N_Eval':len(evaluation),'Params':params,'Status':'failed','Result_Origin':'new' if self.profile=='full' else 'smoke_not_rankable'}
        try:
            if set(features)-set(fit) or set(features)-set(evaluation):raise ValueError('Frozen member missing')
            if model=='persistence':pred=evaluation[self.spec['current']].to_numpy(float);best=None
            else:
                empty=[f for f in features if not pd.to_numeric(fit[f],errors='coerce').replace([np.inf,-np.inf],np.nan).notna().any()]
                if empty:raise ValueError(f'Frozen inputs entirely missing in fit fold: {empty}')
                adapter=self.adapter(model,features,params,seed)
                adapter.fit(fit,'target',validation=evaluation if stage=='internal_cv' else None,fixed_epochs=epochs)
                if hasattr(adapter,'preprocessor'):
                    if adapter.preprocessor.transform(fit.iloc[:1]).shape[1]!=len(features):raise ValueError('Preprocessor changed frozen input width')
                pred=np.maximum(0,adapter.predict(evaluation));best=adapter.best_epoch
                if stage=='test':adapter.save(dest/'model.bin')
            pred=np.maximum(0,pred)
            metric=regression_metrics(evaluation.target,pred)
            predictions=evaluation[['row_id','facility_id','feature_date','target_date','target']].copy()
            predictions['prediction']=pred;predictions['Run_ID']=rid;predictions['Eval_Split']=stage
            predictions.to_parquet(dest/'predictions.parquet',index=False)
            record.update(Status='success',Metrics=metric,Best_Epoch=best)
        except Exception as e:record.update(Error=f'{type(e).__name__}: {e}')
        freeze(dest/'result.json',record)
        print(f'{stage} {model} {rid} {record["Status"]}',flush=True)
        return record
    def screen(self,group):
        group=tuple(sorted(group))
        if group in self.cache:return self.cache[group]
        models=['poisson'] if self.profile=='smoke' else MODELS[:3]
        results=[self.evaluate(group,m,self.config['feature_selection_baseline_params'][m],42,'screening') for m in models]
        valid=[r for r in results if r['Status']=='success']
        score=min(((r['Metrics']['rmse'],r['Metrics']['mae'],r['Feature_Count'],r['Variable_Group_ID']) for r in valid),default=(float('inf'),)*3+('',))
        self.cache[group]=score;return score
    def tune(self,group,model):
        params_list=TFT_GRID if model=='tft' else parameter_grid(self.config['tuning_grid'][model])
        folds=build_expanding_folds(self.train,'feature_date',n_folds=3)
        choices=[]
        for params in params_list:
            trials=[]
            for fold,(a,b) in enumerate(folds,1):
                validation=self.train.iloc[b]
                fit=self.train.iloc[a]
                fit=fit[fit.target_date<validation.feature_date.min()]
                if fit.empty:raise ValueError('Purged temporal fold empty')
                trials.append(self.evaluate(group,model,params,42,'internal_cv',fit,validation,fold=fold))
            if all(r['Status']=='success' for r in trials):
                epochs=[r['Best_Epoch'] for r in trials if r['Best_Epoch'] is not None]
                choices.append((float(np.mean([r['Metrics']['rmse'] for r in trials])),float(np.mean([r['Metrics']['mae'] for r in trials])),json.dumps(params,sort_keys=True),params,int(statistics.median(epochs)) if epochs else None))
        if not choices:raise ValueError(f'No successful complete CV configuration for {model}')
        winner=min(choices,key=lambda x:x[:3]);return winner[3],winner[4]
    def run(self):
        groups=initial_groups(self.base,self.categories)
        if self.profile=='smoke':groups=[(self.base,)]
        for group in groups:self.screen(group)
        if self.profile=='smoke':
            freeze(self.output/'status.json',{'status':'smoke_complete','full_experiment_complete':False});return
        for _ in range(3):
            top=sorted((g for g in self.cache if self.base in g),key=lambda g:self.cache[g])[:3]
            todo=set().union(*(neighbours(g,self.categories,self.base) for g in top))-set(self.cache)
            if not todo:break
            for group in sorted(todo):self.screen(group)
        top=sorted((g for g in self.cache if self.base in g),key=lambda g:self.cache[g])[:5]
        finalists=sorted(set(top+[(self.base,)]));winners={};failures=[]
        for model in MODELS:
            candidates=[]
            for group in finalists:
                try:
                    params,epochs=self.tune(group,model)
                    runs=[self.evaluate(group,model,params,seed,'validation',epochs=epochs) for seed in SEEDS]
                    if not all(r['Status']=='success' for r in runs):raise ValueError('Incomplete validation seeds')
                    score=(float(np.mean([r['Metrics']['rmse'] for r in runs])),float(np.mean([r['Metrics']['mae'] for r in runs])),len(union(self.categories,group)),self.group_id(group))
                    candidates.append((score,{'group':list(group),'params':params,'epochs':epochs,'validation_rmse':score[0],'validation_mae':score[1]}))
                except Exception as e:failures.append({'model':model,'group':list(group),'reason':str(e)})
            if candidates:winners[model]=min(candidates,key=lambda x:x[0])[1]
        if not winners:raise ValueError('No models eligible for final test')
        overall=min(winners,key=lambda m:(winners[m]['validation_rmse'],winners[m]['validation_mae'],len(union(self.categories,winners[m]['group'])),self.group_id(winners[m]['group']),m))
        freeze(self.output/'selected.json',{'winners':winners,'overall':overall,'failures':failures,'selection_metric':'validation mean per-seed RMSE','test_accessed_during_selection':False})
        # The test file is not opened before the selected configuration is frozen.
        if self.data_hashes and file_hash(self.prepared/'test.parquet')!=self.data_hashes['test']:raise ValueError('Prepared test data changed')
        test=pd.read_parquet(self.prepared/'test.parquet')
        if set(test.facility_id)&(set(self.train.facility_id)|set(self.val.facility_id)):raise ValueError('Test facility overlap')
        combined=pd.concat([self.train,self.val],ignore_index=True)
        final=[]
        for model,w in winners.items():
            final.extend(self.evaluate(w['group'],model,w['params'],s,'test',combined,test,w['epochs']) for s in SEEDS)
        self.evaluate((self.base,),'persistence',{},0,'validation')
        final.append(self.evaluate((self.base,),'persistence',{},0,'test',combined,test))
        freeze(self.output/'status.json',{'status':'complete' if len(winners)==6 and all(r['Status']=='success' for r in final) else 'partial','full_experiment_complete':len(winners)==6 and all(r['Status']=='success' for r in final),'failures':failures})
