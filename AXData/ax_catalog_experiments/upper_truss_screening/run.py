"""One-seed, fixed-budget screening and frozen per-model Test evaluation."""
from pathlib import Path
import sys,os,json,time,traceback,concurrent.futures,multiprocessing,contextlib
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from AXData.ax_catalog_experiments.upper_truss_screening.prepare import OUT,prepare,digest
from AXData.ax_catalog_experiments.stability.policy import score
MODELS=['poisson','random_forest','catboost','mlp','tabm','tft']
PARAMS={
'poisson':{'alpha':1.0,'max_iter':400,'tol':1e-5},
'random_forest':{'n_estimators':200,'max_depth':16,'min_samples_leaf':5,'max_features':.8,'n_jobs':4},
'catboost':{'iterations':500,'depth':6,'learning_rate':.05,'loss_function':'RMSE','early_stopping_rounds':50,'task_type':'GPU','devices':'0','thread_count':4,'gpu_ram_part':.15},
'mlp':{'hidden':[128,64],'dropout':.1,'learning_rate':.001,'batch_size':1024,'max_epochs':60,'early_stopping_rounds':10},
 'tabm':{'k':8,'n_blocks':2,'dropout':.1,'learning_rate':.001,'batch_size':1024,'max_epochs':40,'early_stopping_rounds':10},
 'tft':{'hidden_size':16,'attention_head_size':2,'hidden_continuous_size':8,'dropout':.1,'learning_rate':.001,'batch_size':1024,'max_epochs':12}}
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def metrics(frame,p):
 result=score(frame.target,p);mask=frame.target.gt(0).to_numpy();changed=frame.target.ne(frame.current).to_numpy();result.update(n=len(frame),positive_n=int(mask.sum()),positive_mae=float(np.abs(frame.target.to_numpy()[mask]-p[mask]).mean()) if mask.any() else None,change_mae=float(np.abs(frame.target.to_numpy()[changed]-p[changed]).mean()) if changed.any() else None)
 return result
def job(task):
 crop,config,model,stage,epochs=task;key=f'{crop}__{config}__{model}__{stage}';path=OUT/'runs'/f'{key}.json'
 if path.exists():return json.loads(path.read_text())
 log=OUT/'logs'/f'{key}.log';log.parent.mkdir(exist_ok=True);started=time.time()
 with log.open('w') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
  try:
   import torch
   torch.set_num_threads(4)
   from AXData.ax_catalog_experiments.stability.adapters import adapter
   manifest=json.loads((OUT/'manifest.json').read_text());spec=manifest['crops'][crop];features=spec['features'][config];train=pd.read_parquet(OUT/crop/'train.parquet');val=pd.read_parquet(OUT/crop/'validation.parquet')
   if stage=='test':train=pd.concat([train,val],ignore_index=True);evaluation=pd.read_parquet(OUT/crop/'test.parquet');fit_validation=None
   else:evaluation=val;fit_validation=val
   a=adapter(model,features,PARAMS[model],42,spec['entity'],'current');a.fit(train,'target',validation=fit_validation,fixed_epochs=epochs if stage=='test' else None)
   raw=a.predict(evaluation);assert np.isfinite(raw).all();pred=np.maximum(raw,0)
   rec={'crop':crop,'config':config,'model':model,'stage':stage,'seed':42,'status':'success','best_epoch':a.best_epoch,'features':len(features),'seconds':time.time()-started,'metrics':metrics(evaluation,pred),'raw_metrics':metrics(evaluation,raw),'params':PARAMS[model]}
   keep=['row_id','facility_id','crop_sn','sample_num','truss','feature_date','target_date','current','target'];p=evaluation[keep].copy();p['prediction_raw']=raw;p['prediction']=pred;p.to_parquet(OUT/'predictions'/f'{key}.parquet',index=False)
   per=[]
   for t,g in p.groupby('truss'):per.append({'truss':int(t),**metrics(g,g.prediction.to_numpy())})
   rec['per_truss']=per
   if stage=='test':a.save(OUT/'models'/f'{key}.model')
  except Exception as e:rec={'crop':crop,'config':config,'model':model,'stage':stage,'status':'failed','error':str(e),'traceback':traceback.format_exc(),'seconds':time.time()-started};traceback.print_exc()
 write(path,rec);return rec
def execute(tasks,stage):
 records=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  fs={pool.submit(job,t):t for t in tasks}
  for future in concurrent.futures.as_completed(fs):
   rec=future.result();records.append(rec);write(OUT/'progress.json',{'phase':stage,'completed':len(records),'total':len(tasks),'failed':sum(r['status']!='success' for r in records),'last':{k:rec.get(k) for k in ['crop','config','model','status','seconds']}});print(stage,len(records),'/',len(tasks),rec['crop'],rec['config'],rec['model'],rec['status'],round(rec['seconds'],1),flush=True)
 return records
def main():
 manifest=prepare()
 for folder in ['runs','predictions','logs','models']:(OUT/folder).mkdir(exist_ok=True)
 settings={'params':PARAMS,'seed':42,'workers':4,'selection':'min validation pooled RMSE, then MAE, then input count/config; per-model and overall winners frozen before Test','test_refit':'train+validation, train-frozen feature set, selected best epoch','caps':'nonnegative only, no upper bound'}
 if (OUT/'settings.json').exists():assert json.loads((OUT/'settings.json').read_text())==settings
 else:write(OUT/'settings.json',settings)
 tasks=[(crop,cfg,model,'validation',None) for crop,spec in manifest['crops'].items() for cfg in spec['features'] for model in MODELS]
 records=execute(tasks,'validation');winners=[]
 for crop in manifest['crops']:
  for model in MODELS:
   candidates=[r for r in records if r['crop']==crop and r['model']==model and r['status']=='success']
   if not candidates:continue
   winner=min(candidates,key=lambda r:(r['metrics']['rmse'],r['metrics']['mae'],r['features'],r['config']));winners.append(winner)
 overall={crop:min([w for w in winners if w['crop']==crop],key=lambda r:(r['metrics']['rmse'],r['metrics']['mae'],r['features'],r['model']))['model'] for crop in manifest['crops']}
 write(OUT/'frozen_winners.json',{'selected_on':'validation','overall_model':overall,'per_model':winners})
 tests=execute([(r['crop'],r['config'],r['model'],'test',r['best_epoch']) for r in winners],'test')
 baselines=[]
 for crop in manifest['crops']:
  train=pd.read_parquet(OUT/crop/'train.parquet')
  for split in ['validation','test']:
   fit=train if split=='validation' else pd.concat([train,pd.read_parquet(OUT/crop/'validation.parquet')]);d=pd.read_parquet(OUT/crop/(split+'.parquet'))
   for name,pred in [('persistence',d.current.to_numpy()),('zero',np.zeros(len(d))),('train_mean',np.full(len(d),fit.target.mean()))]:
    baselines.append({'crop':crop,'stage':split,'model':name,**metrics(d,pred),'per_truss':[{'truss':int(t),**metrics(g,pred[d.index.get_indexer(g.index)])} for t,g in d.groupby('truss')]})
 write(OUT/'baselines.json',baselines)
 rows=[];trusses=[]
 for r in records+tests:
  if r['status']=='success':
   rows.append({k:r[k] for k in ['crop','config','model','stage','features','best_epoch','seconds']}|r['metrics'])
   trusses.extend([{'crop':r['crop'],'config':r['config'],'model':r['model'],'stage':r['stage'],**v} for v in r['per_truss']])
 pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(trusses).to_csv(OUT/'per_truss_metrics.csv',index=False);write(OUT/'progress.json',{'phase':'complete','validation_runs':len(records),'test_runs':len(tests),'failed':sum(r['status']!='success' for r in records+tests),'overall_models':overall});print('COMPLETE',overall,flush=True)
if __name__=='__main__':main()
