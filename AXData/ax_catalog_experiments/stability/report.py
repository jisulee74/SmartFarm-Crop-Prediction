from __future__ import annotations
import json,time
from pathlib import Path
import pandas as pd
import numpy as np
from ..report import write_xlsx
from .data import TARGETS
from .engine import MODELS,SEEDS,grid,atomic
from .policy import score

def progress(root):
    root=Path(root);runs=[json.loads(p.read_text()) for p in (root/'experiments').glob('*/runs/*/result.json')]
    done=len(runs);counts=pd.Series([r['Status'] for r in runs],dtype=str).value_counts().to_dict()
    expected=0
    for p in (root/'prepared').glob('*/manifest.json'):
        spec=json.loads(p.read_text());expected+=len(spec['groups'])*sum(3*len(grid(m))+len(SEEDS) for m in MODELS)+len(MODELS)*len(SEEDS)+2
    unavailable=pd.read_csv(root/'Group_Mapping.csv').query("Status == 'unavailable'").shape[0] if (root/'Group_Mapping.csv').exists() else 0
    excluded=0
    for p in (root/'experiments').glob('*/ineligible/*.json'):
        r=json.loads(p.read_text());evaluated=sum(x['Target_ID']==r['Target_ID'] and x['Variable_Group_ID']==r['Variable_Group_ID'] and x['Model_Type']==r['Model_Type'] and x['Stage']=='validation' for x in runs)
        excluded+=max(0,len(SEEDS)-evaluated)
    pending=max(0,expected-done-excluded)
    durations=[r['Duration_Seconds'] for r in runs if r.get('Duration_Seconds')]
    # Model/stage stratification avoids treating an inexpensive Poisson fit as a TFT estimate.
    represented={(r['Model_Type'],r['Stage']) for r in runs}
    eta=float(np.mean(durations)*pending) if durations and all((m,s) in represented for m in MODELS for s in ['internal_cv','validation']) else None
    return dict(Completed_Runs=done,Status_Counts=counts,Unavailable_Group_Mappings=unavailable,Excluded_Validation_Runs=excluded,
      Planned_Run_Upper_Bound=expected,Pending_Run_Upper_Bound=pending,Estimated_Remaining_Seconds=eta,
      ETA_Note='rough observed runtime estimate; unavailable until all model/CV+validation strata have observations',
      Last_Completed_At=max((r['Completed_At'] for r in runs),default=None),
      Heartbeat=json.loads((root/'heartbeat.json').read_text()) if (root/'heartbeat.json').exists() else None)

def sums(root):
    records=[];keys=['facility_id','crop_sn','sample_num','feature_date','target_date']
    index={}
    for p in (root/'experiments').glob('*/runs/*/result.json'):
        r=json.loads(p.read_text())
        if r['Status']=='success' and r['Stage'] in ['validation','test']:
            index.setdefault((r['Target_ID'],r['Model_Type'],r['Variable_Group_ID'],r['Stage'],r['Seed']),[]).append(p)
    selections={p.parent.name:json.loads(p.read_text()) for p in (root/'experiments').glob('*/selection_frozen.json')}
    for crop in ['tomato','strawberry']:
        for model in MODELS:
            for split in ['validation','test']:
                for seed in SEEDS:
                    tables=[]
                    for part in ['first','second','third','sum123']:
                        target=f'{crop}_{part}'
                        if target not in selections:break
                        win=selections[target]['winners'].get(model)
                        if not win:break
                        candidates=index.get((target,model,win['group'],split,seed),[])
                        if len(candidates)!=1:break
                        d=pd.read_parquet(candidates[0].parent/'predictions.parquet')
                        tables.append(d[keys+['target','prediction_raw','prediction_bounded']].rename(columns={c:part+'_'+c for c in ['target','prediction_raw','prediction_bounded']}))
                    if len(tables)!=4:continue
                    merged=tables[0]
                    for t in tables[1:]:merged=merged.merge(t,on=keys,validate='one_to_one')
                    if merged.empty:continue
                    actual=sum(merged[p+'_target'] for p in ['first','second','third'])
                    if not np.allclose(actual,merged.sum123_target):raise ValueError('Sum target observation mismatch')
                    for variant in ['raw','bounded']:
                        for method in ['direct_sum_model','sum_of_truss_predictions']:
                            pred=merged['sum123_prediction_'+variant] if method=='direct_sum_model' else sum(merged[p+'_prediction_'+variant] for p in ['first','second','third'])
                            for metric,value in score(actual,pred).items():records.append(dict(Crop=crop,Model_Type=model,Seed=seed,Eval_Split=split,Prediction_Variant=variant,Method=method,N_Eval=len(merged),Metric_Name=metric.upper(),Metric_Value=value))
    return pd.DataFrame(records)

def report(root, include_sums=True):
    root=Path(root);tables={};runs=[];results=[];best=[]
    for p in (root/'experiments').glob('*/runs/*/result.json'):
        r=json.loads(p.read_text());runs.append({k:v if not isinstance(v,(dict,list)) else json.dumps(v,ensure_ascii=False) for k,v in r.items() if k!='Metrics'})
        for variant,metrics in r.get('Metrics',{}).items():
            for metric,value in metrics.items():results.append({**{k:r.get(k) for k in ['Run_ID','Target_ID','Variable_Group_ID','Model_Type','Seed','N_Eval','Status','Eval_Row_Hash']},'Eval_Split':r['Stage'],'Fold_ID':r.get('Fold_ID'),'Prediction_Variant':variant,'Metric_Name':metric.upper(),'Metric_Value':value,'Metric_Status':'available' if value is not None else 'undefined'})
    for p in (root/'experiments').glob('*/selection_frozen.json'):
        r=json.loads(p.read_text())
        for model,w in r['winners'].items():best.append(dict(Target_ID=p.parent.name,Model_Type=model,Variable_Group_ID=w['group'],Validation_RMSE=w['validation_rmse'],Validation_MAE=w['validation_mae'],Overall_Selected=model==r['overall']))
    tables['README']=pd.DataFrame([dict(Item='Scope',Value='8 next-observation targets; sum means trusses 1+2+3 only'),dict(Item='Metrics',Value='Separate raw/bounded, stage, target and seed; never sum metrics'),dict(Item='Test',Value='Previously inspected holdout; not a new independent test'),dict(Item='Campaign',Value=json.dumps(progress(root),ensure_ascii=False))])
    for name in ['Group_Mapping','Group_Detail','Data_Quality','Target_Status']:
        if (root/f'{name}.csv').exists():tables[name]=pd.read_csv(root/f'{name}.csv')
    experiments=[]
    pairs=[]
    for p in sorted((root/'prepared').glob('*/manifest.json')):
        spec=json.loads(p.read_text())
        for gid in sorted([*spec['groups'],'PERSISTENCE']):pairs.append((spec['target_id'],gid))
    ids={pair:f'EXP_V2_{i:03d}' for i,pair in enumerate(pairs,1)}
    labels={'first':'1화방','second':'2화방','third':'3화방','sum123':'1·2·3화방 합계'}
    for (target,gid),eid in ids.items():
        crop,part=TARGETS[target]
        experiments.append(dict(Experiment_ID=eid,Target_ID=target,Variable_Group_ID=gid,
          Crop_Target='토마토' if crop=='tomato' else '딸기',
          Prediction_Target=labels[part]+(' 꽃 수' if crop=='tomato' else ' 착과수')))
    lookup={(r['Target_ID'],r['Variable_Group_ID']):r for r in experiments}
    for rows in [runs,results,best]:
        for r in rows:r.update(lookup.get((r['Target_ID'],r['Variable_Group_ID']),{}))
    tables['Experiments']=pd.DataFrame(experiments)
    tables['Target_Catalog']=tables['Experiments'][['Target_ID','Crop_Target','Prediction_Target']].drop_duplicates().assign(
      Horizon='same individual next recorded survey',Sum_Rule='sum123 requires all 3 observed truss counts',
      Missing_Rule='missing truss is not zero; audit source-cohort exclusions',Version=2)
    from .data import SOURCE
    original=pd.read_csv(SOURCE/'csv/Group_Detail.csv').rename(columns={'Variable_ID':'Old_Variable_ID','Category_ID':'Old_Category_ID'})
    meta=['Old_Variable_ID','Old_Category_ID','Variable_Name','Source_Table','Source_Field_or_Code','Derivation','Window','Unit','Reference_Time']
    meta=[c for c in meta if c in original]
    original=original[meta].drop_duplicates(['Old_Variable_ID','Old_Category_ID'])
    if 'Group_Detail' in tables:
        details=tables['Group_Detail'].merge(original,on=['Old_Variable_ID','Old_Category_ID'],how='left',validate='many_to_one')
        tables['Group_Detail']=details
        tables['Variable_Catalog']=details.drop(columns=['Variable_Group_ID']).drop_duplicates()
    tables['Source_Inventory']=pd.read_csv(SOURCE/'csv/Source_Inventory.csv')
    tables['Runs']=pd.DataFrame(runs);tables['Results']=pd.DataFrame(results,columns=[
      'Experiment_ID','Run_ID','Target_ID','Crop_Target','Prediction_Target','Variable_Group_ID',
      'Model_Type','Seed','Eval_Split','Fold_ID','Prediction_Variant','Metric_Name','Metric_Value',
      'Metric_Status','Status','N_Eval','Eval_Row_Hash'])
    tables['Best_Configurations']=pd.DataFrame(best)
    if results:
        d=tables['Results'];d=d[(d.Status=='success')&d.Eval_Split.isin(['validation','test'])]
        tables['Results_Summary']=d.groupby(['Target_ID','Variable_Group_ID','Model_Type','Eval_Split','Prediction_Variant','Metric_Name','Eval_Row_Hash'],dropna=False).Metric_Value.agg(Mean='mean',Std='std',Seed_Count='count').reset_index()
    diagnostic=root/'diagnostics/comparison.json'
    if diagnostic.exists():
        dr=[]
        for r in json.loads(diagnostic.read_text()):
            for variant,key in [('raw','Raw_Metrics'),('bounded','Bounded_Metrics')]:
                for metric,value in r.get(key,{}).items():dr.append(dict(Target_ID=r['Target_ID'],Model_Type=r['Model_Type'],Implementation=r['Variant'],Prediction_Variant=variant,Metric_Name=metric.upper(),Metric_Value=value,Status=r['Status']))
        tables['Diagnostic_Comparison']=pd.DataFrame(dr)
    tables['Sum_Comparisons']=sums(root) if include_sums else pd.DataFrame()
    tables['Exclusions']=pd.DataFrame([json.loads(p.read_text()) for p in (root/'experiments').glob('*/ineligible/*.json')])
    dest=root/'csv';dest.mkdir(exist_ok=True)
    for name,df in tables.items():df.to_csv(dest/f'{name}.csv',index=False,encoding='utf-8-sig')
    write_xlsx(root/'AX_Experiment_Report.xlsx',tables)
    atomic(root/'progress.json',progress(root))
    (root/'Progress.md').write_text('# 안정화 재실험 상태\n\n'+('진단을 통과한 실행만 최종 선택합니다.\n\n')+json.dumps(progress(root),ensure_ascii=False,indent=2)+'\n\n기존 테스트는 이미 확인된 holdout이며 새 독립 검증이 아닙니다.\n')
    return progress(root)
