"""Read-only analysis of the frozen stability_v2 experiment. No fitting or source writes.
Run from workspace root: envs/jslee_py311/bin/python AXData/ax_catalog_experiments/analysis/20260914_ax_direction/analyze.py
"""
from pathlib import Path
import hashlib, json, itertools, datetime, subprocess
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'outputs' / 'stability_v2'
AX = HERE.parents[2]
OUT = HERE / 'tables'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = {42, 52, 62, 72, 82}
METRICS = ['rmse', 'mae', 'r2', 'ccc']
TARGETS = [f'{c}_{p}' for c in ['tomato', 'strawberry'] for p in ['first', 'second', 'third', 'sum123']]
inputs, checks = {}, []

def track(p):
    p = Path(p)
    inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def readjson(p):
    return json.loads(track(p).read_text())

def check(name, condition, detail=''):
    checks.append(dict(check=name, passed=bool(condition), detail=str(detail)))
    if not condition:
        raise AssertionError(f'{name}: {detail}')

def save(name, rows):
    d = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    return d

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p-y
    den = y.var()+p.var()+(y.mean()-p.mean())**2
    return dict(rmse=np.sqrt(np.mean(e**2)), mae=np.mean(abs(e)),
                r2=1-np.sum(e**2)/np.sum((y-y.mean())**2) if len(y)>1 and np.ptp(y)>0 else np.nan,
                ccc=2*np.mean((y-y.mean())*(p-p.mean()))/den if den else 1.,
                bias=e.mean(), p90_ae=np.quantile(abs(e), .9), within1=np.mean(abs(e)<=1))

def describe(d, current):
    horizon = (d.target_date-d.feature_date).dt.total_seconds()/86400
    delta = d.target-d[current]
    return dict(n=len(d), facilities=d.facility_id.nunique(),
                crop_cycles=len(d[['facility_id','crop_sn']].drop_duplicates()),
                entities=len(d[['facility_id','crop_sn','sample_num']].drop_duplicates()),
                feature_start=str(d.feature_date.min().date()), target_end=str(d.target_date.max().date()),
                zero_rate=d.target.eq(0).mean(), changed_rate=delta.ne(0).mean(),
                large_change_rate=delta.abs().ge(2).mean(), positive_n=int(d.target.gt(0).sum()),
                target_mean=d.target.mean(), target_std=d.target.std(), target_max=d.target.max(),
                horizon_median=horizon.median(), horizon_max=horizon.max(),
                previous_gap_median=d.days_since_previous.median())

def bins(x, edges, labels):
    return pd.cut(x, edges, labels=labels, include_lowest=True).astype('string').fillna('missing')

def category_label(fs):
    s='|'.join(fs)
    if all(f.startswith('legacy__') for f in fs): return '기존 복합 입력(legacy)'
    if any(f.startswith('flower_count__') for f in fs): return '꽃수 관측 이력'
    if any(f.startswith('fruit_count__') for f in fs): return '착과수 관측 이력'
    if any(f.startswith('current_max_truss') for f in fs): return '화방 정보'
    if any(f.startswith('observed_truss') for f in fs): return '관측 화방 범위'
    if 'EI__CI' in s: return '실내 CO2'
    if 'EI__HI' in s or 'EI__TI' in s: return '실내 온습도'
    if fs == ['history_count']: return '조사 횟수'
    if 'days_since' in s: return '시간 정보'
    return '기타 정의 확인'

def main():
    print('Loading campaign metadata', flush=True)
    scheduler=readjson(ROOT/'parallel/scheduler.json')
    progress=readjson(ROOT/'progress.json')
    cfg=readjson(ROOT/'execution_config.json')
    readjson(ROOT/'policy.json')
    runs=pd.read_csv(track(ROOT/'csv/Runs.csv'))
    summary=pd.read_csv(track(ROOT/'csv/Results_Summary.csv'))
    check('scheduler_finished',scheduler['Status']=='finished')
    check('all_runs_successful',len(runs)==39856 and runs.Status.eq('success').all())
    check('no_duplicate_run_ids',not runs.Run_ID.duplicated().any())
    check('progress_matches',len(runs)==progress['Completed_Runs'])
    for f in ['engine.py','policy.py','data.py','adapters.py','tft.py']:
        track(ROOT.parents[1]/'stability'/f)
    for fname in ['회의록_AX데이터구축_0826.pdf','회의록_AX데이터구축_0909.pdf']:
        p=track(AX/fname)
        (HERE/'sources').mkdir(exist_ok=True)
        subprocess.run(['pdftotext','-layout',str(p),str(HERE/'sources'/p.with_suffix('.txt').name)],check=True)
    cohorts=[]; baseline=[]; strata=[]; sensitivity=[]; pred_export=[]; cat_export=[]; coverage=[]
    selected_meta=[]; paired=[]; input_diag=[]; exclusions=[]; all_model=[]; error_cases=[]
    metric_checks=0
    for target in TARGETS:
        print('Analyzing',target,flush=True)
        spec=readjson(ROOT/'prepared'/target/'manifest.json')
        selected=readjson(ROOT/'experiments'/target/'selection_frozen.json')
        status=readjson(ROOT/'experiments'/target/'status.json')
        check(target+'_complete',status['status']=='complete' and not status['failures'] and not status['test_failures'])
        actual_paths=list((ROOT/'experiments'/target/'runs').glob('*/result.json'))
        rr=runs[runs.Target_ID.eq(target)]
        check(target+'_index_matches_files',set(rr.Run_ID)=={p.parent.name for p in actual_paths})
        current=spec['current']; model=selected['overall']; gid=selected['winners'][model]['group']
        train=None
        for group,g in spec['groups'].items():
            for cid,fs in g['categories'].items():
                cat_export.append(dict(target=target,group_id=group,category_id=cid,category=category_label(fs),
                                       features=json.dumps(fs,ensure_ascii=False),category_feature_count=len(fs),group_feature_count=len(g['features'])))
        selcats=[category_label(fs) for fs in spec['groups'][gid]['categories'].values()]
        selected_meta.append(dict(target=target,model=model,group_id=gid,feature_count=len(spec['groups'][gid]['features']),
                                  categories=' + '.join(selcats),params=json.dumps(selected['winners'][model]['params']),
                                  selection_validation_rmse=selected['winners'][model]['validation_rmse']))
        valid=summary[(summary.Target_ID==target)&(summary.Eval_Split=='validation')&(summary.Prediction_Variant=='bounded')]
        piv=valid.pivot(index=['Variable_Group_ID','Model_Type','Eval_Row_Hash','Seed_Count'],columns='Metric_Name',values='Mean').reset_index()
        check(target+'_all_group_model_pairs',len(piv[piv.Model_Type!='persistence'])==len(spec['groups'])*6)
        check(target+'_same_eval_rows',piv.Eval_Row_Hash.nunique()==1)
        check(target+'_five_validation_seeds',piv[piv.Model_Type!='persistence'].Seed_Count.eq(5).all())
        for m in cfg['models']:
            b=piv[piv.Model_Type==m].set_index('Variable_Group_ID')
            for g1,g2 in itertools.permutations(spec['groups'],2):
                c1=spec['groups'][g1]['categories'];c2=spec['groups'][g2]['categories']
                if set(c1)<set(c2) and len(set(c2)-set(c1))==1:
                    added=next(iter(set(c2)-set(c1)))
                    paired.append(dict(target=target,model=m,base_group=g1,added_group=g2,
                                       added_category=category_label(c2[added]),added_category_id=added,
                                       base_rmse=b.loc[g1,'RMSE'],added_rmse=b.loc[g2,'RMSE'],
                                       delta_rmse=b.loc[g2,'RMSE']-b.loc[g1,'RMSE'],
                                       relative_improvement_pct=100*(1-b.loc[g2,'RMSE']/b.loc[g1,'RMSE']),
                                       stage='validation',interpretation='retuned paired group comparison; descriptive, not causal'))
        p=ROOT/'prepared'/target/'excluded_rows.csv'
        if p.exists():
            ex=pd.read_csv(track(p))
            for reason,n in ex.exclusion_reason.value_counts().items(): exclusions.append(dict(target=target,reason=reason,n=n))
        for split in ['train','validation','test']:
            path=track(ROOT/'prepared'/target/f'{split}.parquet')
            check(target+'_'+split+'_hash',inputs[str(path)]==spec['dataset_hashes'][split]==selected['dataset_hashes'][split])
            d=pd.read_parquet(path)
            check(target+'_'+split+'_unique_rows',not d.row_id.duplicated().any())
            check(target+'_'+split+'_future_label',(d.target_date>d.feature_date).all())
            cohorts.append(dict(target=target,split=split,**describe(d,current)))
            if split=='train':train=d;continue
            check(target+'_'+split+'_facility_disjoint',not set(train.facility_id)&set(d.facility_id))
            if split=='test':check(target+'_val_test_disjoint',not set(spec['split']['validation'])&set(d.facility_id))
            n=len(d); rows=d.set_index('row_id').sort_index()
            y=rows.target.to_numpy(float); persist=rows[current].to_numpy(float)
            selected_predictions={}; raw_predictions={}
            for m in [*cfg['models'],'persistence']:
                group=selected['winners'][m]['group'] if m!='persistence' else 'PERSISTENCE'
                records=rr[(rr.Model_Type==m)&(rr.Stage==split)&(rr.Variable_Group_ID==group)]
                check(target+'_'+split+'_'+m+'_seeds',set(records.Seed)==(SEEDS if m!='persistence' else {42}))
                perseed=[]
                for rec in records.itertuples():
                    rp=ROOT/'experiments'/target/'runs'/rec.Run_ID
                    record=readjson(rp/'result.json')
                    p=pd.read_parquet(track(rp/'predictions.parquet')).set_index('row_id').sort_index()
                    check(rec.Run_ID+'_row_alignment',p.index.equals(rows.index) and np.array_equal(y,p.target) and p[['facility_id','crop_sn','sample_num','feature_date','target_date']].equals(rows[['facility_id','crop_sn','sample_num','feature_date','target_date']]))
                    for variant in ['raw','bounded']:
                        met=metrics(y,p['prediction_'+variant])
                        for k in METRICS:
                            saved=record['Metrics'][variant][k]
                            check(rec.Run_ID+'_'+variant+'_'+k, (saved is None and np.isnan(met[k])) or (saved is not None and np.isclose(saved,met[k],rtol=1e-8,atol=1e-9)))
                            metric_checks+=1
                        all_model.append(dict(target=target,split=split,model=m,group_id=group,seed=rec.Seed,variant=variant,n=n,run_id=rec.Run_ID,**met))
                        if m==model:
                            (selected_predictions if variant=='bounded' else raw_predictions)[int(rec.Seed)]=p['prediction_'+variant].to_numpy(float)
                    if m==model:
                        pp=p.reset_index();pp['seed']=rec.Seed;pp['model']=m;pp['group_id']=group;pp['target_id']=target
                        pp['current_count']=persist;pp['horizon_days']=(rows.target_date-rows.feature_date).dt.total_seconds().to_numpy()/86400
                        pred_export.append(pp)
                if m==model:
                    diagnostics=pd.read_csv(track(rp/'input_diagnostics.csv'));diagnostics['target']=target;diagnostics['split']=split
                    input_diag.append(diagnostics)
            pmat=np.array([selected_predictions[s] for s in sorted(SEEDS)])
            rawmat=np.array([raw_predictions[s] for s in sorted(SEEDS)])
            change=y-persist
            gap=(rows.target_date-rows.feature_date).dt.total_seconds()/86400
            features=spec['groups'][gid]['features']
            xm=rows[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            xt=train[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            missing=xm.isna().mean(axis=1)
            outside=(xm.lt(xt.min())|xm.gt(xt.max())).mean(axis=1)
            upper=readjson(ROOT/'experiments'/target/'runs'/rr[(rr.Model_Type==model)&(rr.Stage==split)&(rr.Variable_Group_ID==gid)].iloc[0].Run_ID/'result.json')['Upper_Bound']
            full_sse=((pmat-y)**2).mean(axis=0).sum()
            masks={'all':pd.Series('all',index=rows.index),
                'facility':rows.facility_id.astype(str),
                'crop_cycle':rows.facility_id.astype(str)+' / '+rows.crop_sn.astype(str),
                'horizon_days':bins(gap,[0,7,14,28,np.inf],['1-7','8-14','15-28','29+']),
                'previous_gap_days':bins(rows.days_since_previous,[0,7,14,28,np.inf],['1-7','8-14','15-28','29+']),
                'crop_age_days':bins(rows.days_since_crop_start,[-np.inf,60,120,180,np.inf],['<=60','61-120','121-180','181+']),
                'change':pd.Series(np.select([change<=-2,change<0,change==0,change<2],['decrease_ge2','decrease_lt2','unchanged','increase_lt2'],default='increase_ge2'),index=rows.index),
                'changed_binary':pd.Series(np.where(change==0,'unchanged','changed'),index=rows.index),
                'label_positive':pd.Series(np.where(y>0,'positive','zero'),index=rows.index),
                'current_positive':pd.Series(np.where(persist>0,'positive','zero'),index=rows.index),
                'history_count':bins(rows.history_count,[-1,1,3,np.inf],['0-1','2-3','4+']),
                'selected_input_missing':pd.Series(np.where(missing>0,'any_missing','complete'),index=rows.index),
                'selected_input_outside_train':pd.Series(np.where(outside>0,'any_outside','in_range'),index=rows.index)}
            for dimension,lab in masks.items():
                for level in sorted(lab.unique()):
                    mask=lab.eq(level).to_numpy(); nn=int(mask.sum())
                    mets=[metrics(y[mask],p[mask]) for p in pmat]
                    item=dict(target=target,split=split,model=model,dimension=dimension,level=level,n=nn,
                              facilities=rows.loc[mask,'facility_id'].nunique(),
                              crop_cycles=len(rows.loc[mask,['facility_id','crop_sn']].drop_duplicates()),
                              row_share=nn/n,zero_rate=np.mean(y[mask]==0),mean_actual=y[mask].mean(),
                              squared_error_share=((pmat[:,mask]-y[mask])**2).mean(axis=0).sum()/full_sse if full_sse else 0,
                              small_n=nn<30,posthoc_label_based=dimension in ['change','changed_binary','label_positive'])
                    for k in mets[0]:
                        vals=np.array([m[k] for m in mets]);ok=vals[np.isfinite(vals)]
                        item[k+'_mean']=ok.mean() if len(ok) else np.nan
                        item[k+'_seed_std']=ok.std(ddof=1) if len(ok)>1 else np.nan
                    item.update({'persistence_'+k:v for k,v in metrics(y[mask],persist[mask]).items()})
                    item['zero_predictor_rmse']=metrics(y[mask],np.zeros(nn))['rmse']
                    strata.append(item)
                    if dimension=='all':
                        item=item.copy();item['group_id']=gid;item['feature_count']=len(features)
                        item['skill_vs_persistence_pct']=100*(1-item['rmse_mean']/item['persistence_rmse']) if item['persistence_rmse'] else np.nan
                        item['raw_rmse_mean']=np.mean([metrics(y,p)['rmse'] for p in rawmat]);item['output_clip_rate']=np.mean(rawmat!=pmat)
                        item['target_above_train_bound_rate']=np.mean(y>upper);item['upper_bound']=upper
                        baseline.append(item)
            for facility in sorted(rows.facility_id.unique()):
                mask=rows.facility_id.ne(facility).to_numpy()
                sensitivity.append(dict(target=target,split=split,removed_facility=facility,remaining_n=int(mask.sum()),
                                        rmse_mean=np.mean([metrics(y[mask],p[mask])['rmse'] for p in pmat]),
                                        persistence_rmse=metrics(y[mask],persist[mask])['rmse']))
            ec=rows[['facility_id','crop_sn','sample_num','feature_date','target_date','target']].copy()
            ec['target_id']=target;ec['split']=split;ec['current_count']=persist;ec['actual_change']=change
            ec['prediction_seed_mean']=pmat.mean(axis=0);ec['mean_absolute_error_across_seeds']=np.abs(pmat-y).mean(axis=0)
            ec['mean_squared_error_across_seeds']=((pmat-y)**2).mean(axis=0);ec['horizon_days']=gap
            ec['crop_age_days']=rows.days_since_crop_start;ec['selected_input_missing_fraction']=missing
            error_cases.append(ec.nlargest(15,'mean_absolute_error_across_seeds').reset_index())
        # Availability is a coverage audit, not feature importance.
        allframes=pd.concat([pd.read_parquet(ROOT/'prepared'/target/f'{s}.parquet') for s in ['train','validation','test']])
        used=set(f for g in spec['groups'].values() for f in g['features'])
        for f in sorted(used|set(c for c in allframes if c in ['leaves_length','leaves_num','leaves_width','petiole_length','stem_diameter','theca_diameter','flower_length','grow_length'])):
            vals=pd.to_numeric(allframes[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
            coverage.append(dict(target=target,feature=f,in_any_registered_group=f in used,nonmissing_rate=vals.notna().mean(),unique_values=vals.nunique()))
    b=save('baseline_performance',baseline)
    save('cohort_profile',cohorts);s=save('error_strata',strata);save('leave_one_facility_sensitivity',sensitivity)
    save('selected_models',selected_meta);save('registered_categories',cat_export);save('feature_coverage',coverage)
    save('paired_category_comparisons',paired);save('selected_input_diagnostics',pd.concat(input_diag,ignore_index=True))
    save('excluded_observations',exclusions);am=save('model_seed_metrics',all_model);save('largest_error_cases',pd.concat(error_cases,ignore_index=True))
    pp=pd.concat(pred_export,ignore_index=True);pp.to_parquet(OUT/'selected_predictions.parquet',index=False)
    group_metrics=summary[(summary.Eval_Split=='validation')&(summary.Prediction_Variant=='bounded')].copy()
    save('all_group_validation_metrics',group_metrics)
    for source in ['Data_Quality.csv','Sum_Comparisons.csv']:
        save('source_'+Path(source).stem.lower(),pd.read_csv(track(ROOT/'csv'/source)))
    for target in TARGETS:
        v=b[(b.target==target)&(b.split=='validation')].iloc[0]
        sel=next(x for x in selected_meta if x['target']==target)
        check(target+'_frozen_selection_metric_reproduced',np.isclose(v.rmse_mean,sel['selection_validation_rmse']))
    # Each partition must account for all rows and all squared error.
    for (t,sp,dim),d in s.groupby(['target','split','dimension']):
        total=int(b[(b.target==t)&(b.split==sp)].iloc[0].n)
        check(f'partition_{t}_{sp}_{dim}',d.n.sum()==total and np.isclose(d.squared_error_share.sum(),1))
    save('integrity_checks',checks)
    audit=dict(created_at=datetime.datetime.now().astimezone().isoformat(),source_root=str(ROOT),
               scheduler_status=scheduler['Status'],campaign_finished_at=datetime.datetime.fromtimestamp(scheduler['Updated_At']).astimezone().isoformat(),
               completed_runs=len(runs),verified_prediction_files=len(am)//2,metric_recomputations=metric_checks,
               checks=len(checks),passed=all(x['passed'] for x in checks),
               scope='All run file IDs/status index, dataset hashes, frozen selection and 496 selected/baseline prediction files. Other candidate metrics use hashed saved summary; no refitting.',
               confidence='seed SD is optimization variability, not population uncertainty; test has 2 tomato / 3 strawberry facilities',
               holdout_notice='Previously inspected holdout; diagnostic reuse only, not a new independent confirmation')
    (HERE/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    track(Path(__file__))
    (HERE/'source_hashes.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    print(b[b.split=='test'][['target','model','n','rmse_mean','mae_mean','r2_mean','ccc_mean','persistence_rmse','skill_vs_persistence_pct']].to_string(index=False))

if __name__=='__main__':main()
