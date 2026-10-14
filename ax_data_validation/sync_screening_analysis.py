"""Rebuild diagnostics from frozen screening summaries and winner Test predictions.
No fitting, DB queries, or writes to experiment sources.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK.parent))
from AXData.ax_catalog_experiments.exhaustive.prepare import _tomato, _strawberry, ENTITY

ROOT = WORK / 'ax_catalog_experiments/outputs/exhaustive_e6_semantic_screening_v1'
V1 = WORK / 'ax_catalog_experiments/outputs/exhaustive_e1_e7_v1'
DEST = HERE / 'ax_direction'
SOURCE = ROOT.name
METRICS = ('rmse', 'mae', 'r2', 'ccc')
GROUP_NAMES = {'E1':'생육·생식 이력','E2':'영양생장','E3':'시설 내 환경','E4':'외부 기상','E5':'근권·양액','E6':'장치 의미 통합 제어','E7':'시설·작기 맥락'}


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False, separators=(',',':')))
    tmp.replace(path)


def stats(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if not len(y): return {k:None for k in (*METRICS,'bias','p90_ae','within1')}
    if not (np.isfinite(y).all() and np.isfinite(p).all()): raise ValueError('Nonfinite evaluation values')
    e=p-y;den=y.var()+p.var()+(y.mean()-p.mean())**2
    return dict(rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),
                r2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
                ccc=float(2*np.mean((y-y.mean())*(p-p.mean()))/den) if den else 1.,
                bias=float(e.mean()),p90_ae=float(np.quantile(abs(e),.9)),within1=float(np.mean(abs(e)<=1)))


def average_metrics(y, matrix):
    values=[stats(y,p) for p in matrix];out={}
    for k in values[0]:
        finite=[v[k] for v in values if v[k] is not None]
        out[k+'_mean']=float(np.mean(finite)) if finite else None
        out[k+'_seed_std']=float(np.std(finite,ddof=1)) if len(finite)>1 else None
    return out


def bins(series, edges, labels):
    return pd.cut(series, edges, labels=labels,include_lowest=True).astype('string').fillna('missing')


def current_values(frame, spec):
    cols=[f'count_{i}__current' for i in spec['components']]
    return frame[cols].sum(axis=1,min_count=len(cols)).to_numpy(float)


def build():
    scheduler=json.loads((ROOT/'screening_scheduler.json').read_text())
    assert scheduler['Status']=='finished' and scheduler['Consolidated']['prediction_runs']==144
    bundle=json.loads((DEST/'dashboard_bundle.json').read_text())
    backup=WORK/'ax_catalog_experiments/outputs/dashboard_analysis_before_screening_20261014.json'
    if not backup.exists(): backup.write_text(json.dumps(bundle,ensure_ascii=False))
    rank=pd.read_csv(ROOT/'validation_rankings.csv')
    assert len(rank)==6096 and rank.Status.eq('success').all()
    allpred=pd.read_parquet(ROOT/'final_test_predictions.parquet')
    hashes={}
    for f in ['validation_rankings.csv','validation_winners.csv','final_test_predictions.parquet']:
        hashes[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    tables={k:[] for k in ['baseline_performance','cohort_profile','selected_models','error_strata','crossed_error_strata','target_diagnosis','category_effect_summary','model_comparison','target_distribution']}
    frozen_snapshot=json.loads((V1/'data_snapshot/manifest.json').read_text())
    for name in ['sfkr_hbfm_grow','sfkr_pvsn_grow','sfkr_pvsn_crop','sfkr_pvsn_id']:
        definition=frozen_snapshot['tables'][name]
        path=V1/'data_snapshot'/definition['file']
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==definition['sha256'],f'Frozen source changed: {name}'
        hashes['snapshot/'+name]=actual
    gap_by_crop={}
    for crop,loader in [('tomato',_tomato),('strawberry',_strawberry)]:
        raw,_=loader(V1)
        raw.feature_date=pd.to_datetime(raw.feature_date).dt.normalize()
        for key in ENTITY: raw[key]=raw[key].astype(str)
        raw=raw.sort_values(ENTITY+['feature_date'])
        assert not raw.duplicated(ENTITY+['feature_date']).any()
        raw['previous_gap_days']=raw.groupby(ENTITY,dropna=False).feature_date.diff().dt.days
        gap_by_crop[crop]=raw[ENTITY+['feature_date','previous_gap_days']]
    checks=0
    for target in sorted(rank.Target_ID.unique()):
        print('Analyze',target,flush=True)
        candidates=rank[rank.Target_ID.eq(target)].sort_values(['Validation_RMSE','Validation_MAE','Feature_Count','Combination_ID','Model_Type'],kind='stable')
        best=candidates.iloc[0];model=best.Model_Type;gid=best.Combination_ID
        spec=json.loads((ROOT/'prepared'/target/'manifest.json').read_text())
        selection=json.loads((ROOT/'selections'/target/f'{model}.json').read_text())
        assert selection['winner']['Combination_ID']==gid
        tables['selected_models'].append(dict(target=target,model=model,group_id=gid,feature_count=int(best.Feature_Count),categories=best.Variable_Groups.replace('|',' + '),selection_validation_rmse=best.Validation_RMSE,source=SOURCE))
        frames={}
        for split in ['train','validation','test']:
            f=ROOT/'prepared'/target/f'{split}.parquet'
            assert hashlib.sha256(f.read_bytes()).hexdigest()==spec['dataset_hashes'][split]
            frame=pd.read_parquet(f).set_index('row_id').sort_index()
            assert frame.index.is_unique
            for c in ['feature_date','target_date']:frame[c]=pd.to_datetime(frame[c])
            for c in ENTITY:frame[c]=frame[c].astype(str)
            gap=frame.reset_index()[['row_id',*ENTITY,'feature_date']].merge(gap_by_crop[target.split('_')[0]],on=ENTITY+['feature_date'],how='left',validate='one_to_one',indicator=True)
            assert gap['_merge'].eq('both').all()
            frame['previous_gap_days']=gap.set_index('row_id').previous_gap_days.reindex(frame.index)
            frames[split]=frame; current=current_values(frame,spec);y=frame.target.to_numpy(float)
            tables['cohort_profile'].append(dict(target=target,split=split,n=len(frame),facilities=int(frame.facility_id.nunique()),crop_cycles=len(frame[ENTITY[:2]].drop_duplicates()),zero_rate=float(np.mean(y==0)),positive_n=int(sum(y>0)),feature_start=str(frame.feature_date.min().date()),target_end=str(frame.target_date.max().date()),source=SOURCE))
            if split=='train':
                observed=current[np.isfinite(current)]
                if not len(observed): raise ValueError(f'No train current values: {target}')
                fallback=float(np.median(observed))
                continue
            for v,n in frame.target.value_counts().sort_index().items():
                tables['target_distribution'].append(dict(target_id=target,split=split,value=float(v),n=int(n),rate=n/len(y),total_n=len(y),zero_n=int(sum(y==0)),positive_n=int(sum(y>0)),zero_rate=float(np.mean(y==0)),positive_rate=float(np.mean(y>0)),source=SOURCE))
            base=dict(target=target,split=split,model=model,group_id=gid,feature_count=int(best.Feature_Count),n=len(y),facilities=int(frame.facility_id.nunique()),crop_cycles=len(frame[ENTITY[:2]].drop_duplicates()),zero_rate=float(np.mean(y==0)),source=SOURCE)
            persistence=np.where(np.isfinite(current),current,fallback)
            base.update(persistence_missing_n=int(np.sum(~np.isfinite(current))),persistence_fallback_value=fallback,persistence_policy='현재 관측값 유지; 현재값 미관측 행은 train 현재값 중앙값 사용')
            for label,pred in [('persistence',persistence),('zero',np.zeros(len(y)))]:
                base.update({label+'_'+k:v for k,v in stats(y,pred).items()})
            for m in sorted(rank.Model_Type.unique()):
                win=json.loads((ROOT/'validation_winners'/target/f'{m}.json').read_text())
                if split=='validation':
                    for variant,key in [('bounded','Bounded'),('raw','Raw')]:
                        met=win['Runs'][0][key]
                        tables['model_comparison'].append(dict(target=target,split=split,model=m,group_id=win['Combination_ID'],variant=variant,n=len(y),seed_count=1,**{k+'_mean':met[k] for k in METRICS},source=SOURCE))
                    if m==model:base.update({k+'_mean':win['Runs'][0]['Bounded'][k] for k in METRICS});base['seed_count']=1
                    continue
                selected=json.loads((ROOT/'selections'/target/f'{m}.json').read_text())
                assert selected['frozen_features']==win['Selected_Features']
                pred=allpred[allpred.Target_ID.eq(target)&allpred.Model_Type.eq(m)]
                assert set(pred.Seed)=={42,52,62}
                matrices={}
                for variant in ['raw','bounded']:
                    matrix=[]
                    for seed in [42,52,62]:
                        pf=pred[pred.Seed.eq(seed)].set_index('row_id').sort_index()
                        assert pf.index.equals(frame.index) and np.array_equal(pf.target,y)
                        assert set(pf.Combination_ID)=={win['Combination_ID']}
                        assert set(pf.Run_ID).issubset(set(selected['test_runs']))
                        for key in ENTITY:assert np.array_equal(pf[key].astype(str),frame[key])
                        for key in ['feature_date','target_date']:assert np.array_equal(pd.to_datetime(pf[key]),frame[key])
                        matrix.append(pf['prediction_'+variant].to_numpy(float))
                    matrices[variant]=np.array(matrix)
                    met=average_metrics(y,matrices[variant])
                    if variant=='bounded':
                        for k in METRICS:
                            expected=selected['Test_'+k.upper()+'_Mean']
                            assert np.isclose(met[k+'_mean'],expected,rtol=1e-8,atol=1e-10)
                            checks+=1
                    tables['model_comparison'].append(dict(target=target,split=split,model=m,group_id=win['Combination_ID'],variant=variant,n=len(y),seed_count=3,**met,source=SOURCE))
                if m==model:
                    pmat=matrices['bounded'];base.update(average_metrics(y,pmat));base['seed_count']=3
                    base['output_clip_rate']=float(np.mean(matrices['raw']!=pmat))
            base['skill_vs_persistence_pct']=100*(1-base['rmse_mean']/base['persistence_rmse']) if base['persistence_rmse'] else None
            tables['baseline_performance'].append(base)
            if split=='validation':
                tables['target_diagnosis'].append(dict(target=target,split=split,finding='고정 설정·1시드 Validation 전체 성능지표입니다. 행별 예측값을 저장하지 않아 구간별 오차는 제공하지 않습니다.',next_validation='최신 Test 구간별 진단을 확인하고, 정식 추가 검증 완료 후 Validation 상세 분석을 갱신합니다.',evidence='validation_rankings.csv',source=SOURCE))
                continue
            change=y-current
            change_label=np.select([change<=-2,change<0,change==0,change<2],['decrease_ge2','decrease_lt2','unchanged','increase_lt2'],default='increase_ge2')
            change_label=np.where(np.isfinite(change),change_label,'current_missing')
            selected_cols=selection['frozen_features']
            missing=frame[selected_cols].replace([np.inf,-np.inf],np.nan).isna().any(axis=1)
            masks={'all':pd.Series('all',index=frame.index),'facility':frame.facility_id,
                   'crop_cycle':frame.facility_id+' / '+frame.crop_sn,
                   'crop_age_days':bins(frame.days_since_crop_start,[-np.inf,60,120,180,np.inf],['<=60','61-120','121-180','181+']),
                   'previous_gap_days':bins(frame.previous_gap_days,[0,7,14,28,np.inf],['1-7','8-14','15-28','29+']),
                   'horizon_days':bins((frame.target_date-frame.feature_date).dt.days,[0,7,14,28,np.inf],['1-7','8-14','15-28','29+']),
                   'change':pd.Series(change_label,index=frame.index),
                   'changed_binary':pd.Series(np.where(~np.isfinite(change),'current_missing',np.where(change==0,'unchanged','changed')),index=frame.index),
                   'label_positive':pd.Series(np.where(y>0,'positive','zero'),index=frame.index),
                   'current_positive':pd.Series(np.where(~np.isfinite(current),'current_missing',np.where(current>0,'positive','zero')),index=frame.index),
                   'selected_input_missing':pd.Series(np.where(missing,'any_missing','complete'),index=frame.index)}
            sse=float(np.sum(np.mean((pmat-y)**2,axis=0)))
            def stratum(dimension,level,mask):
                n=int(mask.sum());err=float(np.sum(np.mean((pmat[:,mask]-y[mask])**2,axis=0)))
                return dict(target=target,split='test',model=model,dimension=dimension,level=str(level),n=n,facilities=int(frame.loc[mask].facility_id.nunique()),crop_cycles=len(frame.loc[mask,ENTITY[:2]].drop_duplicates()),row_share=n/len(y),squared_error_share=err/sse if sse else 0.,small_n=n<30,posthoc_label_based=dimension in ['change','changed_binary','label_positive','horizon_days'],**average_metrics(y[mask],pmat[:,mask]),source=SOURCE)
            local=[]
            for dim,labels in masks.items():
                partition=[stratum(dim,label,labels.eq(label).to_numpy()) for label in sorted(labels.unique())]
                assert sum(r['n'] for r in partition)==len(y)
                if sse: assert np.isclose(sum(r['squared_error_share'] for r in partition),1.)
                checks+=1;local.extend(partition)
            tables['error_strata'].extend(local)
            for left,right in [('facility','crop_age_days'),('facility','horizon_days'),('change','label_positive')]:
                labels=masks[left].astype(str)+' / '+masks[right].astype(str)
                for label in sorted(labels.unique()):
                    row=stratum(left+'+'+right,label,labels.eq(label).to_numpy());row['dimensions']=left+'+'+right
                    tables['crossed_error_strata'].append(row)
            worst=max([r for r in local if r['dimension']=='facility'],key=lambda r:r['squared_error_share'])
            relation=f"현재값 유지 대비 RMSE {abs(base['skill_vs_persistence_pct']):.1f}% {'감소' if base['skill_vs_persistence_pct']>=0 else '증가'}" if base['skill_vs_persistence_pct'] is not None else '유지 기준 오차가 0이라 상대 개선율 미정의'
            tables['target_diagnosis'].append(dict(target=target,split='test',finding=f"Validation에서 선정한 {model.upper()} / {best.Variable_Groups.replace('|','+')}의 Test {len(y)}행: {relation}. 정답 0 비율 {np.mean(y==0)*100:.1f}%, 양수 {sum(y>0)}행. {worst['level']} 시설의 제곱오차 비중 {worst['squared_error_share']*100:.1f}% (N={worst['n']}).",next_validation='시설·변화 구간의 표본 수와 오차를 함께 확인하고, 관측 정의·작업 이력의 추가 효과를 동일 신규 평가 코호트에서 검증합니다. 사후 오차 집중만으로 원인을 단정하지 않습니다.',evidence='최신 선별 우승 조합 Test 예측 · error_strata.csv',source=SOURCE))
    # Matched nonempty subsets; each pair differs by exactly one variable group.
    paired=[]
    for (target,model),group in rank.groupby(['Target_ID','Model_Type']):
        lookup={frozenset(r.Variable_Groups.split('|')):r for r in group.itertuples()}
        for key,r in lookup.items():
            for added in GROUP_NAMES:
                other=lookup.get(key|{added}) if added not in key else None
                if other:paired.append(dict(target=target,model=model,added_category=GROUP_NAMES[added],added_group=added,base_combination=r.Combination_ID,expanded_combination=other.Combination_ID,delta_rmse=other.Validation_RMSE-r.Validation_RMSE))
    pairs=pd.DataFrame(paired)
    for (target,cat),g in pairs.groupby(['target','added_category']):
        tables['category_effect_summary'].append(dict(target=target,added_category=cat,comparisons=len(g),median_rmse_delta=g.delta_rmse.median(),min_rmse_delta=g.delta_rmse.min(),max_rmse_delta=g.delta_rmse.max(),improved_comparisons=int(g.delta_rmse.lt(0).sum()),source=SOURCE,interpretation='같은 모델·고정 설정·1시드의 한 변수군 추가 비교; 선택 변수가 함께 달라질 수 있으며 인과 효과가 아님'))
    bundle['tables'].update(tables)
    for row in bundle['tables']['kpi_planning_reference']:
        target=row['target'];base=next(r for r in tables['baseline_performance'] if r['target']==target and r['split']=='validation')
        changed=next((r for r in tables['error_strata'] if r['target']==target and r['dimension']=='changed_binary' and r['level']=='changed'),None)
        assert np.isclose(base['rmse_mean'],row['reference_rmse'])
        row['reference_snapshot_label']=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
        row.update(reference_persistence_rmse=base['persistence_rmse'],reference_zero_rmse=base['zero_rmse'],reference_change_mae=changed['mae_mean'] if changed else None,illustrative_15pct_change_mae=changed['mae_mean']*.85 if changed else None,reference_change_n=changed['n'] if changed else 0,reference_change_split='test',reference_change_label='최신 Test · 실제 수량 변화 행 · 3시드 MAE 평균',reference_change_small_n=changed['small_n'] if changed else True,reference_persistence_missing_n=base['persistence_missing_n'],reference_persistence_policy=base['persistence_policy'])
    h=bundle['tables']['hypotheses']
    evidence={
        'H01':'E1·E2를 포함한 127개 조합 6,096건의 1시드 선별 완료. E2 추가 쌍별 비교는 category_effect_summary에 수록.',
        'H02':'최신 Test의 타깃별 0·양수 분포와 변화·작기 경과일별 오차를 재계산함. target_distribution 및 error_strata 참조.',
        'H03':'최신 Test의 증가·감소 구간 MAE와 편향을 계산함. 작업 기록이 없어 관측된 오차를 특정 작업의 영향으로 단정할 수 없음.',
        'H04':'최신 Test 시설별 오차 및 시설×작기 경과일 진단을 재계산함. target_diagnosis 및 crossed_error_strata 참조.',
        'H05':'독립 시설 수와 양수·변화 사건 수를 최신 평가 코호트에서 집계함. 소표본 구간은 N<30 표시.',
        'H06':'현 선별 실험에는 신규 제약 실측 입력이 없어 제약의 증분 효과는 아직 검증하지 않음.',
        'H07':'최신 합계 타깃의 Test 성능을 동일 행의 현재값 유지·0 예측과 비교함. 합계 예측 방식 간 우열은 별도 검증 필요.'}
    for row in h:
        row['evidence']=evidence[row['id']]
        if row['id']=='H01':row.update(status='기존 E1·E2 선별 완료 · 정식 추가 검증 중',validation='기존 E1·E2 포함 여부에 따른 동일 모델·평가 행의 성능을 비교하고 새 코호트에서 재검증')
    for row in bundle['tables']['comparison_arms']:
        if row['arm']=='B0':row['inputs']='E1~E7 기존 DB 입력 중 선정한 기준 구성; 새 코호트에서 재학습'
        if row['arm']=='B1':row.update(name='기존 입력 활용범위 비교',inputs='B0 대비 기존 생육·영양생장 입력 포함 범위 비교',purpose='E1·E2 활용 선별 결과를 새 코호트에서 확인')
    now=datetime.now(timezone.utc).isoformat()
    preserved=['ax_field_schema','excluded_observations','crop_cycle_quality']
    bundle['metadata']={'created_at':now,'source_campaign':SOURCE,'validation_seeds':[42],'test_seeds':[42,52,62],'completed_screening':6096,'verified_test_predictions':144,'checks':checks,'passed':True,'source_hashes':hashes,'diagnostic_model_selection':'타깃별 Validation RMSE/MAE/변수 수/조합 ID 순으로 선정; Test로 재선정하지 않음','available_error_splits':['test'],'validation_error_notice':'Validation은 집계 성능만 저장했으므로 행별 구간 오차·시계열은 미제공. 상세 오차는 Test에서 확인.','persistence_policy':'모든 평가 행을 유지. 현재값이 없을 때 train 현재값 중앙값으로 대체하며 대체 행 수를 표시. 변화 구간에서는 현재값 결측을 별도 분리.', 'observation_notice':'기존 고정 원천 스냅숏 기준. 실시간 DB 신규 적재를 재조회한 결과가 아님.','preserved_reference_tables':preserved,'table_sources':{k:('고정 원천 관측 품질 자료 또는 사업 정의서; 이번 실험으로 재산출하지 않음' if k in preserved else SOURCE) for k in bundle['tables']},'holdout_notice':'이미 확인한 Test 코호트의 재사용 진단이며 새로운 독립 외부 검증이 아님. 시드 표준편차는 모집단 신뢰구간이 아님.'}
    export=DEST/'screening_analysis'
    export.mkdir(exist_ok=True)
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(export/f'{name}.csv',index=False,encoding='utf-8-sig')
    pairs.to_csv(export/'paired_category_comparisons.csv',index=False,encoding='utf-8-sig')
    write(DEST/'dashboard_bundle.json',bundle)
    write(export/'audit.json',bundle['metadata'])
    print(json.dumps({'checks':checks,'table_rows':{k:len(v) for k,v in tables.items()}},ensure_ascii=False))


if __name__=='__main__':build()
