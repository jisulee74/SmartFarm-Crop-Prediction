from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .catalog import AX
from .core import digest,freeze,growth_features,sensor_features,facility_split,qualify,file_hash

TARGETS={'tomato_total_flower':('토마토','전체 꽃 수'),'strawberry_first':('딸기','1화방 착과수'),'strawberry_second':('딸기','2화방 착과수'),'strawberry_third':('딸기','3화방 착과수')}


def prepare_tomato(out,variables):
    from AXData.tomato_flower_count_prediction.pipeline import prepare_growth
    root=AX/'tomato_flower_count_prediction/data/raw'
    raw=pd.read_parquet(root/'growth.parquet')
    rows,audits=prepare_growth(raw)
    rows=rows.rename(columns={'current_total_flower_count':'flower_count','next_total_flower_count':'target'})
    entity=['facility_id','crop_sn','sample_num']
    bases=['flower_count','current_max_truss_number','positive_truss_count','observed_truss_count']
    # Drop legacy derived columns rather than accidentally reusing their semantics.
    retained=[*entity,'feature_date','target_date','target','row_id','cropping_date',*bases]
    rows=growth_features(rows[retained],entity,bases)
    rows['days_since_crop_start']=(rows.feature_date-rows.cropping_date).dt.days
    sensor=pd.read_parquet(root/'environment.parquet')
    keys=[c for c in ['facility_id','fld_code','sect_code','fatr_code','maker_id','sen_id','meas_date'] if c in sensor]
    if 'source_sn' in sensor:sensor=sensor.sort_values('source_sn')
    sensor=sensor.drop_duplicates(keys,keep='last')
    sensor['variable_id']=sensor.sect_code.astype(str)+'__'+sensor.fatr_code.astype(str)
    sensor=sensor.rename(columns={'sen_val':'sensor_value'})
    sensor['sensor_value']=pd.to_numeric(sensor.sensor_value,errors='coerce').astype(float)
    # Existing documented bounds; other raw codes are retained but not invented.
    for var,lo,hi in [('EI__TI',-10,60),('EI__HI',0,100),('EI__CI',0,5000)]:
        mask=sensor.variable_id.eq(var); val=pd.to_numeric(sensor.loc[mask,'sensor_value'],errors='coerce')
        sensor.loc[mask,'sensor_value']=val.where(val.between(lo,hi))
    sensor_defs={}
    available_sensor_ids=set(sensor.variable_id)
    for f in variables[variables.Availability_Status.eq('definition_ready')].itertuples():
        if f.Source_Table=='sfkr_hbfm_env_con':
            v=f.Variable_ID.rsplit('__',1)[0]
            if v in available_sensor_ids:sensor_defs[v]='active_fraction' in f.Variable_ID
    rows=sensor_features(rows,sensor,sensor_defs)
    return save_prepared(out,rows,variables,'tomato_total_flower','flower_count__current',entity,{'growth':str(root/'growth.parquet'),'environment':str(root/'environment.parquet'),'audit_counts':{k:len(v) for k,v in audits.items()},'history_complete':True})


def save_prepared(out,rows,variables,target_id,current,entity,provenance):
    out.mkdir(parents=True,exist_ok=True)
    if rows.row_id.duplicated().any():raise ValueError('Duplicate row IDs')
    if not (rows.feature_date<rows.target_date).all():raise ValueError('Targets must be future observations')
    split=facility_split(rows); train=rows[rows.facility_id.isin(split['train'])]
    categories={};availability=[];cat_records=[]
    for cid,v in variables.groupby('Category_ID',sort=False):
        candidates=v[v.Availability_Status.eq('definition_ready')].Variable_ID.tolist()
        keep,audit=qualify(train,candidates)
        for a in audit:a.update(Target_ID=target_id,Category_ID=cid);availability.append(a)
        for r in v[~v.Availability_Status.eq('definition_ready')].itertuples():
            availability.append({'Target_ID':target_id,'Category_ID':cid,'Variable_ID':r.Variable_ID,'Missing_Rate':None,'Status':'needs_review','Reason':r.Reason})
        if keep:
            frozen_id=cid.replace('_V1','')+'_'+target_id+'_'+digest(keep)[:10]+'_V1'
            categories[frozen_id]=keep
            cat_records.append({'Category_ID':frozen_id,'Source_Category_ID':cid,'Target_ID':target_id,'Version':1,'Status':'frozen'})
    if not any(current in v for v in categories.values()):raise ValueError('Required current target unavailable')
    columns=sorted(set(['row_id','facility_id','feature_date','target_date','target',current,*entity,*[x for v in categories.values() for x in v]]))
    rows=rows[columns].sort_values('row_id').reset_index(drop=True)
    dataset_id=digest(pd.util.hash_pandas_object(rows,index=False).astype(str).tolist())
    spec={'target_id':target_id,'target':'target','current':current,'entity':entity,'categories':categories,'dataset_id':dataset_id,'split':split,'split_id':digest(split),'provenance':provenance,'prediction_policy':'clip_zero_no_rounding','time_window':'[t-days,t)'}
    freeze(out/'variable_definitions.json',variables[variables.Variable_ID.isin({f for fs in categories.values() for f in fs})].to_dict('records'))
    freeze(out/'manifest.json',spec)
    for name,facilities in split.items():rows[rows.facility_id.isin(facilities)].to_parquet(out/f'{name}.parquet',index=False)
    freeze(out/'data_hashes.json',{name:file_hash(out/f'{name}.parquet') for name in split})
    pd.DataFrame(availability).to_csv(out/'availability.csv',index=False)
    pd.DataFrame(cat_records).to_csv(out/'categories.csv',index=False)
    return spec


def prepare_all(output,variables):
    status=[]
    try:
        spec=prepare_tomato(output/'tomato_total_flower',variables)
        status.append({'Target_ID':'tomato_total_flower','Status':'prepared','Reason':'local raw cache rebuilt with new definitions','Dataset_ID':spec['dataset_id']})
    except Exception as e:
        status.append({'Target_ID':'tomato_total_flower','Status':'blocked','Reason':str(e)})
    for target in list(TARGETS)[1:]:
        try:
            spec=prepare_strawberry(output/target,variables,target)
            status.append({'Target_ID':target,'Status':'prepared','Reason':'audited observation cohort; bounded history documented','Dataset_ID':spec['dataset_id']})
        except Exception as e:
            status.append({'Target_ID':target,'Status':'blocked','Reason':str(e)})
    pd.DataFrame(status).to_csv(output/'target_status.csv',index=False)
    return status


def prepare_strawberry(out,variables,target_id):
    root=AX/'strawberry_fruit_set_prediction'
    audit=root/'data_audits/fruit_zero_semantics_20260812/artifacts/analysis_rows.csv'
    rows=pd.read_csv(audit,dtype={'user_id':str,'sample_num':str,'cropping_serl_no':str})
    mapping=pd.concat([pd.read_parquet(root/'source_processed_data'/f'{s}_candidates.parquet',columns=['user_id','facility_id']) for s in ('train','val','test')]).drop_duplicates()
    if mapping.groupby('user_id').facility_id.nunique().gt(1).any():raise ValueError('Ambiguous user to facility mapping')
    before=len(rows);rows=rows.merge(mapping,on='user_id',how='inner',validate='many_to_one')
    rows['feature_date']=pd.to_datetime(rows.meas_date)
    rows['cropping_date']=pd.to_datetime(rows.cropping_date)
    rows['crop_sn']=rows.user_id+'|'+rows.cropping_serl_no+'|'+rows.cropping_date.astype(str)
    field=target_id.split('_')[1]+'_fruits_num'
    rows['fruit_count']=pd.to_numeric(rows[field],errors='coerce')
    rows=rows[rows.fruit_count.notna() & rows.fruit_count.ge(0)].copy()
    entity=['facility_id','crop_sn','sample_num']
    rows=rows.sort_values([*entity,'feature_date']).reset_index(drop=True)
    if rows.duplicated([*entity,'feature_date']).any():raise ValueError('Audit contains duplicate observation keys')
    rows['target']=rows.groupby(entity).fruit_count.shift(-1)
    rows['target_date']=rows.groupby(entity).feature_date.shift(-1)
    rows['row_id']=[digest([target_id,*values]) for values in rows[[*entity,'feature_date']].itertuples(index=False,name=None)]
    rows=growth_features(rows,entity,['fruit_count'])
    rows['days_since_crop_start']=(rows.feature_date-rows.cropping_date).dt.days
    first_positive=rows.feature_date.where(rows.fruit_count.gt(0)).groupby([rows[c] for c in entity]).transform('min')
    rows['days_since_first_positive']=(rows.feature_date-first_positive).dt.days.where(rows.feature_date.ge(first_positive))
    rows=rows[rows.target.notna()].copy()
    sensors=pd.read_parquet(root/'data/sensor_category_cache/sensor_collapsed.parquet')
    available=set(sensors.variable_id);defs={}
    for r in variables[variables.Availability_Status.eq('definition_ready')].itertuples():
        if r.Source_Table=='sfkr_hbfm_env_con':
            v=r.Variable_ID.rsplit('__',1)[0]
            if v in available:defs[v]='active_fraction' in r.Variable_ID
    rows=sensor_features(rows,sensors,defs)
    return save_prepared(out,rows,variables,target_id,'fruit_count__current',entity,{'source':str(audit),'history_scope':'valid observations retained by source audit, not full DB history','source_filter':'meas_date >= 2024-05-02; fruit_cluster_num IS NULL; all three fruit count fields non-null; ambiguous crop assignments/conflicting observations removed by audit','unmapped_rows_excluded':before-len(pd.read_csv(audit).merge(mapping,on='user_id',how='inner')),'facility_mapping':'unique mapping from retained candidates','sensor_source':str(root/'data/sensor_category_cache/sensor_collapsed.parquet')})
