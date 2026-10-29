from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..catalog import AX
from ..core import digest,freeze,file_hash,growth_features,sensor_features

ENTITY=['facility_id','crop_sn','sample_num']
PARTS={'first':1,'second':2,'third':3,'sum123':None}
TARGETS={f'{crop}_{part}':(crop,part) for crop in ('tomato','strawberry') for part in PARTS}
SOURCE=AX/'ax_catalog_experiments/outputs'

def csv(name):return pd.read_csv(SOURCE/'csv'/f'{name}.csv')
def manifest(target):return json.loads((SOURCE/'prepared'/target/'manifest.json').read_text())
def normalize(frame):
    f=frame.copy()
    for k in ENTITY:f[k]=f[k].astype(str)
    f['feature_date']=pd.to_datetime(f.feature_date).dt.normalize()
    return f

def observations(crop):
    if crop=='tomato':
        p=AX/'tomato_flower_count_prediction/data/raw/growth.parquet'
        d=pd.read_parquet(p);d=d.rename(columns={'examin_date':'feature_date'});d=normalize(d)
        for c in ['cropping_date','cropping_end_date']:d[c]=pd.to_datetime(d[c]).dt.normalize()
        d['truss']=pd.to_numeric(d.growth_measure_code,errors='coerce')-10000234+1
        d['value']=pd.to_numeric(d.flower_count,errors='coerce')
        badsource=d.groupby('source_sn').crop_sn.nunique();badsource=set(badsource[badsource>1].index)
        structure_bad=d.source_sn.isin(badsource)|~d.feature_date.between(d.cropping_date,d.cropping_end_date)|~d.truss.between(1,34)
        calendar=d.loc[~structure_bad,ENTITY+['feature_date','cropping_date']].drop_duplicates()
        invalid=structure_bad|~d.value.ge(0)
        audits=[dict(Reason='invalid_or_ambiguous_raw_row',Rows=int(invalid.sum()))]
        d=d[~invalid].copy();keys=ENTITY+['feature_date']
        conflicts=d.groupby(keys+['truss']).value.nunique();bad=conflicts[conflicts>1].reset_index()[keys].drop_duplicates()
        d=d.merge(bad.assign(_bad=True),on=keys,how='left');audits.append(dict(Reason='conflicting_observation_rows',Rows=int(d._bad.notna().sum())))
        d=d[d._bad.isna()].drop_duplicates(keys+['truss'])
        rows=d.groupby(keys).agg(cropping_date=('cropping_date','first'),
          current_max_truss_number=('truss','max'),observed_truss_count=('truss','nunique'),
          positive_truss_count=('value',lambda s:int(s.gt(0).sum()))).reset_index()
        wide=d[d.truss.isin([1,2,3])].pivot(index=keys,columns='truss',values='value').reindex(columns=[1,2,3])
        wide.columns=[f'count_{i}' for i in (1,2,3)]
        rows=rows.merge(wide.reset_index(),on=keys,how='left',validate='one_to_one')
        rows=calendar.merge(rows.drop(columns='cropping_date'),on=keys,how='left',validate='one_to_one')
    else:
        p=AX/'strawberry_fruit_set_prediction/data_audits/fruit_zero_semantics_20260812/artifacts/analysis_rows.csv'
        d=pd.read_csv(p,dtype={'user_id':str,'sample_num':str,'cropping_serl_no':str})
        root=AX/'strawberry_fruit_set_prediction/source_processed_data'
        m=pd.concat([pd.read_parquet(root/f'{s}_candidates.parquet',columns=['user_id','facility_id']) for s in ['train','val','test']]).drop_duplicates()
        if m.user_id.duplicated().any():raise ValueError('Ambiguous facility mapping')
        d=d.merge(m,on='user_id',how='left',validate='many_to_one')
        audits=[dict(Reason='unmapped_facility',Rows=int(d.facility_id.isna().sum()))];d=d[d.facility_id.notna()].copy()
        d['crop_sn']=d.user_id+'|'+d.cropping_serl_no+'|'+d.cropping_date.astype(str)
        d=d.rename(columns={'meas_date':'feature_date','first_fruits_num':'count_1','second_fruits_num':'count_2','third_fruits_num':'count_3'})
        rows=normalize(d);rows['cropping_date']=pd.to_datetime(rows.cropping_date)
        for c in ['count_1','count_2','count_3']:rows[c]=pd.to_numeric(rows[c],errors='coerce').where(lambda x:x>=0)
    rows=rows.sort_values(ENTITY+['feature_date']).reset_index(drop=True)
    if rows.duplicated(ENTITY+['feature_date']).any():raise ValueError('Conflicting individual dates')
    rows['count_sum123']=rows[['count_1','count_2','count_3']].sum(axis=1,min_count=3)
    g=rows.groupby(ENTITY,dropna=False)
    rows['target_date']=g.feature_date.shift(-1)
    for i in (1,2,3):rows[f'target_truss_{i}']=g[f'count_{i}'].shift(-1)
    return rows,audits,{'path':str(p),'sha256':file_hash(p),'history_scope':'retained source cache/audit; not full DB inventory'}

def environmental_context(rows,crop):
    old='tomato_total_flower' if crop=='tomato' else 'strawberry_first'
    frames=[pd.read_parquet(SOURCE/'prepared'/old/f'{s}.parquet') for s in ('train','validation','test')]
    cache=pd.concat(frames,ignore_index=True)
    columns=[c for c in cache if '__' in c and not c.startswith(('flower_count','fruit_count','current_max','observed_truss','positive_truss'))]
    keys=['facility_id','feature_date'];cache=cache[keys+columns].drop_duplicates()
    if cache.duplicated(keys).any():raise ValueError('Conflicting facility environment cache')
    cache['facility_id']=cache.facility_id.astype(str)
    cache['feature_date']=pd.to_datetime(cache.feature_date)
    joined=rows[keys].drop_duplicates().merge(cache,on=keys,how='left',indicator=True)
    missing=joined.loc[joined._merge.eq('left_only'),keys]
    if len(missing):
        if crop=='tomato':
            raw=pd.read_parquet(AX/'tomato_flower_count_prediction/data/raw/environment.parquet')
            dedup=[c for c in ['facility_id','fld_code','sect_code','fatr_code','maker_id','sen_id','meas_date'] if c in raw]
            if 'source_sn' in raw:raw=raw.sort_values('source_sn')
            raw=raw.drop_duplicates(dedup,keep='last')
            raw['variable_id']=raw.sect_code.astype(str)+'__'+raw.fatr_code.astype(str)
            raw=raw.rename(columns={'sen_val':'sensor_value'})
            raw['sensor_value']=pd.to_numeric(raw.sensor_value,errors='coerce')
            for v,lo,hi in [('EI__TI',-10,60),('EI__HI',0,100),('EI__CI',0,5000)]:
                mask=raw.variable_id.eq(v);raw.loc[mask,'sensor_value']=raw.loc[mask,'sensor_value'].where(lambda x:x.between(lo,hi))
        else:raw=pd.read_parquet(AX/'strawberry_fruit_set_prediction/data/sensor_category_cache/sensor_collapsed.parquet')
        defs={c.rsplit('__',1)[0]:'active_fraction' in c for c in columns}
        extra=sensor_features(missing,raw,defs)
        cache=pd.concat([cache,extra],ignore_index=True)
    if crop=='strawberry' and 'EI__TI__7d_mean' not in columns:
        # Legacy FruitA temperature was not retained in the new category cache.
        raw=pd.read_parquet(AX/'strawberry_fruit_set_prediction/data/sensor_category_cache/sensor_collapsed.parquet')
        extra=sensor_features(rows[keys].drop_duplicates(),raw,{'EI__TI':False})
        cache=cache.merge(extra[keys+['EI__TI__7d_mean']],on=keys,how='outer',validate='one_to_one')
        columns.append('EI__TI__7d_mean')
    return rows.merge(cache,on=keys,how='left',validate='many_to_one'),columns

def legacy_feature(name,base):
    plain=name.split('::')[-1]
    aliases={'current_total_flower_count':base+'__current','previous_total_flower_count':'legacy_previous',
      'total_flower_delta':'legacy_delta','current_fruit_count':base+'__current',
      'fruit_count_delta':base+'__delta','week_of_year_sin':'week_sin','week_of_year_cos':'week_cos',
      'indoor_temperature_7d_mean':'EI__TI__7d_mean','indoor_humidity_7d_mean':'EI__HI__7d_mean',
      'indoor_co2_7d_mean':'EI__CI__7d_mean','sensor_EI__TI_7d_mean':'EI__TI__7d_mean'}
    if plain.startswith('fruit_count_roll'):return plain.replace('fruit_count_',base+'__',1)
    return aliases.get(plain,plain)

def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source_paths=[SOURCE/'csv/Group_Detail.csv', SOURCE/'csv/Experiments.csv',
      AX/'tomato_flower_count_prediction/data/raw/growth.parquet',
      AX/'tomato_flower_count_prediction/data/raw/environment.parquet',
      AX/'strawberry_fruit_set_prediction/data_audits/fruit_zero_semantics_20260812/artifacts/analysis_rows.csv',
      AX/'strawberry_fruit_set_prediction/data/sensor_category_cache/sensor_collapsed.parquet']
    source_paths += list((SOURCE/'prepared').glob('*/*.parquet'))+list((SOURCE/'prepared').glob('*/manifest.json'))
    source_paths += list((AX/'strawberry_fruit_set_prediction/source_processed_data').glob('*_candidates.parquet'))
    inputs={str(p):file_hash(p) for p in source_paths}
    freeze(output/'inputs.json',inputs)
    if (output/'Target_Status.csv').exists():
        for target in TARGETS:
            p=output/'prepared'/target
            spec=json.loads((p/'manifest.json').read_text())
            for split,h in spec['dataset_hashes'].items():
                if file_hash(p/f'{split}.parquet')!=h:raise ValueError('Prepared file changed; use a new output version')
        return pd.read_csv(output/'Target_Status.csv').to_dict('records')
    details=csv('Group_Detail');exps=csv('Experiments')
    maps=[];quality=[];status=[];catalog=[]
    for crop in ('tomato','strawberry'):
        rows,audits,provenance=observations(crop)
        rows,env=environmental_context(rows,crop)
        split=manifest('tomato_total_flower' if crop=='tomato' else 'strawberry_first')['split']
        if crop=='strawberry':
            for part in ('second','third'):
                if manifest('strawberry_'+part)['split']!=split:raise ValueError('Historical strawberry splits disagree')
        assigned={f:s for s,fs in split.items() for f in fs}
        if len(assigned)!=sum(map(len,split.values())):raise ValueError('Overlapping historical facilities')
        rows['split']=rows.facility_id.map(assigned)
        quality.extend([dict(Crop=crop,**a) for a in audits])
        quality.append(dict(Crop=crop,Reason='unassigned_facility_rows',Rows=int(rows.split.isna().sum())))
        rows=rows[rows.split.notna()].copy()
        oldgroups=exps[exps.Target_ID.str.startswith(crop)].Variable_Group_ID.unique()
        for part,i in PARTS.items():
            target=f'{crop}_{part}';base='flower_count' if crop=='tomato' else 'fruit_count'
            d=rows.copy();d[base]=d['count_sum123' if i is None else f'count_{i}']
            bases=[base]+(['current_max_truss_number','positive_truss_count','observed_truss_count'] if crop=='tomato' else [])
            d=growth_features(d,ENTITY,bases)
            d['days_since_crop_start']=(d.feature_date-d.cropping_date).dt.days
            positive=d.feature_date.where(d[base]>0).groupby([d[k] for k in ENTITY]).cummin().groupby([d[k] for k in ENTITY]).ffill()
            d['days_since_first_positive']=(d.feature_date-positive).dt.days
            d['legacy_previous']=d[base+'__lag1'].fillna(d[base+'__current'])
            d['legacy_delta']=d[base+'__current']-d.legacy_previous
            d['target']=d[[f'target_truss_{j}' for j in (1,2,3)]].sum(axis=1,min_count=3) if i is None else d[f'target_truss_{i}']
            d['row_id']=[digest([target,*v])[:24] for v in d[ENTITY+['feature_date','target_date']].itertuples(index=False,name=None)]
            reasons=np.select([d.target_date.isna(),d.target.isna(),d[base].isna()],['no_next_observation','target_missing','current_target_missing'],default='eligible')
            d['exclusion_reason']=reasons
            dest=output/'prepared'/target;dest.mkdir(parents=True,exist_ok=True)
            d.loc[reasons!='eligible',ENTITY+['feature_date','target_date','exclusion_reason']].to_csv(dest/'excluded_rows.csv',index=False)
            d=d[reasons=='eligible'].copy();train=d[d.split.eq('train')]
            groups={};group_maps=[]
            for old in sorted(oldgroups):
                subset=details[details.Variable_Group_ID.eq(old)]
                cats={};mapping=[]
                for cid,vs in subset.groupby('Category_ID'):
                    fs=sorted(set(vs.Variable_ID.astype(str)))
                    names=[]
                    for f in fs:
                        if '::' in f:
                            new='legacy__'+f.split('::')[-1];src=legacy_feature(f,base)
                            if src in d:d[new]=d[src]
                            if src=='days_since_previous' and crop=='tomato':d[new]=d[src].fillna(0)
                        else:new=f
                        names.append(new);mapping.append((cid,f,new))
                    cats['CAT_V2_'+digest([target,cid,names])[:12]]=names
                features=sorted({f for fs in cats.values() for f in fs})
                gid='VG_'+('Flower' if crop=='tomato' else 'Fruit')+'_'+part+'_'+digest(features)[:8]+'_V2'
                absent=[f for f in features if f not in d or not pd.to_numeric(d[f],errors='coerce').replace([np.inf,-np.inf],np.nan).loc[train.index].notna().any()]
                reason='missing_or_empty_required_inputs:'+','.join(absent) if absent else ''
                if not features:reason='historical_group_has_no_definition'
                record=dict(Old_Variable_Group_ID=old,Target_ID=target,Variable_Group_ID=gid,Status='unavailable' if reason else 'available',Reason=reason,Feature_Count=len(features))
                maps.append(record);group_maps.append(record)
                if not reason:
                    existing=groups.get(gid)
                    if existing and existing['features']!=features:raise ValueError('Group collision')
                    groups[gid]=dict(features=features,categories=cats)
                for cid,oldf,newf in mapping:catalog.append(dict(Target_ID=target,Variable_Group_ID=gid,Old_Category_ID=cid,Old_Variable_ID=oldf,Variable_ID=newf,Definition_Version=2,Adaptation='target history remapped; environment uses frozen [t-days,t) cache'))
            # Fixed target cohort, independent of feature availability and model.
            for s in split:
                frame=d[d.split.eq(s)].sort_values('row_id').reset_index(drop=True)
                if frame.empty:raise ValueError(f'{target}: empty {s}')
                frame.to_parquet(dest/f'{s}.parquet',index=False)
                quality.append(dict(Crop=crop,Target_ID=target,Split=s,Reason='target_distribution',Rows=len(frame),Zero_Rate=float(frame.target.eq(0).mean()),Target_Max=float(frame.target.max()),Target_Mean=float(frame.target.mean()),Target_Std=float(frame.target.std(ddof=0))))
            spec=dict(target_id=target,crop=crop,part=part,components=[1,2,3] if i is None else [i],
              entity=ENTITY,current=base+'__current',groups=groups,split=split,provenance=provenance,
              source_group_hash=file_hash(SOURCE/'csv/Group_Detail.csv'),
              dataset_hashes={s:file_hash(dest/f'{s}.parquet') for s in split})
            freeze(dest/'manifest.json',spec)
            status.append(dict(Target_ID=target,Status='prepared',Group_Count=len(groups),Unavailable_Mappings=sum(r['Status']=='unavailable' for r in group_maps)))
            print(target,len(d),'rows',len(groups),'groups',flush=True)
    for name,records in [('Group_Mapping',maps),('Group_Detail',catalog),('Data_Quality',quality),('Target_Status',status)]:pd.DataFrame(records).to_csv(output/f'{name}.csv',index=False)
    freeze(output/'source_inventory.json',{'original_csv_hashes':{str(p.relative_to(SOURCE)):file_hash(p) for p in (SOURCE/'timeseries').glob('*observed_counts_by_crop_cycle.csv')}})
    return status
