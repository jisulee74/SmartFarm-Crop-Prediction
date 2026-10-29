"""Prepare a frozen pooled-truss cohort; no future survey values enter features."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from AXData.ax_catalog_experiments.exhaustive.prepare import _assign_crop
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'ax_catalog_experiments/outputs/upper_truss_screening_20261020_v2';SOURCE=ROOT/'diagnostics/20261020_upper_truss_audit';OLD=ROOT/'ax_catalog_experiments/outputs/exhaustive_e6_semantic_screening_v1/prepared';SNAP=ROOT/'ax_catalog_experiments/outputs/exhaustive_e1_e7_v1/data_snapshot'
ENTITY=['facility_id','crop_sn','sample_num'];SERIES=ENTITY+['truss'];KEY=ENTITY+['feature_date']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(output=None, strawberry_limit=7):
 OUT=Path(output) if output is not None else globals()['OUT']
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'manifest.json').exists():
  m=json.loads((OUT/'manifest.json').read_text())
  for c,spec in m['crops'].items():
   for split,h in spec['hashes'].items():assert digest(OUT/c/(split+'.parquet'))==h
  return m
 m={'seed':42,'method':'4 fixed input sets x 6 models, one seed, no CV/tuning','test_notice':'Previously inspected facility holdout; not independent external validation','cohort':'same-truss next scheduled source survey, 1..14 days, current and next nonmissing','prediction':'clip lower bound 0 only; no flower-maximum upper cap','selection':'validation pooled RMSE then MAE; per-model input winner frozen before test','crops':{}}
 names=pd.read_csv(ROOT/'diagnostics/20261020_flower_cap_audit/verified_code_names.csv',dtype=str);crops=pd.read_parquet(SNAP/'sfkr_pvsn_crop.parquet')
 for crop,limit,pattern in [('tomato',20,r'^화방개화수\d+$'),('strawberry',strawberry_limit,r'^화방착과수\d+$')]:
  p=SOURCE/f'{crop}_live.parquet';r=pd.read_parquet(p);r['_join_user']=r.facility_id.str.rsplit('_',n=1).str[0];r,qa=_assign_crop(r,crops,'_join_user','examin_date');r=r.rename(columns={'sample_number':'sample_num','examin_date':'feature_date'});r['sample_num']=r.sample_num.astype(str);r['crop_sn']=r.crop_sn.astype(str);r.feature_date=pd.to_datetime(r.feature_date)
  code=names[names.examin_iem_nm.str.match(pattern)].drop_duplicates('examin_iem_code');mapping=code.set_index('examin_iem_code').examin_iem_nm.str.extract(r'(\d+)$')[0].astype(int).to_dict();r['truss']=r.growth_measure_code.map(mapping);r=r[r.truss.between(1,limit)].copy();r.truss=r.truss.astype(int);r['value']=pd.to_numeric(r.growth_measure_value,errors='coerce').where(lambda x:x>=0)
  conflicts=r.groupby(KEY+['truss']).value.nunique();bad=conflicts[conflicts>1].reset_index()[KEY+['truss']];r=r.merge(bad.assign(_bad=True),on=KEY+['truss'],how='left');r.loc[r._bad.eq(True),'value']=np.nan
  r=r.sort_values('sn').drop_duplicates(KEY+['truss']);wide=r.pivot(index=KEY,columns='truss',values='value').reindex(columns=range(1,limit+1));calendar=wide.reset_index()[KEY];dates=calendar.merge(r[KEY+['cropping_date']].drop_duplicates(KEY),on=KEY,validate='one_to_one');dates['days_since_crop_start']=(dates.feature_date-pd.to_datetime(dates.cropping_date)).dt.days
  wide.columns=[f'v{i}' for i in range(1,limit+1)];base=wide.reset_index().merge(dates,on=KEY,validate='one_to_one');vals=base[[f'v{i}' for i in range(1,limit+1)]];positive=vals.gt(0).to_numpy();nums=np.arange(1,limit+1);base['observed_trusses']=vals.notna().sum(axis=1);base['positive_trusses']=positive.sum(axis=1);base['positive_min']=np.where(positive.any(axis=1),np.where(positive,nums,np.inf).min(axis=1),np.nan);base['positive_max']=np.where(positive.any(axis=1),np.where(positive,nums,-np.inf).max(axis=1),np.nan);base['observed_total']=vals.sum(axis=1,min_count=1);base['weighted_position']=(vals.mul(nums).sum(axis=1,min_count=1)/base.observed_total.where(base.observed_total>0));base=base.sort_values(KEY)
  bg=base.groupby(ENTITY,sort=False);base['position_previous']=bg.weighted_position.shift(1);base['position_movement']=base.weighted_position-base.position_previous
  frames=[]
  for t in range(1,limit+1):
   d=base.drop(columns=[f'v{i}' for i in range(1,limit+1)]).copy();d['truss']=t;d['current']=base[f'v{t}'];d['lower_neighbor']=base[f'v{t-1}'] if t>1 else np.nan;d['upper_neighbor']=base[f'v{t+1}'] if t<limit else np.nan;frames.append(d)
  d=pd.concat(frames,ignore_index=True).sort_values(SERIES+['feature_date']);g=d.groupby(SERIES,sort=False);d['target']=g.current.shift(-1);d['target_date']=g.feature_date.shift(-1);d['gap_days']=(d.target_date-d.feature_date).dt.days;d['days_since_previous']=(d.feature_date-g.feature_date.shift(1)).dt.days
  for lag in [1,2,3]:d[f'lag{lag}']=g.current.shift(lag)
  d['delta']=d.current-d.lag1;d['roll3_mean']=g.current.transform(lambda x:x.rolling(3,min_periods=1).mean());d['past_max']=g.current.transform(lambda x:x.cummax().ffill());d['past_observations']=g.current.transform(lambda x:x.notna().cumsum())
  # First positive observed so far; never backfill a future first event.
  first=d.feature_date.where(d.current.gt(0));d['first_positive_date']=first.groupby([d[k] for k in SERIES]).transform(lambda x:x.where(x.notna()&~x.notna().cummax().shift(fill_value=False)).ffill())
  d['days_since_first_positive']=(d.feature_date-d.first_positive_date).dt.days;d['relative_position']=d.truss-d.weighted_position;d['relative_max']=d.truss-d.positive_max
  spec=json.loads((OLD/f'{crop}_first/manifest.json').read_text());splitmap={f:s for s,fs in spec['split'].items() for f in fs};d['split']=d.facility_id.map(splitmap)
  # Reuse only contemporaneous vegetation and past-window sensors, never old target/history columns.
  veg=[x for x in spec['groups']['E2'] if x.endswith('__current')];env=[x for k in ['E3','E4','E5','E6'] for x in spec['groups'][k] if x.endswith('7d_mean') or x.endswith('7d_active_fraction') or ('__7d__' in x and (x.endswith('_mean') or x.endswith('connection_rate')))];ctx=[x for x in spec['groups']['E7'] if x not in ['days_since_crop_start','item_code','cropping_system','cropping_season_name']];external=sorted(set(veg+env+ctx));cache=[]
  for part in ['first','second','third','sum123']:
   for split in ['train','validation','test']:
    old=pd.read_parquet(OLD/f'{crop}_{part}/{split}.parquet');cols=[x for x in external if x in old];old=old[KEY+cols].copy();old.crop_sn=old.crop_sn.astype(str);old.sample_num=old.sample_num.astype(str);cache.append(old)
  cache=pd.concat(cache,ignore_index=True);variations=cache.groupby(KEY)[external].nunique(dropna=True);assert variations.max().max()<=1,'Conflicting frozen context values';cache=cache.groupby(KEY)[external].first().reset_index();d=d.merge(cache,on=KEY,how='left',validate='many_to_one');d['context_available']=d[external].notna().any(axis=1).astype(int)
  d=d[d.current.notna()&d.target.notna()&d.gap_days.between(1,14)&d.split.notna()].copy();d['row_id']=d[SERIES+['feature_date']].astype(str).agg('|'.join,axis=1);assert d.row_id.is_unique;assert (d.feature_date<d.target_date).all()
  local=['current','truss','days_since_crop_start','days_since_previous','lag1','lag2','lag3','delta','roll3_mean','past_max','past_observations','days_since_first_positive'];position=['lower_neighbor','upper_neighbor','observed_trusses','positive_trusses','positive_min','positive_max','observed_total','weighted_position','position_previous','position_movement','relative_position','relative_max'];groups={'A_history':local,'B_position':local+position,'C_vegetative':local+position+veg,'D_full':local+position+veg+env+ctx+['context_available']}
  # Registered train-only validity screening. No test-dependent feature choice.
  train=d[d.split=='train'];selected={};excluded={}
  for name,features in groups.items():
   keep=[];reasons=[]
   for f in dict.fromkeys(features):
    v=pd.to_numeric(train[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
    if v.isna().mean()>.5 or v.nunique()<2:reasons.append({'feature':f,'missing':float(v.isna().mean()),'unique':int(v.nunique())})
    else:keep.append(f)
   selected[name]=keep;excluded[name]=reasons
  out=OUT/crop;out.mkdir(exist_ok=True);hashes={};counts={}
  for split in ['train','validation','test']:
   x=d[d.split==split].reset_index(drop=True);x.to_parquet(out/f'{split}.parquet',index=False);hashes[split]=digest(out/f'{split}.parquet');counts[split]={'rows':len(x),'positive':int(x.target.gt(0).sum()),'facilities':int(x.facility_id.nunique()),'context_available_rate':float(x.context_available.mean()),'trusses':int(x.truss.nunique())}
  d.groupby(['split','truss']).agg(n=('target','size'),positive=('target',lambda x:int(x.gt(0).sum()))).to_csv(out/'cohort.csv');cs={'limit':limit,'source_hash':digest(p),'split':spec['split'],'features':selected,'exclusions':excluded,'hashes':hashes,'counts':counts,'assignment':qa,'conflicting_keys':len(bad),'entity':SERIES,'context_policy':'exact-date frozen context only, missing retained and train-median imputed'};m['crops'][crop]=cs;print(crop,counts,{k:len(v) for k,v in selected.items()},flush=True)
 (OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));return m
if __name__=='__main__':prepare()
