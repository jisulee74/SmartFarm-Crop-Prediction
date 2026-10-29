from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from AXData.ax_catalog_experiments.upper_truss_screening.prepare import OUT,SOURCE
m=json.loads((OUT/'manifest.json').read_text());audit=pd.read_parquet(SOURCE/'next_observation_pairs.parquet')
for crop,spec in m['crops'].items():
 sets=[set(fs) for fs in spec['split'].values()];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
 frames=[]
 for split in spec['split']:
  d=pd.read_parquet(OUT/crop/(split+'.parquet'));assert d.row_id.is_unique;assert d.target.notna().all() and d.current.notna().all();assert d.gap_days.between(1,14).all();assert (d.feature_date<d.target_date).all();assert d.past_max.ge(d.current).all();assert d.days_since_first_positive.dropna().ge(0).all();assert set(d.truss)==set(range(1,spec['limit']+1));assert set(d.facility_id)==set(spec['split'][split]);frames.append(d)
  for f in spec['features'].values():assert not set(f)&{'target','target_date','gap_days','row_id','facility_id','crop_sn','sample_num','split'}
 d=pd.concat(frames);a=audit[audit.crop.eq(crop)].rename(columns={'sample_number':'sample_num','examin_date':'feature_date','next_date':'target_date','value':'source_current','next_value':'source_target'})
 for k in ['crop_sn','sample_num']:a[k]=a[k].astype(str)
 keys=['facility_id','crop_sn','sample_num','truss','feature_date','target_date'];joined=d.merge(a[keys+['source_current','source_target']],on=keys,validate='one_to_one');assert len(joined)==len(d);assert np.array_equal(joined.target,joined.source_target);assert np.array_equal(joined.current,joined.source_current)
 print(crop,len(d),'rows: source target/current match; no split overlap; next-survey timing and feature allowlist passed')
