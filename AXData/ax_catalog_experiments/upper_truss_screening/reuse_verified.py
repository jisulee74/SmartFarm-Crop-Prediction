"""Reuse completed A/B/C validation runs only after exact input/row equality."""
from pathlib import Path
import sys,json,shutil
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from AXData.ax_catalog_experiments.upper_truss_screening.prepare import OUT
old=OUT.parent/'upper_truss_screening_20261020_v1';a=json.loads((old/'manifest.json').read_text());b=json.loads((OUT/'manifest.json').read_text());reused=[]
for p in (old/'runs').glob('*.json'):
 r=json.loads(p.read_text());crop=r['crop'];cfg=r['config']
 if cfg=='D_full' or r['stage']!='validation' or r['status']!='success':continue
 features=a['crops'][crop]['features'][cfg];assert features==b['crops'][crop]['features'][cfg]
 cols=list(dict.fromkeys(['row_id','facility_id','crop_sn','sample_num','truss','feature_date','target_date','target',*features]))
 for split in ['train','validation']:
  x=pd.read_parquet(old/crop/(split+'.parquet'))[cols];y=pd.read_parquet(OUT/crop/(split+'.parquet'))[cols];pd.testing.assert_frame_equal(x,y)
 for folder,extension in [('runs','.json'),('predictions','.parquet'),('logs','.log')]:
  source=old/folder/(p.stem+extension);dest=OUT/folder/source.name;dest.parent.mkdir(exist_ok=True)
  if source.exists():shutil.copy2(source,dest)
 reused.append(p.stem)
(OUT/'reused_runs.json').write_text(json.dumps({'from':str(old),'checks':'exact feature list and train/validation model input, target, identity, date equality','runs':reused},indent=2));print('Verified reusable runs:',len(reused))
