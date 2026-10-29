"""Read-only integrity checks, with a separate verification record."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import TARGETS,PARTS,ENTITY,SOURCE,observations,manifest
from ..core import file_hash,digest
from .engine import atomic
from .policy import score

def verify(root):
    root=Path(root);checks=[];mapping=pd.read_csv(root/'Group_Mapping.csv')
    old=pd.read_csv(SOURCE/'csv/Experiments.csv')
    source={crop:observations(crop)[0] for crop in ['tomato','strawberry']}
    manifests={}
    for target,(crop,part) in TARGETS.items():
        p=root/'prepared'/target;s=json.loads((p/'manifest.json').read_text());manifests[target]=file_hash(p/'manifest.json')
        all_rows=[];facilities={}
        for split,h in s['dataset_hashes'].items():
            assert file_hash(p/f'{split}.parquet')==h,(target,split,'file hash')
            d=pd.read_parquet(p/f'{split}.parquet');assert not d.row_id.duplicated().any()
            assert (d.feature_date<d.target_date).all();assert d.target.notna().all()
            facilities[split]=set(d.facility_id)
            assert facilities[split]<=set(s['split'][split])
            all_rows.append(d)
        assert not (facilities['train']&facilities['validation'] or facilities['train']&facilities['test'] or facilities['validation']&facilities['test'])
        assert s['split']==manifest('tomato_total_flower' if crop=='tomato' else 'strawberry_first')['split']
        d=pd.concat(all_rows,ignore_index=True)
        keys=ENTITY+['feature_date'];raw=source[crop]
        joined=d.merge(raw[keys+['target_date','target_truss_1','target_truss_2','target_truss_3']],on=keys,suffixes=('','_source'),validate='one_to_one')
        assert len(joined)==len(d) and (joined.target_date==joined.target_date_source).all()
        i=PARTS[part]
        actual=joined[[f'target_truss_{j}_source' for j in [1,2,3]]].sum(axis=1,min_count=3) if i is None else joined[f'target_truss_{i}_source']
        assert np.allclose(joined.target,actual,equal_nan=False)
        expected=set(old[old.Target_ID.str.startswith(crop)].Variable_Group_ID)
        assert set(mapping[mapping.Target_ID.eq(target)].Old_Variable_Group_ID)==expected
        for group in s['groups'].values():assert group['features']==sorted({f for fs in group['categories'].values() for f in fs})
        checks.append(dict(Target_ID=target,Rows=len(d),Groups=len(s['groups']),Status='passed'))
    preserved=json.loads((root/'source_inventory.json').read_text())['original_csv_hashes']
    for path,h in preserved.items():assert file_hash(SOURCE/path)==h
    # At least one representative per model/evaluation stage; metrics recomputed from row predictions.
    selected={}
    for p in (root/'experiments').glob('*/runs/*/result.json'):
        r=json.loads(p.read_text())
        if 'Metrics' in r:selected[(r['Model_Type'],r['Stage'])]=(p,r)
    for p,r in selected.values():
        d=pd.read_parquet(p.parent/'predictions.parquet')
        for v in ['raw','bounded']:
            for k,value in score(d.target,d['prediction_'+v]).items():
                old_value=r['Metrics'][v][k]
                assert value==old_value if value is None or old_value is None else np.isclose(value,old_value,rtol=1e-10,atol=1e-12)
        assert digest(sorted(d.row_id.tolist()))==r['Eval_Row_Hash']
    result=dict(Status='passed',Targets=checks,Prediction_Files_Recomputed=len(selected),Prepared_Manifest_Hashes=manifests,Observation_CSVs_Preserved=True)
    atomic(root/'verification.json',result);return result
