"""Persistent presentation IDs; experiment artifacts retain canonical identities."""
import json
from pathlib import Path
import pandas as pd


def simplify_ids(tables, output):
    path=Path(output)/'display_id_registry.json'
    registry=json.loads(path.read_text()) if path.exists() else {}
    fields={'Experiment_ID':'experiment','Variable_Group_ID':'group','Base_Group_ID':'group','Added_Group_ID':'group','Run_ID':'run','Category_ID':'category','Source_Category_ID':'category'}
    group_types={}
    for r in tables['Experiments'].itertuples():
        kind='Flower' if r.Target_ID.startswith('tomato') else 'Fruit'
        if r.Variable_Group_ID in group_types and group_types[r.Variable_Group_ID]!=kind:
            raise ValueError('Group shared across incompatible target families')
        group_types[r.Variable_Group_ID]=kind
    for column,namespace in fields.items():
        mapping=registry.setdefault(namespace,{})
        ids=set()
        for frame in tables.values():
            if column in frame:ids.update(frame[column].dropna().astype(str))
        for original in sorted(ids):
            if original in mapping:continue
            prefix={'experiment':'EXP','run':'RUN','category':'CAT'}.get(namespace)
            if namespace=='group':
                if original not in group_types:raise ValueError(f'Unknown group family: {original}')
                prefix='VG_'+group_types[original]
            numbers=[int(v.rsplit('_',1)[1]) for v in mapping.values() if v.startswith(prefix+'_')]
            mapping[original]=f'{prefix}_{max(numbers,default=0)+1:03d}'
    for namespace,mapping in registry.items():
        if len(set(mapping.values()))!=len(mapping):raise ValueError(f'Duplicate display IDs: {namespace}')
    for frame in tables.values():
        for column,namespace in fields.items():
            if column in frame:
                frame[column]=frame[column].map(lambda x:registry[namespace][str(x)] if pd.notna(x) else x)
    rows=[{'ID_Type':namespace,'Display_ID':short,'Original_ID':original} for namespace,m in registry.items() for original,short in m.items()]
    tables['ID_Mapping']=pd.DataFrame(rows)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(registry,ensure_ascii=False,indent=2));tmp.replace(path)
