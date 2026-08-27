from __future__ import annotations
import importlib.util,json,re
from pathlib import Path
import pandas as pd
from common_regression.contracts import SplitBundle,TargetSpec

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT

def source_module():
    spec=importlib.util.spec_from_file_location("original_fruit_dataset",SOURCE/"original_dataset.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def load_dataset(truss: str):
    src=source_module(); base=ROOT/"processed_data"/truss
    raw={s:pd.read_parquet(base/f"{s}_candidates.parquet") for s in ("train","val","test")}
    metadata=pd.read_csv(SOURCE/"data/sensor_category_cache/sensor_metadata.csv")
    sensor=pd.read_parquet(SOURCE/"data/sensor_category_cache/sensor_collapsed.parquet"); sensor["meas_date"]=pd.to_datetime(sensor.meas_date)
    selected=src._select_sensor_variables(raw["train"],sensor,metadata)
    enriched={s:src._add_sensor_features(frame,sensor,selected) for s,frame in raw.items()}
    blocks=src._growth_blocks(set(enriched["train"].columns))
    blocks={name:[f for f in features if f!="fruit_column"] for name,features in blocks.items()}
    canonical=json.loads((SOURCE/"data/canonical_feature_groups_219.json").read_text())
    canonical_sensor=set(canonical["sensor_only"]["features"])-{"fruit_column"}
    sensor_features=[f for f in enriched["train"] if f in canonical_sensor]
    for section,block in (("NT","nutrient"),("EI","indoor"),("EO","outdoor"),("CR","control")):
        variables=set(metadata.loc[metadata.sect_code.eq(section),"variable_id"]).intersection(selected)
        blocks[block]=[f for f in sensor_features if any(f.startswith(f"sensor_{v}_") for v in variables)]
    records=[]
    for block,features in blocks.items():
        for feature in features:
            if feature.startswith("sensor_"):
                match=re.match(r"^sensor_(.+)_(?:1|3|7)d_",feature); source=match.group(1); indicator=feature.endswith(("_missing","_coverage")); direct=feature.endswith("_mean")
            else:
                source=feature.split("_roll")[0].replace("current_","").replace("previous_",""); indicator=feature.startswith("observed_"); direct=feature==source or feature.startswith("current_")
            records.append({"feature":feature,"block":block,"source_variable":source,"semantic_group":source,"is_quality_indicator":indicator,"inference_available":True,"is_direct":direct,"derivation_complexity":0 if direct else 1,"semantic_equivalent":True})
    exclusions=("next_fruit_count","target_sn","target_date","target_split","raw_split","split","sn","user_id","sample_num","crop_sn","feature_date","meas_date","fruit_count","fruit_column")
    spec=TargetSpec(f"fruit_set_{truss}","next_fruit_count","current_fruit_count","feature_date","target_date",("user_id","sample_num","crop_sn"),("user_id","sample_num","crop_sn","target_date"),exclusions,.95,{k:tuple(v) for k,v in blocks.items()},{"control":"no_control_variable_passed_new_train_quality_filter"} if not blocks["control"] else {})
    test_path=base/"test_candidates.parquet"
    return SplitBundle(enriched["train"],enriched["val"],test_path,spec,test_loader=lambda:enriched["test"]),blocks,pd.DataFrame(records),selected
