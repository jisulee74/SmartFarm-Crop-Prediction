from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"source_processed_data"
TRUSSES=("first_fruits_num","second_fruits_num","third_fruits_num")
COHORT_USERS=(
    "PF_0023621","PF_0002037","PF_0021351","PF_0022105","PF_0000574",
    "PF_0023732","PF_0023743","PF_0023923","PF_0024357","PF_0024365",
    "PF_0024688","PF_0024654","PF_0025106","PF_0024646","PF_0024657",
    "PF_0024700","PF_0025103","PF_0024651","PF_0024806",
)

def split_one(frame: pd.DataFrame):
    counts=frame.groupby("target_date").size().sort_index(); cumulative=counts.cumsum(); total=len(frame)
    train_end=cumulative.index[(cumulative.ge(total*.70)).argmax()]
    val_end=cumulative.index[(cumulative.ge(total*.85)).argmax()]
    labels=frame.target_date.map(lambda x:"train" if x<=train_end else "val" if x<=val_end else "test")
    return {name:frame.loc[labels.eq(name)].assign(target_split=name).sort_values("target_date",kind="stable").reset_index(drop=True) for name in ("train","val","test")},train_end,val_end

def run():
    all_data=pd.concat([pd.read_parquet(SOURCE/f"{s}_candidates.parquet") for s in ("train","val","test")],ignore_index=True)
    source_users=sorted(all_data.user_id.dropna().unique().tolist())
    missing_users=sorted(set(COHORT_USERS)-set(source_users))
    if missing_users:
        raise RuntimeError(f"Configured cohort users missing from source candidates: {missing_users}")
    all_data=all_data[all_data.user_id.isin(COHORT_USERS)].copy()
    if set(all_data.user_id.unique()) != set(COHORT_USERS):
        raise RuntimeError("Filtered candidates do not contain the exact configured 19-user cohort")
    all_data["target_date"]=pd.to_datetime(all_data.target_date); all_data["meas_date"]=pd.to_datetime(all_data.meas_date)
    manifest={"source":str(SOURCE),"cohort_rule":"2024+ users with valid all-three-truss fruit-set rows and CI/TI/HI sensor logs","cohort_user_count":len(COHORT_USERS),"cohort_users":list(COHORT_USERS),"source_user_count":len(source_users),"excluded_source_users":sorted(set(source_users)-set(COHORT_USERS)),"rule":"within each fruit_column, sort target_date; keep one date intact; choose cumulative-row cutoffs near 70/15/15","trusses":{}}
    for truss in TRUSSES:
        frame=all_data[all_data.fruit_column.eq(truss)].copy(); splits,train_end,val_end=split_one(frame)
        out=ROOT/"processed_data"/truss; out.mkdir(parents=True,exist_ok=True)
        summary={}
        for split,data in splits.items():
            data.to_parquet(out/f"{split}_candidates.parquet",index=False)
            summary[split]={"rows":len(data),"target_min":str(data.target_date.min().date()),"target_max":str(data.target_date.max().date()),"users":data.user_id.nunique(),"crops":data.crop_sn.nunique()}
        manifest["trusses"][truss]={"rows":len(frame),"train_cutoff":str(train_end.date()),"validation_cutoff":str(val_end.date()),"splits":summary}
    (ROOT/"processed_data"/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return manifest

if __name__=="__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
