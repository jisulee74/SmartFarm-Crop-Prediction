#!/usr/bin/env python3
"""Apply the reviewed facility/crop/date exclusions and rerun with GPU TFT."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

import run_filtered_with_tft as base
from pipeline import split_and_filter_features as original_split
from run_pipeline import _eligible_sensor_facilities
from tft_candidate_gpu import fit_predict, tune, validate

ROOT=Path(__file__).resolve().parent

def reviewed_policy(raw):
    data=raw.copy(); data['examin_date']=pd.to_datetime(data.examin_date)
    f=data.facility_id.astype(str)
    masks={
      'PF_0025108_crop205_after_2025_05_31': f.eq('PF_0025108_01') & data.crop_sn.eq(205) & data.examin_date.gt('2025-05-31'),
      'PF_0025108_crop207': f.eq('PF_0025108_01') & data.crop_sn.eq(207),
      'PF_0026456_all': f.eq('PF_0026456_01'),
      'PF_0024696_crop176': f.eq('PF_0024696_01') & data.crop_sn.eq(176),
      'PF_0024697_crop178_2025_09_24': f.eq('PF_0024697_01') & data.crop_sn.eq(178) & data.examin_date.eq(pd.Timestamp('2025-09-24')),
    }
    remove=pd.Series(False,index=data.index)
    for mask in masks.values(): remove |= mask
    audit={'rules':list(masks),'removed_raw_rows':{k:int(v.sum()) for k,v in masks.items()},'total_removed_raw_rows':int(remove.sum())}
    return data.loc[~remove].copy(),audit

def eligible_split(data):
    eligible,_=_eligible_sensor_facilities(data)
    return original_split(eligible)

def main():
    base.crop_policy=reviewed_policy
    base.split_and_filter_features=eligible_split
    base.tune_tft=tune; base.validate_tft=validate; base.fit_predict_tft=fit_predict
    output=base.run()
    pre_path=output/'preprocessing_manifest.json'; pre=json.loads(pre_path.read_text())
    pre['sensor_population_policy']='complete TI/HI/CI ratio >= 0.50 per facility-crop'
    pre_path.write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n')
    manifest_path=output/'final_run_manifest.json'; manifest=json.loads(manifest_path.read_text())
    manifest['sensor_population_policy']='complete TI/HI/CI ratio >= 0.50 per facility-crop'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(output)
if __name__=='__main__': main()
