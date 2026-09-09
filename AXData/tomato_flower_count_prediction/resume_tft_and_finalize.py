#!/usr/bin/env python3
"""Resume the interrupted run from saved five-model validation artifacts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent))
from common_regression.metrics import regression_metrics
from common_regression.models import make_adapter
from common_regression.evaluation import write_json
from pipeline import HISTORY_FEATURES, ENV_SPECS, CONTROL_CODES
from run_pipeline import _grouped_metrics
from tft_candidate_gpu import tune, validate, fit_predict

OUT=ROOT/'artifacts/total_flower_count_v1_sample_key_retry3'
ENTITY=['facility_id','crop_sn','sample_num']; TARGET='next_total_flower_count'; CURRENT='current_total_flower_count'

def choose(t):
    x=t[t.is_learning_model]; x=x[x.rmse.le(x.rmse.min()*1.01)]; x=x[x.mae.le(x.mae.min()*1.01)]
    return str(x.sort_values(['inference_latency_ms','model_size_bytes','model']).iloc[0].model)

def main():
    frames={s:pd.read_parquet(OUT/f'processed_data/{s}.parquet') for s in ('train','validation','test')}
    fs=json.loads((OUT/'feature_selection/feature_sets.json').read_text()); selected=fs['selected']; features=fs['sets'][selected]
    cfg=yaml.safe_load((ROOT/'config/model_suite.yaml').read_text()); md=OUT/'model_selection'
    params,epochs=tune(frames['train'],TARGET,'target_date',features,ENTITY,CURRENT,cfg['feature_selection_seeds'],md)
    metric=validate(frames['train'],frames['validation'],TARGET,features,ENTITY,CURRENT,params,epochs,cfg['validation_seeds'])
    table=pd.read_csv(md/'validation/model_metrics.csv'); table=pd.concat([table,pd.DataFrame([metric])],ignore_index=True)
    table.to_csv(md/'validation/model_metrics.csv',index=False); winner=choose(table)
    combined=pd.concat([frames['train'],frames['validation']],ignore_index=True); test=frames['test']; seeds=cfg['validation_seeds'] if winner in {'mlp','tabm','tft'} else [cfg['validation_seeds'][0]]
    if winner=='tft': pred=fit_predict(frames['train'],frames['validation'],test,TARGET,features,ENTITY,CURRENT,params,epochs,seeds)
    else:
        saved=json.loads((md/'internal_cv/hyperparameter_tuning/selected_params.json').read_text()); preds=[]
        for seed in seeds:
            adapter=make_adapter(winner,features,[],saved['params'][winner],seed)
            adapter.fit(combined,TARGET,fixed_epochs=saved['fixed_epochs'][winner]); preds.append(adapter.predict(test))
        pred=np.mean(preds,axis=0)
    fm=regression_metrics(test[TARGET],pred); pm=regression_metrics(test[TARGET],test[CURRENT]); keys=ENTITY+['target_date']
    f=test[keys+[TARGET,CURRENT]].copy(); f['prediction']=pred; f.to_parquet(md/'final_test_predictions.parquet',index=False)
    p=test[keys+[TARGET,CURRENT]].copy(); p['prediction']=test[CURRENT]; p.to_parquet(md/'persistence_test_predictions.parquet',index=False)
    write_json(md/'final_test_metrics.json',fm); write_json(md/'persistence_test_metrics.json',pm); write_json(md/'test_access_log.json',{'access_count':1,'reason':'configuration frozen'})
    result={'final':fm,'persistence':pm,'baseline_beaten_on_test':fm['rmse']<pm['rmse']}; _grouped_metrics(OUT,test)
    pre=json.loads((OUT/'preprocessing_manifest.json').read_text())
    manifest={'selected_feature_set':selected,'selected_features':features,'selected_learning_model':winner,'candidate_models':['poisson','random_forest','catboost','mlp','tabm','tft'],'crop_policy':pre['crop_policy'],'sensor_population_policy':'quality_audit_only','tft_params':params,'tft_fixed_epochs':epochs,'test':result,'test_access_count':1}
    write_json(OUT/'final_run_manifest.json',manifest)
    lines=['# 토마토 다음 조사 전체 꽃수 예측 결과','',f'- 선택 변수군: `{selected}`',f'- 선택 학습모델: `{winner}`',f"- 시설 분할 행 수: Train {len(frames['train'])}, Validation {len(frames['validation'])}, Test {len(test)}",'', '## Validation','',table[['model','rmse','mae','r2','ccc']].to_markdown(index=False,floatfmt='.4f'),'', '## 최종 Test','',pd.DataFrame([{'model':winner,**fm},{'model':'persistence',**pm}]).to_markdown(index=False,floatfmt='.4f'),'',f"- Persistence 대비 Test RMSE 개선 여부: {result['baseline_beaten_on_test']}"]
    (OUT/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(OUT)
if __name__=='__main__': main()
