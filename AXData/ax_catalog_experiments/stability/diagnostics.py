from __future__ import annotations
import json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from ..core import digest,freeze
from ..engine import make_adapter
from .data import SOURCE,manifest
from .engine import folds,atomic
from .policy import POLICY,StablePreprocessor,constrain,score
from .adapters import adapter

CASES=[('tomato_total_flower','전체 꽃 수','mlp'),('tomato_total_flower','전체 꽃 수','tabm'),
       ('tomato_total_flower','전체 꽃 수','poisson'),('strawberry_third','3화방 착과수','poisson')]

def diagnose(root):
    base=Path(root)/'diagnostics';out=base/digest(POLICY)[:12];out.mkdir(parents=True,exist_ok=True)
    results=pd.read_csv(SOURCE/'csv/Results.csv');mapping=pd.read_csv(SOURCE/'csv/ID_Mapping.csv')
    results.Metric_Value=pd.to_numeric(results.Metric_Value,errors='coerce')
    records=[];gate=[]
    import torch
    torch.set_num_threads(4)
    for target,label,model in CASES:
        case=out/f'{target}_{model}';case.mkdir(exist_ok=True)
        old=results[(results.Prediction_Target==label)&(results.Model_Type==model)&(results.Metric_Name=='R2')&(results.Result_Origin=='new')].nsmallest(1,'Metric_Value').iloc[0]
        original=mapping.loc[mapping.Display_ID.eq(old.Run_ID),'Original_ID'].iloc[0]
        source=SOURCE/'experiments'/target/'full/runs'/original
        record=json.loads((source/'result.json').read_text());identity=json.loads((source/'identity.json').read_text())
        train=pd.read_parquet(SOURCE/'prepared'/target/'train.parquet')
        fit,val=next((f,v) for n,f,v in folds(train) if n==record['Fold_ID'])
        if digest(sorted(fit.row_id.tolist()))!=identity['fit_rows'] or digest(sorted(val.row_id.tolist()))!=identity['eval_rows']:raise ValueError('Historical fold mismatch')
        features=record['Features'];spec=manifest(target);params=record['Params'];seed=record['Seed']
        upper=float(fit.target.max());prep=StablePreprocessor(features).fit(fit)
        audit=prep.diagnostics(val)
        from ...common_regression.preprocessing import TabularPreprocessor
        oldprep=TabularPreprocessor(features,[],True).fit(fit)
        audit['Legacy_Transformed_Abs_Max']=np.abs(oldprep.transform(val)).max(axis=0)
        audit.to_csv(case/'input_diagnostics.csv',index=False)
        quality=pd.DataFrame([dict(Split=name,Rows=len(f),Zero_Rate=float(f.target.eq(0).mean()),Target_Variance=float(f.target.var(ddof=0)),Target_Max=float(f.target.max()),Duplicate_Rows=int(f.row_id.duplicated().sum()),Median_Interval_Days=float((f.target_date-f.feature_date).dt.days.median())) for name,f in [('fit',fit),('evaluation',val)]]);quality.to_csv(case/'data_quality.csv',index=False)
        archived=pd.read_parquet(source/'predictions.parquet')
        archived_score=score(archived.target,archived.prediction)
        if not np.isclose(archived_score['r2'],float(old.Metric_Value),rtol=1e-6):raise ValueError('Archived metric mismatch')
        freeze(case/'identity.json',dict(source_run=original,fit_rows=identity['fit_rows'],eval_rows=identity['eval_rows'],policy_hash=digest(POLICY),params=params,seed=seed))
        for variant in ['legacy','stable']:
            dest=case/f'{variant}.json'
            if dest.exists():r=json.loads(dest.read_text())
            else:
                start=time.monotonic();r=dict(Target_ID=target,Model_Type=model,Variant=variant,Source_Run_ID=old.Run_ID,Status='failed')
                try:
                    a=make_adapter(model,features,[],params,seed) if variant=='legacy' else adapter(model,features,params,seed,spec['entity'],spec['current'])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always',ConvergenceWarning)
                        r['Effective_Params']=a.params
                        a.fit(fit,'target',validation=val);raw=np.asarray(a.predict(val),float)
                    bounded,unstable=constrain(raw,upper)
                    # Legacy engine used nonnegative output for its archived scores.
                    reproduced=score(val.target,np.maximum(0,raw))
                    r.update(Status='unstable' if unstable else 'nonconverged' if any(issubclass(w.category,ConvergenceWarning) for w in caught) else 'success',
                      Raw_Abs_Max=float(np.abs(raw).max()),Upper_Bound=upper,
                      Raw_Metrics=score(val.target,raw),Bounded_Metrics=score(val.target,bounded),
                      Warnings=[str(w.message) for w in caught],Legacy_Reproduction_Relative_Error=float(abs(reproduced['rmse']-archived_score['rmse'])/max(archived_score['rmse'],1e-12)))
                    pred=val[['row_id','target']].copy();pred['prediction_raw']=raw;pred['prediction_bounded']=bounded;pred.to_parquet(case/f'{variant}_predictions.parquet',index=False)
                except Exception as e:r['Reason']=f'{type(e).__name__}: {e}'
                r['Duration_Seconds']=time.monotonic()-start;atomic(dest,r)
            records.append(r);print(target,model,variant,r['Status'],flush=True)
        old_r,new_r=records[-2:]
        passed=new_r['Status']=='success' and old_r.get('Legacy_Reproduction_Relative_Error',1)<.1
        gate.append(dict(Target_ID=target,Model_Type=model,Passed=passed,Reason='stable raw predictions and reproduced historical failure' if passed else 'Diagnostic reproduction/stability/convergence requires investigation'))
        atomic(base/'gate.json',dict(Status='running',Policy_Hash=digest(POLICY),Cases=gate))
    atomic(out/'comparison.json',records);atomic(base/'comparison.json',records)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]).to_csv(out/'Comparison.csv',index=False)
    outcome=dict(Status='passed' if all(g['Passed'] for g in gate) else 'blocked',Policy_Hash=digest(POLICY),Cases=gate)
    atomic(out/'gate.json',outcome);atomic(base/'gate.json',outcome);return outcome
