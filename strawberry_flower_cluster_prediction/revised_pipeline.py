#!/usr/bin/env python3
"""Crop-cycle-aware modeling workflow with baselines and limited RF validation."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
SPLITS=("train","val","test")
KEYS=["user_id","crop_cycle_id","sample_num"]
BASE=["fruit_cluster_num","leaves_length","leaves_num","petiole_length","theca_diameter","first_flower_num","second_flower_num","third_flower_num","first_fruits_num","second_fruits_num","third_fruits_num"]
TIME=["days_since_crop_start","days_since_previous","week_of_year_sin","week_of_year_cos"]
COMPACT=[*BASE,*[f"{c}_daily_change" for c in BASE],*TIME]
EXTENDED=[*COMPACT,*[f"{c}_roll4_{s}" for s in ("mean","min","max","std") for c in BASE]]
RF_CONFIGS=[("RF-1",4,3),("RF-2",4,8),("RF-3",8,3),("RF-4",8,8)]

def metrics(y,p):
 y=np.asarray(y,float); p=np.asarray(p,float); ym=y.mean(); pm=p.mean(); den=y.var()+p.var()+(ym-pm)**2
 ccc=1.0 if den==0 and np.array_equal(y,p) else (None if den==0 else float(2*np.mean((y-ym)*(p-pm))/den))
 return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p)) if np.unique(y).size>1 else None,"ccc":ccc}

def load_raw(data_dir):
 frames={s:pd.read_parquet(data_dir/f"{s}.parquet").assign(raw_split=s) for s in SPLITS}
 columns=list(frames["train"].columns)
 if any(list(frames[s].columns)!=columns for s in SPLITS): raise ValueError("Raw schemas differ")
 data=pd.concat(frames.values(),ignore_index=True)
 for c in ("meas_date","crop_start_date","crop_end_date"): data[c]=pd.to_datetime(data[c],errors="coerce")
 for c in BASE: data[c]=pd.to_numeric(data[c],errors="coerce")
 required=[*KEYS,"meas_date","crop_start_date","crop_end_date",*BASE]
 if data[required].isna().any().any(): raise ValueError("Raw data contains invalid required values")
 if len(data)!=506 or data.groupby(KEYS).ngroups!=16: raise ValueError("Expected 506 rows and 16 entity-cycle series after decline-tail removal")
 if data.duplicated([*KEYS,"meas_date"]).any(): raise ValueError("Duplicate entity-cycle date")
 if ((data.meas_date<data.crop_start_date)|(data.meas_date>data.crop_end_date)).any(): raise ValueError("Measurement outside crop cycle")
 if data.groupby(["user_id","meas_date"]).raw_split.nunique().max()>1: raise ValueError("User/date crosses splits")
 return data.sort_values([*KEYS,"meas_date","sn"],kind="stable").reset_index(drop=True),frames

def engineer(raw):
 d=raw.copy(); g=d.groupby(KEYS,sort=False,group_keys=False); d["history_count"]=g.cumcount()+1; d["target_date"]=g.meas_date.shift(-1); d["next_fruit_cluster_num"]=g.fruit_cluster_num.shift(-1)
 previous=g.meas_date.shift(1); d["days_since_previous"]=(d.meas_date-previous).dt.total_seconds()/86400; d["days_since_crop_start"]=(d.meas_date-d.crop_start_date).dt.total_seconds()/86400
 week=d.meas_date.dt.isocalendar().week.astype(float); d["week_of_year_sin"]=np.sin(2*np.pi*week/52); d["week_of_year_cos"]=np.cos(2*np.pi*week/52)
 for c in BASE:
  d[f"{c}_daily_change"]=(d[c]-g[c].shift(1))/d.days_since_previous
  roll=g[c].rolling(4,min_periods=4)
  for stat in ("mean","min","max"): d[f"{c}_roll4_{stat}"]=getattr(roll,stat)().reset_index(level=KEYS,drop=True)
  d[f"{c}_roll4_std"]=roll.std(ddof=0).reset_index(level=KEYS,drop=True)
 d=d[d.history_count.ge(4)&d.target_date.notna()].reset_index(drop=True)
 expected=len(raw)-4*raw.groupby(KEYS).ngroups
 if len(d)!=expected or not np.isfinite(d[EXTENDED].to_numpy(float)).all(): raise ValueError(f"Expected {expected} finite prediction samples")
 return d

def assign_splits(features,raw):
 mapping=raw[["user_id","meas_date","raw_split"]].drop_duplicates().rename(columns={"meas_date":"target_date","raw_split":"target_split"})
 if mapping.duplicated(["user_id","target_date"]).any(): raise ValueError("Ambiguous target split")
 d=features.rename(columns={"raw_split":"input_raw_split"}).merge(mapping,on=["user_id","target_date"],how="left",validate="many_to_one")
 if d.target_split.isna().any() or d.groupby(["user_id","target_date"]).target_split.nunique().max()>1: raise ValueError("Invalid target split")
 return {s:d[d.target_split.eq(s)].reset_index(drop=True) for s in SPLITS}

def target_summary(frames):
 rows=[]; train=frames["train"].next_fruit_cluster_num; lo,hi=train.min(),train.max()
 for s,d in frames.items():
  y=d.next_fruit_cluster_num
  rows.append({"split":s,"rows":len(d),"mean":y.mean(),"std":y.std(ddof=0),"median":y.median(),"min":y.min(),"max":y.max(),"outside_train_range_count":int(((y<lo)|(y>hi)).sum()),"outside_train_range_ratio":float(((y<lo)|(y>hi)).mean())})
 return pd.DataFrame(rows)

def baseline_results(frames):
 train_mean=frames["train"].next_fruit_cluster_num.mean(); rows=[]; entity=[]
 for split in ("val","test"):
  d=frames[split]; y=d.next_fruit_cluster_num
  predictions={"train_mean":np.full(len(d),train_mean),"persistence":d.fruit_cluster_num.to_numpy(float),"split_mean_reference":np.full(len(d),y.mean())}
  for name,p in predictions.items(): rows.append({"split":split,"model":name,"operational":name!="split_mean_reference",**metrics(y,p)})
  for key,g in d.groupby(KEYS): entity.append({"split":split,"user_id":key[0],"crop_cycle_id":key[1],"sample_num":key[2],"rows":len(g),**metrics(g.next_fruit_cluster_num,g.fruit_cluster_num)})
 return pd.DataFrame(rows),pd.DataFrame(entity)

def audit_outputs(raw,frames,out):
 out.mkdir(parents=True,exist_ok=True); rows=[]; intervals=[]
 for key,g in raw.groupby(KEYS,sort=True):
  delta=g.meas_date.sort_values().diff().dt.days.dropna()
  rows.append({"user_id":key[0],"crop_cycle_id":key[1],"sample_num":key[2],"rows":len(g),"start_date":g.meas_date.min(),"end_date":g.meas_date.max()})
  intervals.append({"user_id":key[0],"crop_cycle_id":key[1],"sample_num":key[2],"interval_count":len(delta),"min_days":delta.min(),"median_days":delta.median(),"max_days":delta.max(),"nonpositive_count":int(delta.le(0).sum()),"over_60_days_count":int(delta.gt(60).sum())})
 pd.DataFrame(rows).to_csv(out/"entity_summary.csv",index=False); pd.DataFrame(intervals).to_csv(out/"interval_summary.csv",index=False)
 manifest={"passed":True,"rows":{"train":int((raw.raw_split=='train').sum()),"val":int((raw.raw_split=='val').sum()),"test":int((raw.raw_split=='test').sum()),"total":len(raw)},"entities":raw.groupby(KEYS).ngroups,"prediction_rows":sum(map(len,frames.values())),"checks":{"duplicate_entity_dates":int(raw.duplicated([*KEYS,'meas_date']).sum()),"null_required":0,"outside_crop_cycle":0,"user_date_split_overlap":0,"expected_sample_nums_per_user_cycle":bool(raw.groupby(['user_id','crop_cycle_id']).sample_num.apply(lambda x:set(x)=={1,2,3,4}).all())}}
 (out/"audit_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
 return manifest

def descriptive_spearman(train):
 return pd.DataFrame([{"feature":f,"spearman_rho":train[f].corr(train.next_fruit_cluster_num,method="spearman")} for f in EXTENDED]).sort_values("spearman_rho",key=lambda x:x.abs(),ascending=False)

def select_rf(frames):
 records=[]; models={}
 for set_name,features in (("compact",COMPACT),("extended",EXTENDED)):
  for name,depth,leaf in RF_CONFIGS:
   model=RandomForestRegressor(n_estimators=500,max_depth=depth,min_samples_leaf=leaf,max_features="sqrt",random_state=42,n_jobs=-1)
   model.fit(frames["train"][features],frames["train"].next_fruit_cluster_num); p=model.predict(frames["val"][features]); result={"feature_set":set_name,"config":name,"max_depth":depth,"min_samples_leaf":leaf,"feature_count":len(features),**metrics(frames["val"].next_fruit_cluster_num,p)}; records.append(result); models[(set_name,name)]=model
 table=pd.DataFrame(records); ranked=table.assign(_ccc=table.ccc.fillna(-np.inf),_compact=(table.feature_set!="compact").astype(int)).sort_values(["_ccc","rmse","_compact","max_depth"],ascending=[False,True,True,True],kind="stable")
 best=ranked.iloc[0].drop(labels=["_ccc","_compact"]).to_dict(); return table,best

def extrapolation(train,test,features):
 lo=train[features].min(); hi=train[features].max(); outside=test[features].lt(lo)|test[features].gt(hi)
 return {"cell_count":int(outside.sum().sum()),"cell_ratio":float(outside.to_numpy().mean()),"row_count":int(outside.any(axis=1).sum()),"row_ratio":float(outside.any(axis=1).mean())}

def final_fit_evaluate(frames,best):
 features=COMPACT if best["feature_set"]=="compact" else EXTENDED; combined=pd.concat([frames["train"],frames["val"]],ignore_index=True)
 model=RandomForestRegressor(n_estimators=500,max_depth=int(best["max_depth"]),min_samples_leaf=int(best["min_samples_leaf"]),max_features="sqrt",random_state=42,n_jobs=-1); model.fit(combined[features],combined.next_fruit_cluster_num)
 test=frames["test"].copy(); test["prediction"]=model.predict(test[features]); test["actual"]=test.next_fruit_cluster_num; test["residual"]=test.actual-test.prediction
 return model,features,test,metrics(test.actual,test.prediction),combined

def plots(test,model,features,out):
 out.mkdir(parents=True,exist_ok=True)
 fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(test.actual,test.prediction); lo=min(test.actual.min(),test.prediction.min()); hi=max(test.actual.max(),test.prediction.max()); ax.plot([lo,hi],[lo,hi],"k--"); ax.set(xlabel="Actual",ylabel="Predicted",title="Test: Actual vs Predicted"); fig.tight_layout(); fig.savefig(out/"actual_vs_predicted.png",dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,max(5,len(features)*.25))); imp=pd.Series(model.feature_importances_,index=features).sort_values(); ax.barh(imp.index,imp.values); ax.set_title("Feature importance"); fig.tight_layout(); fig.savefig(out/"feature_importance.png",dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5)); ax.hist(test.residual,bins=max(8,min(20,len(test)//4)),edgecolor="white"); ax.axvline(0,color="k",ls="--"); ax.set(xlabel="Actual - predicted",ylabel="Count",title="Test residual distribution"); fig.tight_layout(); fig.savefig(out/"residual_distribution.png",dpi=180); plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=False)
 for ax,(user,g) in zip(axes,test.groupby("user_id",sort=True)):
  for key,e in g.groupby(["crop_cycle_id","sample_num"],sort=True):
   label=f"cycle {key[0]} / sample {key[1]}"; ax.plot(e.target_date,e.actual,marker="o",label=label); ax.plot(e.target_date,e.prediction,ls="--",alpha=.75)
  ax.set(title=f"{user}: actual (solid) and predicted (dashed)",ylabel="Fruit cluster number"); ax.legend(ncol=4,fontsize=8)
 axes[-1].set_xlabel("Target date"); fig.tight_layout(); fig.savefig(out/"entity_timeseries.png",dpi=180); plt.close(fig)

def grouped_model_metrics(test):
 rows=[]
 for level,cols in [("overall",[]),("user",["user_id"]),("crop_cycle",["user_id","crop_cycle_id"]),("entity_cycle",KEYS)]:
  groups=[((),test)] if not cols else test.groupby(cols,sort=True)
  for key,g in groups:
   key=(key,) if cols and not isinstance(key,tuple) else key; row={"level":level,"rows":len(g),**metrics(g.actual,g.prediction)}
   for c,v in zip(cols,key): row[c]=v
   rows.append(row)
 return pd.DataFrame(rows)

def render_report(summary,best,baselines,comparison,grouped):
 val_p=baselines[(baselines.split.eq("val"))&(baselines.model.eq("persistence"))].iloc[0]; test_rf=comparison[comparison.model.eq("random_forest")].iloc[0]; test_p=comparison[comparison.model.eq("persistence")].iloc[0]
 improved=bool(test_rf.ccc>test_p.ccc and test_rf.rmse<test_p.rmse)
 conclusion=("Random Forest가 Persistence보다 개선되어 생육·작기 정보의 추가 활용 가능성이 확인되었다." if improved else "Random Forest가 Persistence를 개선하지 못해, 현재 자료에서는 화방수 자기 지속성이 더 강한 기준이며 추가 예측력은 입증되지 않았다.")
 return f"""# 작기 기반 개체별 차기 화방수 예측 결과

## 데이터와 평가 설계
- 작기 1·3 교집합에서 재배 종료 후 최초 화방수 감소 시점 이후 10행을 제외한 raw 데이터 506행, `(user_id, crop_cycle_id, sample_num)` 기준 16개 시계열을 사용했다.
- 각 시계열의 최초 3회는 이력으로만 사용하고 마지막 행은 다음 정답이 없어 제외하여 442개 예측 표본을 생성했다.
- 타깃 날짜 기준 Train {summary['split_rows']['train']}행, Validation {summary['split_rows']['val']}행, Test {summary['split_rows']['test']}행이며 작기 경계를 연결하지 않았다.
- Compact 26개와 Extended 70개 후보를 사전 정의했고, Spearman은 설명용으로만 산출했다.

## Validation 기반 제한적 모델 선정
- 2개 변수군 × 4개 RF 설정만 비교했으며 `{best['feature_set']}` / `{best['config']}`를 Validation CCC로 선택했다.
- 선택 RF Validation: MAE {best['mae']:.4f}, RMSE {best['rmse']:.4f}, R² {best['r2']:.4f}, CCC {best['ccc']:.4f}
- Persistence Validation: MAE {val_p.mae:.4f}, RMSE {val_p.rmse:.4f}, R² {val_p.r2:.4f}, CCC {val_p.ccc:.4f}
- Test는 구성 확정 후 한 번만 평가했다.

## Test 비교
{comparison.to_markdown(index=False,floatfmt='.4f')}

## 사용자·작기·개체별 결과
{grouped.to_markdown(index=False,floatfmt='.4f')}

![실제값과 예측값](results/figures/actual_vs_predicted.png)
![개체-작기별 시계열](results/figures/entity_timeseries.png)
![잔차 분포](results/figures/residual_distribution.png)
![변수 중요도](results/figures/feature_importance.png)

## 보수적 결론
{conclusion}

이는 데이터가 절대적으로 활용 불가능하다는 뜻은 아니지만, 현재 2개 사용자·4개 작기와 시간 후반 Test 조건만으로 생육변수의 Persistence 대비 추가 가치를 주장하기는 어렵다. 분포 이동과 Train 범위 밖 값 비율은 `diagnostics/` 산출물과 함께 해석해야 하며 Test 재튜닝은 수행하지 않았다.
"""

def run(project_root=ROOT):
 data_dir=project_root/"data"; artifacts=project_root/"artifacts"; processed=project_root/"processed_data"; raw,_=load_raw(data_dir); engineered=engineer(raw); frames=assign_splits(engineered,raw)
 processed.mkdir(parents=True,exist_ok=True)
 tracking=["sn","user_id","crop_cycle_id","cropping_serl_no","sample_num","meas_date","target_date","crop_start_date","input_raw_split","target_split"]
 for s,d in frames.items(): d[[*tracking,*EXTENDED,"next_fruit_cluster_num"]].to_parquet(processed/f"{s}_candidates.parquet",index=False)
 manifest={"rows":{**{s:len(d) for s,d in frames.items()},"total":len(engineered)},"entity_key":KEYS,"compact_features":COMPACT,"extended_features":EXTENDED,"history_minimum":4}; (processed/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 feature_dir=artifacts/"feature_engineering"; feature_dir.mkdir(parents=True,exist_ok=True); (feature_dir/"feature_manifest.json").write_text(json.dumps({"entity_key":KEYS,"source_rows":len(raw),"prediction_rows":len(engineered),"compact_feature_count":len(COMPACT),"extended_feature_count":len(EXTENDED),"initial_history_rows_excluded":48,"last_rows_excluded":16,"future_information_used":False},indent=2)+"\n"); audit=audit_outputs(raw,frames,artifacts/"data_audit")
 diag=artifacts/"diagnostics"; diag.mkdir(parents=True,exist_ok=True); target_summary(frames).to_csv(diag/"target_distribution.csv",index=False); baselines,entity=baseline_results(frames); baselines.to_csv(diag/"baseline_metrics.csv",index=False); entity.to_csv(diag/"persistence_entity_metrics.csv",index=False); descriptive_spearman(frames["train"]).to_csv(diag/"descriptive_spearman.csv",index=False)
 table,best=select_rf(frames); selection=artifacts/"model_selection"; selection.mkdir(parents=True,exist_ok=True); table.to_csv(selection/"validation_rf_comparison.csv",index=False); (selection/"selected_config.json").write_text(json.dumps(best,indent=2)+"\n")
 model,features,test,test_metrics,combined=final_fit_evaluate(frames,best); model_dir=artifacts/"model"; result=artifacts/"results"; model_dir.mkdir(parents=True,exist_ok=True); result.mkdir(parents=True,exist_ok=True); joblib.dump(model,model_dir/"random_forest.joblib")
 config={"selected":best,"features":features,"rows":{"train":len(frames['train']),"validation":len(frames['val']),"final_training":len(combined),"test":len(test)}}; (model_dir/"model_config.json").write_text(json.dumps(config,indent=2)+"\n")
 comparison=baselines[baselines.split.eq("test")].copy(); comparison=pd.concat([comparison,pd.DataFrame([{"split":"test","model":"random_forest","operational":True,**test_metrics}])],ignore_index=True); comparison.to_csv(result/"test_model_comparison.csv",index=False)
 test[[*tracking,"actual","prediction","residual"]].to_parquet(result/"test_predictions.parquet",index=False)
 reloaded=joblib.load(model_dir/"random_forest.joblib"); reloaded_prediction=reloaded.predict(test[features]); max_reload_difference=float(np.max(np.abs(reloaded_prediction-test.prediction.to_numpy())))
 if not np.allclose(reloaded_prediction,test.prediction.to_numpy(),rtol=0,atol=1e-14): raise ValueError("Reloaded model predictions differ")
 integrity={"passed":True,"reload_prediction_tolerance":1e-14,"reload_max_absolute_difference":max_reload_difference,"checked_outputs":["processed_data/{train,val,test}_candidates.parquet","artifacts/model/random_forest.joblib","artifacts/results/test_predictions.parquet"]}; (artifacts/"artifact_integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
 entity_rows=[]
 for key,g in test.groupby(KEYS): entity_rows.append({"user_id":key[0],"crop_cycle_id":key[1],"sample_num":key[2],"rows":len(g),**metrics(g.actual,g.prediction)})
 pd.DataFrame(entity_rows).to_csv(result/"entity_metrics.csv",index=False); grouped=grouped_model_metrics(test); grouped.to_csv(result/"grouped_metrics.csv",index=False); diagnostics={"test_metrics":test_metrics,"prediction_mean":float(test.prediction.mean()),"prediction_std":float(test.prediction.std(ddof=0)),"actual_mean":float(test.actual.mean()),"actual_std":float(test.actual.std(ddof=0)),"feature_extrapolation":extrapolation(combined,test,features),"target_train_range":[float(combined.next_fruit_cluster_num.min()),float(combined.next_fruit_cluster_num.max())],"prediction_outside_training_target_range_ratio":float(((test.prediction<combined.next_fruit_cluster_num.min())|(test.prediction>combined.next_fruit_cluster_num.max())).mean())}; (result/"test_metrics.json").write_text(json.dumps(diagnostics,indent=2)+"\n"); plots(test,model,features,result/"figures")
 summary={"status":"complete","raw_rows":len(raw),"entity_cycles":raw.groupby(KEYS).ngroups,"prediction_rows":len(engineered),"split_rows":manifest["rows"],"audit":audit,"artifact_integrity":integrity,"selected_config":best,"validation_persistence":baselines[(baselines.split.eq('val'))&(baselines.model.eq('persistence'))].iloc[0].to_dict(),"test_comparison":comparison.to_dict(orient="records"),"diagnostics":diagnostics}; (artifacts/"pipeline_manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n"); (artifacts/"final_report.md").write_text(render_report(summary,best,baselines,comparison,grouped),encoding="utf-8")
 return summary
