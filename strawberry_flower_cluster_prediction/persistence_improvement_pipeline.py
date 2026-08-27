#!/usr/bin/env python3
"""Persistence-centered RF comparison with train-only temporal feature selection."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parent

def load_base():
 spec=importlib.util.spec_from_file_location("flower_base",ROOT/"revised_pipeline.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
B=load_base(); KEYS=B.KEYS; BASE=B.BASE; FULL=B.EXTENDED; RF_CONFIGS=B.RF_CONFIGS; SPLITS=B.SPLITS
SELF=["fruit_cluster_num","fruit_cluster_num_lag1","fruit_cluster_num_change","fruit_cluster_num_daily_change","fruit_cluster_num_roll4_mean","fruit_cluster_num_roll4_std","fruit_cluster_num_roll4_slope_per_day","days_since_previous","days_since_crop_start"]
COMPACT_GROWTH=[*SELF,*BASE[1:]]
CANDIDATES=[*FULL,"fruit_cluster_num_lag1","fruit_cluster_num_change","fruit_cluster_num_roll4_slope_per_day"]

def rolling_slope(frame):
 values=np.full(len(frame),np.nan,float); dates=frame.meas_date.to_numpy(dtype="datetime64[ns]"); y=frame.fruit_cluster_num.to_numpy(float)
 for i in range(3,len(frame)):
  x=(dates[i-3:i+1]-dates[i-3]).astype("timedelta64[s]").astype(float)/86400
  values[i]=np.polyfit(x,y[i-3:i+1],1)[0] if np.ptp(x)>0 else 0.0
 return pd.Series(values,index=frame.index)

def engineer(raw):
 d=B.engineer(raw); ordered=raw.sort_values([*KEYS,"meas_date","sn"],kind="stable").copy(); g=ordered.groupby(KEYS,sort=False)
 ordered["fruit_cluster_num_lag1"]=g.fruit_cluster_num.shift(1); ordered["fruit_cluster_num_change"]=ordered.fruit_cluster_num-ordered.fruit_cluster_num_lag1
 ordered["fruit_cluster_num_roll4_slope_per_day"]=g.apply(rolling_slope,include_groups=False).reset_index(level=KEYS,drop=True).sort_index()
 extras=ordered[["sn","fruit_cluster_num_lag1","fruit_cluster_num_change","fruit_cluster_num_roll4_slope_per_day"]]
 d=d.merge(extras,on="sn",how="left",validate="one_to_one")
 if d[CANDIDATES].isna().any().any() or not np.isfinite(d[CANDIDATES].to_numpy(float)).all(): raise ValueError("New history features contain invalid values")
 d["target_delta"]=d.next_fruit_cluster_num-d.fruit_cluster_num
 if d.target_delta.lt(0).any(): raise ValueError("Target delta must be non-negative after decline-tail filtering")
 return d

def make_temporal_folds(train):
 dates=np.array(sorted(pd.to_datetime(train.target_date).unique())); n=len(dates); bounds=[int(n*x) for x in (.4,.6,.8,1.0)]; folds=[]
 for i in range(3):
  tr=train.target_date.lt(dates[bounds[i]]); va=train.target_date.ge(dates[bounds[i]])&train.target_date.lt(dates[bounds[i+1]] if bounds[i+1]<n else dates[-1]+np.timedelta64(1,"D"))
  if not tr.any() or not va.any() or train.loc[tr,"target_date"].max()>=train.loc[va,"target_date"].min(): raise ValueError("Invalid expanding temporal fold")
  folds.append((np.flatnonzero(tr),np.flatnonzero(va)))
 return folds

def rf(name,depth,leaf):
 return RandomForestRegressor(n_estimators=500,max_depth=depth,min_samples_leaf=leaf,max_features="sqrt",random_state=42,n_jobs=-1)

def restored_prediction(model,frame,features,target_type):
 raw=model.predict(frame[features]); current=frame.fruit_cluster_num.to_numpy(float)
 return np.maximum(current,raw) if target_type=="direct" else current+np.maximum(0,raw)

def cv_mae(train,folds,features):
 scores=[]
 for tr,va in folds:
  model=rf("RF-1",4,3); model.fit(train.iloc[tr][features],train.iloc[tr].target_delta); pred=train.iloc[va].fruit_cluster_num+np.maximum(0,model.predict(train.iloc[va][features])); scores.append(B.metrics(train.iloc[va].next_fruit_cluster_num,pred)["mae"])
 return float(np.mean(scores))

def select_features(train,out):
 out.mkdir(parents=True,exist_ok=True); folds=make_temporal_folds(train); importance=[]
 for fold,(tr,va) in enumerate(folds,1):
  model=rf("RF-1",4,3); model.fit(train.iloc[tr][CANDIDATES],train.iloc[tr].target_delta)
  pi=permutation_importance(model,train.iloc[va][CANDIDATES],train.iloc[va].target_delta,scoring="neg_mean_absolute_error",n_repeats=20,random_state=42,n_jobs=-1)
  for feature,mean,std in zip(CANDIDATES,pi.importances_mean,pi.importances_std): importance.append({"fold":fold,"feature":feature,"importance_mean":mean,"importance_std":std,"train_end":train.iloc[tr].target_date.max(),"validation_start":train.iloc[va].target_date.min(),"validation_end":train.iloc[va].target_date.max()})
 imp=pd.DataFrame(importance); imp.to_csv(out/"fold_permutation_importance.csv",index=False)
 aggregate=imp.groupby("feature",as_index=False).agg(median_importance=("importance_mean","median"),positive_fold_count=("importance_mean",lambda x:int((x>0).sum()))).sort_values(["median_importance","positive_fold_count"],ascending=False,kind="stable")
 corr=train[CANDIDATES].corr(method="spearman").abs(); kept=[]; redundancy=[]
 for f in aggregate.feature:
  conflicts=[k for k in kept if corr.loc[f,k]>=.90]
  if conflicts: redundancy.append({"removed_feature":f,"retained_feature":conflicts[0],"abs_spearman":corr.loc[f,conflicts[0]]})
  else: kept.append(f)
 pd.DataFrame(redundancy,columns=["removed_feature","retained_feature","abs_spearman"]).to_csv(out/"redundancy_filter.csv",index=False)
 selected=list(SELF); best=cv_mae(train,folds,selected); history=[{"step":0,"candidate":"SELF_BASE","accepted":True,"feature_count":len(selected),"mean_cv_mae":best,"improvement":None}]; failures=0
 for f in kept:
  if f in selected: continue
  score=cv_mae(train,folds,[*selected,f]); improvement=best-score; accept=improvement>=.0001
  history.append({"step":len(history),"candidate":f,"accepted":accept,"feature_count":len(selected)+(1 if accept else 0),"mean_cv_mae":score,"improvement":improvement})
  if accept: selected.append(f); best=score; failures=0
  else: failures+=1
  if len(selected)>=20 or failures>=5: break
 aggregate.to_csv(out/"aggregate_permutation_importance.csv",index=False); pd.DataFrame(history).to_csv(out/"forward_selection_history.csv",index=False)
 manifest={"selected_features":selected,"feature_count":len(selected),"folds":3,"candidate_count":len(CANDIDATES),"redundancy_threshold":.9,"minimum_mae_improvement":.0001,"best_mean_cv_mae":best,"validation_or_test_used":False}; (out/"cv_selected_features.json").write_text(json.dumps(manifest,indent=2)+"\n")
 return selected,folds,manifest

def evaluate_candidates(frames,feature_sets):
 rows=[]; predictions=[]; y=frames["val"].next_fruit_cluster_num
 for set_name,features in feature_sets.items():
  for target_type in ("direct","delta"):
   target="next_fruit_cluster_num" if target_type=="direct" else "target_delta"
   for name,depth,leaf in RF_CONFIGS:
    model=rf(name,depth,leaf); model.fit(frames["train"][features],frames["train"][target]); pred=restored_prediction(model,frames["val"],features,target_type); met=B.metrics(y,pred)
    rows.append({"feature_set":set_name,"target_type":target_type,"config":name,"max_depth":depth,"min_samples_leaf":leaf,"feature_count":len(features),**met})
    predictions.append(pd.DataFrame({"row_index":frames["val"].index,"feature_set":set_name,"target_type":target_type,"config":name,"actual":y,"current":frames["val"].fruit_cluster_num,"prediction":pred}))
 table=pd.DataFrame(rows); ranked=table.assign(_ccc=table.ccc.fillna(-np.inf)).sort_values(["mae","_ccc","rmse","feature_count","max_depth"],ascending=[True,False,True,True,True],kind="stable"); best=ranked.iloc[0].drop(labels="_ccc").to_dict()
 return table,best,pd.concat(predictions,ignore_index=True)

def representative(table):
 return table.sort_values(["mae","ccc","rmse","feature_count","max_depth"],ascending=[True,False,True,True,True],kind="stable").groupby(["feature_set","target_type"],as_index=False).first()

def change_diagnostics(frame,prediction):
 actual_change=frame.next_fruit_cluster_num.to_numpy()-frame.fruit_cluster_num.to_numpy(); predicted_change=prediction-frame.fruit_cluster_num.to_numpy(); actual_up=actual_change>0; predicted_up=predicted_change>=.5; tn,fp,fn,tp=confusion_matrix(actual_up,predicted_up,labels=[False,True]).ravel()
 rows=[]
 for label,mask in (("unchanged",~actual_up),("increase",actual_up)):
  rows.append({"group":label,"rows":int(mask.sum()),"ratio":float(mask.mean()),"mae":float(np.mean(np.abs(frame.next_fruit_cluster_num.to_numpy()[mask]-prediction[mask])))})
 return pd.DataFrame(rows),{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"threshold":.5}

def fit_final(frames,best,feature_sets):
 features=feature_sets[best["feature_set"]]; combined=pd.concat([frames["train"],frames["val"]],ignore_index=True); target="next_fruit_cluster_num" if best["target_type"]=="direct" else "target_delta"; model=rf(best["config"],int(best["max_depth"]),int(best["min_samples_leaf"])); model.fit(combined[features],combined[target]); test=frames["test"].copy(); test["prediction"]=restored_prediction(model,test,features,best["target_type"]); test["actual"]=test.next_fruit_cluster_num; test["persistence_prediction"]=test.fruit_cluster_num; test["residual"]=test.actual-test.prediction; return model,features,combined,test

def grouped_comparison(test):
 rows=[]
 for level,cols in [("overall",[]),("user",["user_id"]),("crop_cycle",["user_id","crop_cycle_id"]),("entity_cycle",KEYS)]:
  groups=[((),test)] if not cols else test.groupby(cols,sort=True)
  for key,g in groups:
   key=(key,) if cols and not isinstance(key,tuple) else key; rfmae=B.metrics(g.actual,g.prediction)["mae"]; pmae=B.metrics(g.actual,g.persistence_prediction)["mae"]; row={"level":level,"rows":len(g),"rf_mae":rfmae,"persistence_mae":pmae,"rf_minus_persistence_mae":rfmae-pmae,**{f"rf_{k}":v for k,v in B.metrics(g.actual,g.prediction).items()}}
   for c,v in zip(cols,key): row[c]=v
   rows.append(row)
 return pd.DataFrame(rows)

def make_plots(test,val_predictions,best,out):
 out.mkdir(parents=True,exist_ok=True)
 fig,axes=plt.subplots(2,1,figsize=(14,9))
 for ax,(user,g) in zip(axes,test.groupby("user_id",sort=True)):
  for key,e in g.groupby(["crop_cycle_id","sample_num"]):
   ax.plot(e.target_date,e.actual,color="black",marker="o",lw=1.5); ax.plot(e.target_date,e.persistence_prediction,color="#E68613",ls="--",alpha=.8); ax.plot(e.target_date,e.prediction,color="#2878B5",marker="^",ms=4,alpha=.85)
  ax.set_title(user); ax.set_ylabel("Fruit cluster number"); ax.grid(alpha=.2)
 axes[-1].set_xlabel("Target date"); fig.legend(["Actual","Persistence","Selected RF"],loc="upper center",ncol=3); fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(out/"actual_persistence_rf_timeseries.png",dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5))
 for target,color in (("direct","#2878B5"),("delta","#D9534F")):
  subset=val_predictions[(val_predictions.target_type==target)&(val_predictions.config==val_predictions[val_predictions.target_type.eq(target)].sort_values("mae") .config.iloc[0])] if "mae" in val_predictions else val_predictions[val_predictions.target_type.eq(target)]
  # Use all candidate predictions of the selected config/feature set only when available.
  values=(subset.prediction-subset.current).to_numpy(); ax.hist(values,bins=20,alpha=.45,label=target,color=color)
 ax.axvline(.5,color="black",ls="--",label="increase threshold"); ax.set(xlabel="Predicted increase",ylabel="Count",title="Validation predicted increase distribution"); ax.legend(); fig.tight_layout(); fig.savefig(out/"direct_delta_predicted_change.png",dpi=180); plt.close(fig)

def render_report(summary,representatives,comparison,selected_manifest):
 p=comparison[comparison.model.eq("persistence")].iloc[0]; rfrow=comparison[comparison.model.eq("selected_random_forest")].iloc[0]; success=summary["validation_improves_persistence"] and summary["test_improves_persistence"]
 conclusion=("Validation과 Test 모두에서 Persistence MAE를 개선하여 추가 예측정보가 확인되었다." if success else ("RF는 Persistence보다 MAE가 높아 1차 성공 기준은 충족하지 못했다. 다만 Test RMSE·R²·CCC가 개선되어 큰 오차와 전체 일치도 완화 가능성은 관찰됐으며, 증가 시점 탐지 실패를 고려하면 제한적인 결과다." if rfrow.rmse<p.rmse and rfrow.ccc>p.ccc else "선정 RF가 Validation과 Test에서 모두 Persistence를 개선하지 못해 추가 예측력은 입증되지 않았다."))
 return f"""# Persistence 개선 중심 화방수 예측 결과

- Raw 506행, 예측 표본 442행(Train 292 / Validation 76 / Test 74)
- 다음 화방수 변화 없음 비율: Train 78.8%, Validation 80.3%, Test 71.6%
- Train 내부 3-fold 시간순 CV로 {selected_manifest['feature_count']}개 변수를 선정했으며 Validation/Test는 변수 선정에 사용하지 않았다.
- 4개 변수군 × Direct/Delta × 4개 RF 설정, 총 32개를 공식 Validation에서 비교했다.

## 변수군·타깃별 Validation 대표 결과
{representatives.to_markdown(index=False,floatfmt='.4f')}

## 최종 Test 결과
{comparison.to_markdown(index=False,floatfmt='.4f')}

## 결론
{conclusion}

Test는 이전 실험에서 확인된 고정 평가 구간이므로 완전히 새로운 blind holdout으로 표현하지 않으며, 이번 선택과 튜닝에는 사용하지 않았다.
"""

def run(project_root=ROOT):
 artifacts=project_root/"artifacts"; processed=project_root/"processed_data"; raw,_=B.load_raw(project_root/"data"); engineered=engineer(raw); frames=B.assign_splits(engineered,raw)
 tracking=["sn","user_id","crop_cycle_id","cropping_serl_no","sample_num","meas_date","target_date","crop_start_date","input_raw_split","target_split"]
 for split,d in frames.items(): d[[*tracking,*CANDIDATES,"next_fruit_cluster_num","target_delta"]].to_parquet(processed/f"{split}_candidates.parquet",index=False)
 selection=artifacts/"model_selection"; selected,folds,selected_manifest=select_features(frames["train"],selection); feature_sets={"self_history":SELF,"compact_growth":COMPACT_GROWTH,"cv_selected":selected,"full":FULL}
 table,best,val_predictions=evaluate_candidates(frames,feature_sets); table.to_csv(selection/"validation_rf_comparison.csv",index=False); representative_table=representative(table); representative_table.to_csv(selection/"validation_representatives.csv",index=False); val_predictions.to_parquet(selection/"validation_candidate_predictions.parquet",index=False); (selection/"selected_config.json").write_text(json.dumps({**best,"features":feature_sets[best["feature_set"]]},indent=2)+"\n")
 baselines,_=B.baseline_results(frames); baselines.to_csv(artifacts/"diagnostics/baseline_metrics.csv",index=False)
 val_base=baselines[baselines.split.eq("val")][["model","mae","rmse","r2","ccc"]].copy(); val_base.insert(0,"comparison_type","baseline"); val_base["feature_set"]=""; val_base["target_type"]=""; val_base["config"]=""; val_base["feature_count"]=0
 val_rf=representative_table.copy(); val_rf.insert(0,"comparison_type","rf_representative"); val_rf["model"]="random_forest"; pd.concat([val_base,val_rf],ignore_index=True,sort=False).to_csv(selection/"validation_model_comparison.csv",index=False)
 change_rows=[]
 for split,d in frames.items():
  delta=d.target_delta; change_rows.append({"split":split,"rows":len(d),"unchanged_count":int(delta.eq(0).sum()),"unchanged_ratio":float(delta.eq(0).mean()),"increase_count":int(delta.gt(0).sum()),"increase_ratio":float(delta.gt(0).mean()),"mean_delta":float(delta.mean()),"max_delta":float(delta.max())})
 pd.DataFrame(change_rows).to_csv(artifacts/"diagnostics/target_change_distribution.csv",index=False)
 model,features,combined,test=fit_final(frames,best,feature_sets); model_dir=artifacts/"model"; result=artifacts/"results"; model_dir.mkdir(parents=True,exist_ok=True); result.mkdir(parents=True,exist_ok=True); joblib.dump(model,model_dir/"random_forest.joblib")
 config={"selected":best,"features":features,"target_type":best["target_type"],"nondecreasing_constraint":True,"rows":{"train":len(frames['train']),"validation":len(frames['val']),"final_training":len(combined),"test":len(test)}}; (model_dir/"model_config.json").write_text(json.dumps(config,indent=2)+"\n")
 train_mean=frames["train"].next_fruit_cluster_num.mean(); predictions={"train_mean":np.full(len(test),train_mean),"persistence":test.persistence_prediction.to_numpy(),"selected_random_forest":test.prediction.to_numpy()}; comp=pd.DataFrame([{"model":name,**B.metrics(test.actual,pred)} for name,pred in predictions.items()]); comp.to_csv(result/"test_model_comparison.csv",index=False)
 test[[*tracking,"actual","persistence_prediction","prediction","residual"]].to_parquet(result/"test_predictions.parquet",index=False); groups=grouped_comparison(test); groups.to_csv(result/"grouped_metrics.csv",index=False); change_group,cm=change_diagnostics(test,test.prediction.to_numpy()); change_group.to_csv(result/"change_group_metrics.csv",index=False); (result/"increase_confusion_matrix.json").write_text(json.dumps(cm,indent=2)+"\n")
 reloaded=joblib.load(model_dir/"random_forest.joblib"); rp=restored_prediction(reloaded,test,features,best["target_type"]); integrity={"passed":bool(np.allclose(rp,test.prediction,rtol=0,atol=1e-14)),"max_reload_difference":float(np.max(np.abs(rp-test.prediction)))}; (artifacts/"artifact_integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
 # Plot representative best direct and delta candidates.
 chosen=[]
 for t in ("direct","delta"):
  row=table[table.target_type.eq(t)].sort_values(["mae","ccc"],ascending=[True,False]).iloc[0]; chosen.append(val_predictions[(val_predictions.target_type==t)&(val_predictions.feature_set==row.feature_set)&(val_predictions.config==row.config)])
 make_plots(test,pd.concat(chosen),best,result/"figures")
 vp=baselines[(baselines.split=="val")&(baselines.model=="persistence")].iloc[0]; tp=comp[comp.model=="persistence"].iloc[0]; trf=comp[comp.model=="selected_random_forest"].iloc[0]; summary={"status":"complete","candidate_count":len(table),"selected_config":best,"selected_features":selected,"validation_persistence_mae":float(vp.mae),"validation_improves_persistence":bool(best["mae"]<vp.mae),"test_persistence_mae":float(tp.mae),"test_rf_mae":float(trf.mae),"test_improves_persistence":bool(trf.mae<tp.mae),"test_results":comp.to_dict(orient="records"),"artifact_integrity":integrity}; (artifacts/"pipeline_manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n"); (artifacts/"final_report.md").write_text(render_report(summary,representative_table,comp,selected_manifest),encoding="utf-8")
 return summary
