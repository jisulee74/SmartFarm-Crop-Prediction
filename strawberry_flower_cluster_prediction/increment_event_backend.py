#!/usr/bin/env python3
"""Compare five Persistence-correction models on identical flower-cluster rows/features."""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
FLOWER_ROOT = ROOT
RF_ROOT = ROOT
sys.path.insert(0, str(FLOWER_ROOT))
import persistence_improvement_pipeline as legacy

FEATURES = legacy.COMPACT_GROWTH
MODELS = ("poisson", "random_forest", "catboost", "mlp", "tabm")
DL_MODELS = {"mlp", "tabm"}
CV_SEEDS = (42, 52, 62)
FINAL_SEEDS = (42, 52, 62, 72, 82)
THRESHOLDS = np.arange(0.05, 0.951, 0.025)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metrics(y, prediction):
    y = np.asarray(y, float); prediction = np.asarray(prediction, float)
    denominator = y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2
    return {"rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction)),
            "ccc": float(2*np.mean((y-y.mean())*(prediction-prediction.mean()))/denominator) if denominator else 0.0}


def load(split):
    frame = pd.read_parquet(RF_ROOT / "processed_data" / f"{split}_candidates.parquet")
    frame["target_date"] = pd.to_datetime(frame.target_date)
    frame["increase_target"] = (frame.next_fruit_cluster_num > frame.fruit_cluster_num).astype(int)
    if not frame[FEATURES].notna().all().all(): raise ValueError(f"{split}: invalid features")
    return frame


class TorchBinary:
    def __init__(self, kind, seed, fixed_epochs=None):
        self.kind, self.seed, self.fixed_epochs = kind, seed, fixed_epochs
        self.best_epoch = None

    def make_model(self, width):
        if self.kind == "mlp":
            return nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Dropout(.1),
                                 nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
        from tabm import TabM
        return TabM.make(n_num_features=width,cat_cardinalities=[],d_out=1,
                         arch_type="tabm",k=16,n_blocks=2,dropout=0.0)

    def forward(self, x):
        value = self.model(x)
        if self.kind == "tabm": value = value.mean(dim=1)
        return value.squeeze(-1)

    def fit(self, train, validation=None):
        seed_all(self.seed); self.scaler = StandardScaler().fit(train[FEATURES])
        x=torch.tensor(self.scaler.transform(train[FEATURES]),dtype=torch.float32)
        y=torch.tensor(train.increase_target.to_numpy(np.float32))
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.model=self.make_model(x.shape[1]).to(device)
        lr=.001 if self.kind=="mlp" else .002
        opt=torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=3e-4)
        loader=DataLoader(TensorDataset(x,y),batch_size=min(64,len(x)),shuffle=True,
                          generator=torch.Generator().manual_seed(self.seed))
        val_xy=None
        if validation is not None:
            val_xy=(torch.tensor(self.scaler.transform(validation[FEATURES]),dtype=torch.float32,device=device),
                    torch.tensor(validation.increase_target.to_numpy(np.float32),device=device))
        best=float("inf"); state=None; stale=0; epochs=int(self.fixed_epochs or 500)
        for epoch in range(1,epochs+1):
            self.model.train()
            for bx,by in loader:
                bx,by=bx.to(device),by.to(device); opt.zero_grad()
                loss=nn.functional.binary_cross_entropy_with_logits(self.forward(bx),by)
                loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(),1.0); opt.step()
            self.best_epoch=epoch
            if val_xy is None: continue
            self.model.eval()
            with torch.no_grad(): value=float(nn.functional.binary_cross_entropy_with_logits(self.forward(val_xy[0]),val_xy[1]).cpu())
            if value < best-1e-7: best=value;state=copy.deepcopy(self.model.state_dict());self.best_epoch=epoch;stale=0
            else:
                stale+=1
                if stale>=50: break
        if state is not None:self.model.load_state_dict(state)
        self.model.to("cpu");return self

    def predict_proba(self, frame):
        self.model.eval();x=torch.tensor(self.scaler.transform(frame[FEATURES]),dtype=torch.float32)
        with torch.no_grad():return torch.sigmoid(self.forward(x)).numpy()

    def save(self,path):
        torch.save({"kind":self.kind,"features":FEATURES,"seed":self.seed,"best_epoch":self.best_epoch,
                    "state_dict":self.model.state_dict(),"scaler":self.scaler},path)


class Classical:
    def __init__(self, kind, seed, fixed_epochs=None):self.kind,self.seed,self.fixed_epochs=kind,seed,fixed_epochs;self.best_epoch=None
    def fit(self,train,validation=None):
        if self.kind=="poisson":
            self.scaler=StandardScaler().fit(train[FEATURES]);self.model=PoissonRegressor(alpha=1.,max_iter=2000).fit(self.scaler.transform(train[FEATURES]),train.increase_target)
        elif self.kind=="random_forest":
            self.model=RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=3,max_features="sqrt",random_state=self.seed,n_jobs=-1).fit(train[FEATURES],train.increase_target)
        else:
            from catboost import CatBoostClassifier
            iterations=int(self.fixed_epochs or 1000);self.model=CatBoostClassifier(iterations=iterations,depth=6,learning_rate=.03,l2_leaf_reg=7,loss_function="Logloss",verbose=False,allow_writing_files=False,random_seed=self.seed)
            kwargs={}
            if validation is not None and self.fixed_epochs is None:kwargs={"eval_set":(validation[FEATURES],validation.increase_target),"early_stopping_rounds":100}
            self.model.fit(train[FEATURES],train.increase_target,**kwargs);best=self.model.get_best_iteration();self.best_epoch=max(1,best+1) if best is not None and best>=0 else iterations
        return self
    def predict_proba(self,frame):
        if self.kind=="poisson":
            lam=np.maximum(0,self.model.predict(self.scaler.transform(frame[FEATURES])));return 1-np.exp(-lam)
        return self.model.predict_proba(frame[FEATURES])[:,1]
    def save(self,path):joblib.dump(self,path)


def adapter(kind,seed,fixed_epochs=None):return TorchBinary(kind,seed,fixed_epochs) if kind in DL_MODELS else Classical(kind,seed,fixed_epochs)


def cv_model(train,kind):
    fold_probabilities=[];fold_actual=[];epochs=[]
    for fit_idx,val_idx in legacy.make_temporal_folds(train):
        fit,val=train.iloc[fit_idx],train.iloc[val_idx];probs=[]
        for seed in (CV_SEEDS if kind in DL_MODELS else (42,)):
            model=adapter(kind,seed).fit(fit,val);probs.append(model.predict_proba(val))
            if model.best_epoch:epochs.append(model.best_epoch)
        fold_probabilities.extend(np.mean(probs,axis=0));fold_actual.extend(val.increase_target)
    probability=np.asarray(fold_probabilities);actual=np.asarray(fold_actual,bool);rows=[]
    for threshold in THRESHOLDS:
        event=probability>=threshold;rows.append({"model":kind,"threshold":float(threshold),"cv_mae":float(np.mean(event!=actual)),"fp":int((event&~actual).sum())})
    table=pd.DataFrame(rows).sort_values(["cv_mae","fp","threshold"],ascending=[True,True,False],kind="stable")
    return float(table.iloc[0].threshold),float(table.iloc[0].cv_mae),(int(np.median(epochs)) if epochs else None),table


def fit_predict(train,frame,kind,epochs,seeds):
    models=[];probabilities=[]
    for seed in seeds:
        model=adapter(kind,seed,epochs).fit(train);models.append(model);probabilities.append(model.predict_proba(frame))
    return models,np.mean(probabilities,axis=0)


def main():
    out=ROOT/"artifacts";out.mkdir(parents=True,exist_ok=True);train,validation=load("train"),load("val")
    validation_rows=[];settings={};threshold_tables=[]
    for kind in MODELS:
        threshold,cv_mae,epochs,tt=cv_model(train,kind);threshold_tables.append(tt);settings[kind]={"threshold":threshold,"cv_mae":cv_mae,"fixed_epochs":epochs}
        seeds=FINAL_SEEDS if kind in DL_MODELS else (42,);_,probability=fit_predict(train,validation,kind,epochs,seeds);event=probability>=threshold
        prediction=validation.fruit_cluster_num.to_numpy(float)+event.astype(float);truth=validation.increase_target.to_numpy(bool);tn,fp,fn,tp=confusion_matrix(truth,event,labels=[False,True]).ravel()
        validation_rows.append({"model":kind,"is_learning_model":True,"threshold":threshold,"cv_mae":cv_mae,"seed_count":len(seeds),**metrics(validation.next_fruit_cluster_num,prediction),"tn":tn,"fp":fp,"fn":fn,"tp":tp})
    persistence=validation.fruit_cluster_num.to_numpy(float);validation_rows.append({"model":"persistence","is_learning_model":False,"threshold":np.nan,"cv_mae":np.nan,"seed_count":0,**metrics(validation.next_fruit_cluster_num,persistence),"tn":61,"fp":0,"fn":15,"tp":0})
    validation_table=pd.DataFrame(validation_rows);validation_table.to_csv(out/"validation_model_comparison.csv",index=False);pd.concat(threshold_tables).to_csv(out/"internal_cv_threshold_search.csv",index=False)
    learning=validation_table[validation_table.is_learning_model];best=learning.rmse.min();tied=learning[learning.rmse<=best*1.01];winner=tied.sort_values(["mae","model"],kind="stable").iloc[0];selected=str(winner.model)
    frozen={"features":FEATURES,"feature_count":len(FEATURES),"selected_learning_model":selected,"settings":settings[selected],"selection_uses_test":False};(out/"selected_configuration.json").write_text(json.dumps(frozen,indent=2)+"\n")
    combined=pd.concat([train,validation],ignore_index=True);seeds=FINAL_SEEDS if selected in DL_MODELS else (42,);models,_=fit_predict(combined,combined.iloc[:1],selected,settings[selected]["fixed_epochs"],seeds)
    model_dir=out/"models";model_dir.mkdir(exist_ok=True)
    for model,seed in zip(models,seeds):model.save(model_dir/f"{selected}_seed_{seed}.bin")
    test=load("test");probability=np.mean([m.predict_proba(test) for m in models],axis=0);event=probability>=settings[selected]["threshold"];prediction=test.fruit_cluster_num.to_numpy(float)+event.astype(float);persistence=test.fruit_cluster_num.to_numpy(float);truth=test.increase_target.to_numpy(bool);tn,fp,fn,tp=confusion_matrix(truth,event,labels=[False,True]).ravel()
    test_table=pd.DataFrame([{"model":selected,**metrics(test.next_fruit_cluster_num,prediction)},{"model":"persistence",**metrics(test.next_fruit_cluster_num,persistence)}]);test_table.to_csv(out/"test_model_comparison.csv",index=False)
    pred=test[["sn","user_id","crop_cycle_id","sample_num","target_date","fruit_cluster_num","next_fruit_cluster_num"]].copy();pred["increase_probability"]=probability;pred["predicted_increase"]=event.astype(int);pred["prediction"]=prediction;pred["persistence_prediction"]=persistence;pred.to_parquet(out/"test_predictions.parquet",index=False)
    manifest={"status":"complete","stage":"test_once","selection_uses_test":False,"test_is_historical_reused_holdout":True,"selected":frozen,"test_confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"test":test_table.set_index("model").to_dict(orient="index")};(out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n");print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
