#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent)); sys.path.insert(0,str(ROOT))
from common_regression.metrics import regression_metrics
from common_regression.models import make_adapter
from dataset import load_dataset
from prepare_splits import TRUSSES

spec=importlib.util.spec_from_file_location("plot_helpers",ROOT.parent/"plot_final_selected_period_predictions.py")
P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

def predictions(run,truss):
    bundle,_,_,_=load_dataset(truss); test=bundle.load_test(); frames={"train":bundle.train,"val":bundle.validation,"test":test}
    config=json.loads((run/truss/"selected_configuration.json").read_text()); tuned=json.loads((run/truss/"internal_cv/hyperparameter_tuning/selected_params.json").read_text()); suite=yaml.safe_load((ROOT/"config/model_suite.yaml").read_text())
    outputs=[]; columns=["user_id","sample_num","crop_sn","target_date",bundle.spec.target]
    for split,frame in frames.items():
        out=frame[columns].copy().rename(columns={bundle.spec.target:"actual"}); out["dataset_split"]=split; out["persistence"]=frame[bundle.spec.current_value].to_numpy(float); outputs.append(out)
    combined=pd.concat([bundle.train,bundle.validation],ignore_index=True)
    for model in ("poisson","random_forest","catboost","mlp","tabm"):
        seeds=suite["validation_seeds"] if model in {"mlp","tabm"} else [suite["validation_seeds"][0]]; diagnostic=[]; final=[]
        for seed in seeds:
            adapter=make_adapter(model,config["features"],[],tuned["params"][model],seed); adapter.fit(bundle.train,bundle.spec.target,fixed_epochs=tuned["fixed_epochs"][model]); diagnostic.append(adapter)
            adapter=make_adapter(model,config["features"],[],tuned["params"][model],seed); adapter.fit(combined,bundle.spec.target,fixed_epochs=tuned["fixed_epochs"][model]); final.append(adapter)
        for output,frame in zip(outputs[:2],(bundle.train,bundle.validation)): output[model]=np.mean([a.predict(frame) for a in diagnostic],axis=0)
        outputs[2][model]=np.mean([a.predict(test) for a in final],axis=0)
    result=pd.concat(outputs,ignore_index=True); result["truss"]=truss
    expected=pd.read_csv(run/truss/"validation/model_metrics.csv").set_index("model")
    for model in P.MODELS:
        actual=result[result.dataset_split.eq("val")]; got=regression_metrics(actual.actual,actual[model])
        if max(abs(got[k]-expected.loc[model,k]) for k in ("rmse","mae","r2","ccc"))>1e-6: raise RuntimeError(f"{truss} {model}: Validation mismatch")
    selected=config["selected_learning_model"]; stored=pd.read_parquet(run/truss/"final_test_predictions.parquet").sort_values(list(bundle.spec.row_key)); current=result[result.dataset_split.eq("test")].sort_values(list(bundle.spec.row_key))
    if not np.allclose(stored.prediction,current[selected],atol=1e-6,rtol=0): raise RuntimeError(f"{truss}: Test prediction mismatch")
    return result

def observation_index_axis(data, period_column):
    """Place observed dates at equal intervals, without calendar-time whitespace."""
    result=data.copy(); result["target_date"]=pd.to_datetime(result["target_date"])
    periods=result.groupby(period_column).target_date.agg(["min","max"]).sort_values("min",kind="stable")
    period_order={period:index for index,period in enumerate(periods.index)}
    keys=(result[[period_column,"target_date"]].drop_duplicates()
          .assign(_period_order=lambda frame: frame[period_column].map(period_order))
          .sort_values(["_period_order","target_date"],kind="stable").reset_index(drop=True))
    keys["plot_x"]=np.arange(len(keys),dtype=float)
    result=result.merge(keys[[period_column,"target_date","plot_x"]],on=[period_column,"target_date"],how="left",validate="many_to_one")
    bounds=keys.groupby(period_column).plot_x.agg(["min","max"]).reindex(periods.index)
    breaks=[(bounds.iloc[index]["max"]+bounds.iloc[index+1]["min"])/2 for index in range(len(bounds)-1)]
    return result,periods,breaks

def plot(data,out,shared_x=True,compact_observations=False):
    score=(data.groupby(["user_id","sample_num"]).agg(rows=("target_date","size"),truss_coverage=("truss","nunique"),periods=("crop_sn","nunique")).sort_values(["truss_coverage","rows","periods"],ascending=False,kind="stable"))
    user,sample=score.index[0]; selected=data[data.user_id.eq(user)&data.sample_num.eq(sample)].copy(); selected["target_date"]=pd.to_datetime(selected.target_date)
    axis_ref,_,breaks=P.compress_period_axis(selected,"crop_sn"); xlim=(axis_ref.plot_x.min(),axis_ref.plot_x.max())
    values=selected[["actual",*P.MODELS]].to_numpy(float); ymin,ymax=np.nanmin(values),np.nanmax(values); margin=max(.5,(ymax-ymin)*.05)
    fig,axes=plt.subplots(3,1,figsize=(19,18),squeeze=False); plotted=[]
    titles={"first_fruits_num":"1st fruit truss","second_fruits_num":"2nd fruit truss","third_fruits_num":"3rd fruit truss"}
    for row,truss in enumerate(TRUSSES):
        ax=axes[row,0]; g=selected[selected.truss.eq(truss)].copy()
        reference = selected if shared_x else g
        if compact_observations:
            compact,periods,local_breaks=observation_index_axis(g,"crop_sn")
        else:
            compact,periods,local_breaks=P.compress_period_axis(g,"crop_sn",reference=reference)
        plotted.append(compact)
        P.decorate_compact_axis(
            ax,
            compact,
            "crop_sn",
            breaks if shared_x else local_breaks,
            tick_reference=axis_ref if shared_x else compact,
        )
        if compact_observations:
            ax.set_xlim(-.5,max(.5,float(compact.plot_x.max())+.5))
        elif shared_x:
            ax.set_xlim(*xlim)
        first=True
        for (_,split),segment in compact.groupby(["crop_sn","dataset_split"],sort=False):
            segment=segment.sort_values("target_date"); ax.plot(segment.plot_x,segment.actual,color="#222222",lw=2.8,marker="o",ms=5.5,label="Actual" if first else "_nolegend_",zorder=8)
            for model in P.MODELS:
                ls,marker=P.STYLES[model]; ax.plot(segment.plot_x,segment[model],color=P.COLORS[model],ls=ls,marker=marker,lw=2,ms=5,alpha=.9,label=P.LABELS[model] if first else "_nolegend_")
            first=False
        xlabel="Observed target date (equal spacing)" if compact_observations else "Target date (crop gaps compressed)"
        ax.set_title(f"{user} | Sample {sample} | {titles[truss]}",fontsize=18); ax.set_ylabel("Fruit-set count",fontsize=17); ax.set_xlabel(xlabel,fontsize=17); ax.set_ylim(ymin-margin,ymax+margin); ax.grid(alpha=.22); ax.tick_params(axis="y",labelsize=14)
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",ncol=7,frameon=False,fontsize=18,bbox_to_anchor=(.5,.985)); fig.suptitle("Fruit-set Predictions: Independent Temporal Split by Fruit Truss",fontsize=28,y=.999); fig.tight_layout(rect=(0,.015,1,.955),h_pad=1.5)
    out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=200,bbox_inches="tight"); plt.close(fig)
    pd.concat(plotted,ignore_index=True).to_csv(out.with_suffix(".csv"),index=False,encoding="utf-8-sig"); selected.to_csv(out.with_name(out.stem+"_raw_rows.csv"),index=False,encoding="utf-8-sig")
    return user,sample

def main(run_id):
    run=ROOT/"artifacts"/run_id; data=pd.concat([predictions(run,t) for t in TRUSSES],ignore_index=True); out=run/"summary/entity_fruit_set_selected_period_predictions.png"; user,sample=plot(data,out); print({"user":user,"sample":sample,"figure":str(out)})

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); main(p.parse_args().run_id)
