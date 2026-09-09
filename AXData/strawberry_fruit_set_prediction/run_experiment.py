#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent)); sys.path.insert(0,str(ROOT))
from common_regression.feature_catalog import QualityRules,apply_kept_features,greedy_correlation_filter,quality_audit
from common_regression.runner import CommonRegressionRunner
from dataset import load_dataset
from prepare_splits import TRUSSES,run as prepare

def build_runner(truss,run_root,config):
    bundle,blocks,catalog,sensors=load_dataset(truss); rules=QualityRules(near_constant_ratio=.99)
    quality=quality_audit(bundle.train,catalog,rules)
    catalog=catalog.merge(quality[["feature","status","missing_rate"]],on="feature",validate="one_to_one")
    catalog["coverage"]=[float(bundle.train[f].mean()) if f.endswith("_coverage") else 1.0 for f in catalog.feature]
    catalog["target_spearman"]=[bundle.train[f].corr(bundle.train[bundle.spec.target],method="spearman") for f in catalog.feature]
    kept,correlations=greedy_correlation_filter(bundle.train,catalog,{"individual_history":.95,"growth":.95,"nutrient":.95,"indoor":.95,"outdoor":.95})
    blocks=apply_kept_features(blocks,kept); out=run_root/truss; out.mkdir(parents=True,exist_ok=False)
    quality.to_csv(out/"feature_quality_audit.csv",index=False); correlations.to_csv(out/"correlation_clusters.csv",index=False); catalog.to_csv(out/"feature_catalog.csv",index=False)
    (out/"selected_sensor_variables.json").write_text(json.dumps(sensors,ensure_ascii=False,indent=2)+"\n")
    return CommonRegressionRunner(bundle,blocks,[],config,out)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); args=parser.parse_args()
    prepare(); config=yaml.safe_load((ROOT/"config/model_suite.yaml").read_text()); run_root=ROOT/config["artifact_root"]/args.run_id
    run_root.mkdir(parents=True,exist_ok=False)
    for truss in TRUSSES:
        print(f"START {truss}",flush=True); runner=build_runner(truss,run_root,config)
        runner.select_features(); runner.tune(); runner.validate(); runner.freeze(); runner.fit_final(); runner.test_once()
        print(f"COMPLETE {truss}",flush=True)
    print(run_root)

if __name__=="__main__": main()
