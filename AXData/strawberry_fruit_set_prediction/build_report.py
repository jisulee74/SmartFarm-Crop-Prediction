#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
TRUSSES=("first_fruits_num","second_fruits_num","third_fruits_num")
KOREAN={"first_fruits_num":"1화방 착과수","second_fruits_num":"2화방 착과수","third_fruits_num":"3화방 착과수"}
BLOCKS={"individual_history":"개체 이력","growth":"생육정보","nutrient":"양액정보","indoor":"내부환경","outdoor":"외부환경"}

def period(series):
    values=pd.to_datetime(series.dropna()); return "-" if values.empty else f"{values.min().date()}~{values.max().date()}"

def main(run_id):
    run=ROOT/"artifacts"/run_id; out=run/"summary"; out.mkdir(parents=True,exist_ok=True)
    all_base=pd.concat([pd.read_parquet(ROOT/"processed_data"/t/f"{s}_candidates.parquet") for t in TRUSSES for s in ("train","val","test")],ignore_index=True)
    facilities=all_base[["facility_id","user_id"]].drop_duplicates().sort_values(["facility_id","user_id"])
    facilities.to_csv(out/"facility_ids.csv",index=False,encoding="utf-8-sig")
    validation=[]; tests=[]; configs=[]; split_tables={}
    for truss in TRUSSES:
        frames={s:pd.read_parquet(ROOT/"processed_data"/truss/f"{s}_candidates.parquet").assign(dataset_split=s) for s in ("train","val","test")}
        data=pd.concat(frames.values(),ignore_index=True); rows=[]
        for user,g in data.groupby("user_id",sort=True):
            rows.append({"user_id":user,"row_count":len(g),"train_period":period(g.loc[g.dataset_split.eq("train"),"target_date"]),"validation_period":period(g.loc[g.dataset_split.eq("val"),"target_date"]),"test_period":period(g.loc[g.dataset_split.eq("test"),"target_date"])})
        rows.append({"user_id":"합계","row_count":len(data),"train_period":period(frames["train"].target_date),"validation_period":period(frames["val"].target_date),"test_period":period(frames["test"].target_date)})
        table=pd.DataFrame(rows); table.to_csv(out/f"{truss}_user_split_summary.csv",index=False,encoding="utf-8-sig"); split_tables[truss]=table
        val=pd.read_csv(run/truss/"validation/model_metrics.csv"); val.insert(0,"truss",truss); validation.append(val)
        manifest=json.loads((run/truss/"run_manifest.json").read_text()); config=json.loads((run/truss/"selected_configuration.json").read_text())
        configs.append({"truss":truss,"selected_feature_set":config["selected_feature_set"],"feature_count":len(config["features"]),"features":"|".join(config["features"]),"selected_learning_model":config["selected_learning_model"]})
        for model,key in ((config["selected_learning_model"],"final"),("persistence","persistence")): tests.append({"truss":truss,"model":model,**manifest[key]})
    validation=pd.concat(validation,ignore_index=True); validation.to_csv(out/"validation_six_model_metrics.csv",index=False)
    tests=pd.DataFrame(tests); tests.to_csv(out/"final_test_metrics.csv",index=False)
    configs=pd.DataFrame(configs); configs.to_csv(out/"selected_feature_sets.csv",index=False,encoding="utf-8-sig")
    lines=["# 화방별 독립 착과수 회귀 모델 결과 (19개 사용자 코호트)","", "## 시설 ID", "",f"- 시설 수: {facilities.facility_id.nunique()}개", "- 선정 기준: 1·2·3화방 착과수가 모두 유효한 조사 행을 보유하고, 2024년 이후 CI·TI·HI 센서 로그를 모두 보유한 19개 사용자를 동일 코호트로 고정", "- 코호트는 데이터 분할과 모델 평가 전에 확정하며, 특정 모델 성능이나 Validation/Test 결과로 사용자를 선택하지 않음", "",facilities.to_markdown(index=False),"", "## 데이터 분할", "", "- 19개 코호트에서 1·2·3화방을 독립 데이터셋으로 분리한 뒤 `target_date` 오름차순으로 정렬", "- 동일 날짜는 나누지 않고 누적 행 수가 약 70/15/15가 되는 지점에서 Train/Validation/Test cutoff 결정", "- 변수 품질·변수군·하이퍼파라미터·최종 모델은 Train/CV와 Validation으로 결정하고 Test는 freeze 뒤 한 번 평가"]
    for truss in TRUSSES: lines += ["",f"### {KOREAN[truss]}","",split_tables[truss].to_markdown(index=False)]
    lines += ["","## 최종 선택 입력변수군","",configs[["truss","selected_feature_set","feature_count","selected_learning_model"]].to_markdown(index=False),"", "- 변수 품질 필터와 상관 중복 제거는 각 화방의 Train에서만 결정", "- 변수군 비교에는 고정 기준 설정을 사용하고, 선택된 변수군에서만 모델별 grid tuning 수행", "- `fruit_column`은 화방별 데이터에서 상수이므로 입력에서 제외"]
    lines += ["","## Validation 6종 모델 성능", "", "- 주 선정 지표는 Validation RMSE이며 MAE, R², CCC를 함께 제시", "- Persistence는 입력변수군·모델 선택에 참여하지 않는 별도 baseline", "- MLP와 TabM은 5개 seed 평균이며 다른 학습모델은 seed 42를 사용"]
    for truss in TRUSSES: lines += ["",f"### {KOREAN[truss]}","",validation[validation.truss.eq(truss)][["model","rmse","mae","r2","ccc"]].to_markdown(index=False,floatfmt=".4f")]
    lines += ["","## 최종 Test", "",tests.to_markdown(index=False,floatfmt=".4f"),"", "## 대표 예측 그래프 조건", "", "- Validation 행이 충분한 대표 사용자와 1·2·3화방이 모두 존재하는 대표 sample 하나를 데이터 기준으로 선택", "- 1·2·3화방을 1열 3행으로 표시하고 모든 subplot의 x축·y축 범위를 통일", "- Train/Validation/Test 범위를 연한 음영과 경계선으로 표시하고 작기 사이 장기 공백은 `//`로 압축", "- Actual, Persistence, Poisson, Random Forest, CatBoost, MLP, TabM을 동일 행에서 표시"]
    (out/"final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); main(p.parse_args().run_id)
