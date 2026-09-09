# Power BI 보고서 설계

## 데이터 모델

csv/Experiments → Runs → Results를 각각 1:N 단방향으로 연결합니다.
- Experiments.Experiment_ID → Runs.Experiment_ID
- Runs.Run_ID → Results.Run_ID

변수군 차원은 Variable_Groups에서 Variable_Group_ID의 고유 목록으로 만듭니다. 그룹 차원 → Experiments, 그룹 차원 → Variable_Groups, Category_Catalog → Variable_Groups를 1:N 단방향으로 둡니다. Category_Catalog → Category_Variables 및 Variable_Catalog → Category_Variables도 1:N 단방향입니다. 후보 사전의 Category_ID는 탐색 속성으로만 사용하고 이중 경로를 만들지 않습니다.

범주 선택으로 모델 결과를 필터링할 때는 Variable_Groups의 선택된 그룹 ID를 TREATAS로 Experiments.Variable_Group_ID에 전달합니다. 무분별한 양방향 관계를 사용하지 않습니다. 여러 범주 선택은 기본 OR이며 AND 필터가 필요하면 선택 범주 수와 그룹의 일치 범주 수를 비교한 별도 그룹 집합을 사용합니다.

Group_Detail은 사람이 읽는 확장 조회표입니다. 결과 fact에 직접 연결해 복제하지 않습니다. 내부 CV, screening, legacy, smoke 결과는 최종 모델 순위 화면에서 기본 제외합니다.

## 지표

기본 Measure는 동일 타깃·지표·평가구간·출처·실험·모델에서만 seed 평균을 계산합니다. 개별 Run 또는 Seed를 선택하면 원래 값을 보여줍니다. CV의 여러 fold와 seed를 함께 평균하지 않습니다.

```dax
Selected Metric =
IF (
    HASONEVALUE(Results[Metric_Name]) &&
    HASONEVALUE(Results[Prediction_Target]) &&
    HASONEVALUE(Results[Crop_Target]) &&
    HASONEVALUE(Results[Eval_Split]) &&
    HASONEVALUE(Results[Result_Origin]) &&
    HASONEVALUE(Results[Experiment_ID]) &&
    HASONEVALUE(Results[Model_Type]),
    AVERAGE(Results[Metric_Value]),
    BLANK()
)
```

개선율은 동일 타깃·dataset·split·eval_row_hash·평가구간의 Persistence와 비교합니다. (baseline RMSE - candidate RMSE)/baseline RMSE이며 baseline=0 또는 대응 baseline 부재 시 BLANK입니다. R2와 CCC를 개선율에 혼용하지 않습니다. 학습모델들의 평균 RMSE는 예측 앙상블 RMSE와 다르므로 seed 평균이라고 표시합니다.

Legacy는 원본 보고서 집계 방식 또는 seed 수만 보존된 값입니다. seed별 결과로 해석하지 않고 별도 페이지에서 조회합니다. Results의 Metric_Status=undefined는 결측이며 0으로 표시하지 않습니다.

## 화면

1. 실험 개요: 작물·타깃·데이터버전·평가구간 필터, 최종 선택 모델/그룹, RMSE·MAE·R2·CCC 카드, baseline 개선율. Best_Configurations는 validation 선택을 의미하며 test 성능으로 재선택하지 않습니다.
2. 모델·변수군 비교: 그룹×모델 행렬, 지표 단일 선택, seed 평균/표준편차, 표본수. 서로 다른 타깃의 원시 RMSE를 합산하지 않습니다.
3. 범주 구성과 변화: 그룹×범주 포함 행렬, Group_Detail drill-through, 범주 한 개 차이의 paired seed RMSE 변화. 튜닝을 포함한 예측 성능 연관성이지 Action의 인과 효과가 아닙니다.
4. 데이터 품질: 원천 상태, train 결측률, 제외 사유, 표본수, target_status 및 실패 Run. 미확인 코드와 DB 미수집 상태를 명확히 표시합니다.

## 수용 검증

같은 Run_ID의 결과는 범주를 펼쳐도 4개 지표 행으로 유지되어야 합니다. category bridge와 fact를 직접 join하지 않습니다. 같은 모델·그룹의 원본 CSV 값과 카드 값 대조, multiple metric 선택 시 빈 카드, 실패/undefined≠0, legacy/new 분리, baseline=0 분모 처리 등을 확인합니다.

Power BI Desktop 파일 제작과 서비스 게시를 포함하지 않습니다.
