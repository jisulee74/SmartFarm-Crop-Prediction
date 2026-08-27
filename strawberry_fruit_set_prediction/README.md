# 화방별 독립 착과수 회귀 모델

기존 `fruit_set_prediction`을 변경하지 않고 1·2·3화방을 독립 target으로 학습한다.
각 화방 안에서 동일 target 날짜를 나누지 않으면서 누적 행 수가 약 70/15/15가
되도록 시간순 Train/Validation/Test를 구성한다.

```bash
python AXData/strawberry_fruit_set_prediction/run_experiment.py --run-id final_v1
```

결과는 `artifacts/<run-id>`에 저장된다.
