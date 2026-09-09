# 화방수 공통 회귀 모델 v1.3

현재 조사 완료 직후 동일 개체의 다음 화방수를 예측한다. 기존 구현과 결과는
최종 화방수 파이프라인과 결과를 이 폴더에서 관리한다.

```bash
python AXData/strawberry_flower_cluster_prediction/run_model_suite_v1_3.py --run-id <RUN_ID> --through validation
```

`--through test`는 Validation 선택과 구성 freeze, 최종 재학습 뒤에만 Test를 한 번 연다.
