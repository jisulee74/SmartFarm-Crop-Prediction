# 착과수 시설 ID holdout 실험

- 입력변수: 기존 fixed-13과 동일
- 모델 후보: Poisson, Random Forest, CatBoost, MLP, TabM, TFT
- Persistence: 별도 baseline
- 분할: 시설 ID 완전 분리; 행 비율이 70:15:15에 가장 가까운 조합
- 1화방 10/3/3개 활성 시설, 2·3화방 13/3/3개 시설

## 최종 Test

| truss             | model       |   rmse |    mae |      r2 |    ccc |
|:------------------|:------------|-------:|-------:|--------:|-------:|
| first_fruits_num  | tft         | 1.9207 | 1.5824 |  0.5693 | 0.8038 |
| first_fruits_num  | persistence | 2.5091 | 1.7500 |  0.2651 | 0.6890 |
| second_fruits_num | poisson     | 1.9693 | 1.5789 |  0.3296 | 0.4312 |
| second_fruits_num | persistence | 2.6296 | 1.8936 | -0.1952 | 0.4861 |
| third_fruits_num  | tft         | 1.7943 | 1.3291 |  0.5019 | 0.7131 |
| third_fruits_num  | persistence | 2.5907 | 1.8269 | -0.0385 | 0.5151 |

Test는 Validation으로 최종 모델을 고정한 후 한 번만 접근했다.