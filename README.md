# SmartFarm Crop Prediction
 
토마토 꽃 수·딸기 착과수 관측 대시보드와 예측 실험 코드 및 결과를 관리하는 저장소입니다.

- **웹 대시보드 배포 주소**: [https://jisulee74.github.io/SmartFarm-Crop-Prediction/](https://jisulee74.github.io/SmartFarm-Crop-Prediction/)

## 폴더 구성

```text
flower_fruit_prediction/                  웹 대시보드와 관측 CSV 두 파일
AXData/
  common_regression/                      공통 모델·전처리·검증 코드
  ax_catalog_experiments/                 변수 사전·범주 조합 실험·통합 결과
  tomato_flower_count_prediction/         토마토 전체 꽃 수 예측
  strawberry_fruit_set_prediction/        딸기 화방별 착과수 예측
  strawberry_flower_cluster_prediction/   기존 딸기 화방수 예측
```

## 웹 대시보드 실행

저장소 루트에서 실행합니다.

```bash
python flower_fruit_prediction/server.py
```

터미널에 표시된 localhost 주소로 접속합니다. 상세 기능은 [웹사이트 README](flower_fruit_prediction/README.md)를 참고하세요. 로컬 서버 실행 자체가 외부 배포를 의미하지는 않습니다.

## 관측 데이터

- [토마토 개체별 관측 CSV](AXData/ax_catalog_experiments/outputs/timeseries/tomato_observed_counts_by_crop_cycle.csv)
- [딸기 개체별 관측 CSV](AXData/ax_catalog_experiments/outputs/timeseries/strawberry_observed_counts_by_crop_cycle.csv)

두 CSV는 유지하며 웹사이트 폴더에도 동일한 사본을 둡니다. 데이터를 갱신할 때에는 웹사이트 사본도 함께 갱신하세요. 개체는 시설·작기·sample_num을 함께 사용해 구분합니다. 관측값은 예측값과 다르며, 토마토 전체 꽃 수는 1~3화방만의 합으로 대체하지 않습니다.

## 실험 코드와 결과

환경 설치·실행은 [AXData 안내](AXData/README.md), 범주 조합 실험은 [실험 안내](AXData/ax_catalog_experiments/README.md)를 참고하세요.

[통합 Excel 보고서](AXData/ax_catalog_experiments/outputs/AX_Experiment_Report.xlsx)와 CSV·행별 예측·실행 기록은 실험 폴더의 outputs에 보관합니다. 과거 manifest와 실행 기록에 남은 경로는 당시 환경의 기록입니다.

이번 폴더 정리는 데이터를 변경하거나 실험을 다시 수행하지 않습니다. 보관된 일부 모델 결과에는 예측값의 수치적 불안정성이 있으므로, 실행 완료를 모델 품질 검증 완료로 해석하지 마세요.
