# AXData

작물 관측 데이터, 예측 실험 코드, 변수 사전과 실행 결과를 보관합니다.

## 구성

| 폴더 | 내용 |
|---|---|
| common_regression | 공통 전처리·모델 어댑터·지표·검증 |
| ax_catalog_experiments | 변수 사전, 범주 조합 실험, 통합 결과 |
| tomato_flower_count_prediction | 토마토 전체 꽃 수 예측 |
| strawberry_fruit_set_prediction | 딸기 1·2·3화방 착과수 예측 |
| strawberry_flower_cluster_prediction | 기존 딸기 화방수 예측 |

## 환경과 실행

저장소 루트에서 Python 3.11 환경을 준비합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r AXData/common_regression/requirements-v1.3.txt
python -m AXData.ax_catalog_experiments --help
```

상세 명령은 [범주 조합 실험 README](ax_catalog_experiments/README.md)를 참고하세요.

## 결과 위치

- [통합 Excel](ax_catalog_experiments/outputs/AX_Experiment_Report.xlsx)
- [관측 시계열 CSV](ax_catalog_experiments/outputs/timeseries/)
- [전체 실험 산출물](ax_catalog_experiments/outputs/): CSV, prepared 데이터, runs 실행 기록 등

기존 결과와 관측 데이터는 보존합니다. 일부 과거 예측의 수치적 불안정성은 폴더 정리로 해결되지 않았으며 재학습 결과와 구분해야 합니다.

## DB와 제외 파일

DB 추출에는 FARMSTOM_DB_HOST, FARMSTOM_DB_PORT, FARMSTOM_DB_USER, FARMSTOM_DB_PASSWORD, FARMSTOM_DB_NAME 등 연결 환경 설정이 필요합니다. 인증정보는 저장소에 포함하지 않습니다.

Python 캐시, 로컬 환경, 모델 가중치 model.bin, 상세 로그와 GitHub 단일 파일 제한을 넘는 토마토 원시 환경 캐시는 제외합니다. 기존 결과 조회에는 DB가 필요하지 않지만 원천 데이터 재구축에는 접속 설정과 누락된 캐시 재생성이 필요합니다.
