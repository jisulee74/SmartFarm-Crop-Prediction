# AX 데이터 구축·검증 대시보드

공개 주소: https://jisulee74.github.io/SmartFarm-Crop-Prediction/ax_data_validation/

사업 개요, 관측 현황, AI 예측모델 검증, AX 구축/검증 계획의 기존 4개 탭을 유지합니다. 공개 실험 화면은 **장치 의미 통합 기준**만 표시합니다. 방향·층수 세분 코드 결과는 `technical_archive/granular_control_codes_v1/`에 보존합니다.

## 2026-10-14 게시 결과

- 127개 변수군 조합 × 8개 타깃 × 6개 모델 = 6,096건 선별 평가, 모두 성공.
- Validation: 시드 42, 타깃·모델별 기존 고정 하이퍼파라미터를 사용한 빠른 선별. 조합별 재튜닝 결과가 아닙니다. 실제 파라미터·학습 epoch는 결과에 기록합니다.
- 우승 조합: Validation RMSE → MAE → 선택 변수 수 → 조합 ID 순으로 선정. Test로 순위를 변경하지 않습니다.
- Test: 타깃·모델별 우승 조합 48개 × 시드 42/52/62 = 144건. 선별 당시 train에서 선택한 변수 목록을 동결하고 train+validation으로 재학습합니다.
- 선별 단계는 평가지표를 저장했습니다. 행별 예측 시계열은 우승 조합의 **Test**에서 조회합니다. Validation에 가짜 예측 Run ID를 부여하지 않습니다.
- KPI As-Is는 게시된 타깃별 Validation 최저 RMSE, To-Be는 그 값의 90%입니다. 신규 AX 데이터의 효과가 이미 입증됐다는 의미가 아닙니다.
- 별도 정식 추가 검증이 완료되면 동일 화면을 갱신합니다. 현재 선별 결과를 정식 3-fold CV 완료 결과로 해석하지 않습니다.
- 선별 변수 선택은 결측·중복·상관·VIF 기준을 적용한 compact 구현입니다. VIF는 작은 정규화항을 더한 상관행렬 역행렬로 계산하므로, 정식 반복 최소제곱 구현과 수치적으로 완전히 같다고 보장하지 않습니다.

## 재생성·검증

프로젝트 루트에서:

```bash
python ax_data_validation/sync_semantic_screening_dashboard.py
python ax_data_validation/sync_screening_analysis.py
python -m pytest -q ax_data_validation/test_semantic_screening_static.py
bash finalize_ax_semantic_screening_dashboard.sh
```

Python은 프로젝트 의존성이 설치된 환경을 사용합니다. 배포 스크립트는 원본 Test 완료 확인, 데이터 검증, JavaScript 구문 검사 후에만 commit/push를 수행합니다.

주요 파일:

- `cache/campaigns/semantic_control/experiment_index.json`: 6,096건 Validation과 48개 Test 요약
- `cache/campaigns/semantic_control/predictions/`: 144개 Test 예측 파일
- `cache/campaigns/semantic_control/feature_selections/`: 조합별 선택·제외 변수 및 사유
- 같은 캠페인 폴더의 `validation_rankings.csv`, `validation_winners.csv`, `test_results.csv`
- `ax_direction/dashboard_bundle.json`: 최신 선별 전체 성능과 Test 구간별 진단·분포·KPI. `screening_analysis/`에 CSV와 출처 해시를 보존합니다. 사업 스키마·원천 관측 품질 자료는 고정 원본 기준으로 유지하고 출처를 구분합니다.

원본 캠페인과 배포 전 대시보드 백업은 서버에 보존합니다. 정식 후속 배포는 선별 전에 보존한 기존 결과를 읽어, 선별 결과가 정식 결과로 섞이지 않도록 합니다.

## 진단 해석

- Validation은 저장된 집계 성능만 제공하며, 구간별 상세 오차는 최신 Test에서 제공합니다. 분할 전환 시 이전 Test 수치가 Validation 화면에 남지 않습니다.
- 현재값 유지 기준선은 미관측 현재값을 train 중앙값으로 대체해 전체 평가 행을 보존합니다. 대체 행 수를 표시하며 모델과 같은 행에서 비교합니다. 변화구간은 현재값이 관측된 행만 대상으로 하고 미관측은 별도 구간입니다.
- KPI의 RMSE 및 단순 기준선은 Validation, 변화구간 MAE는 Test 진단 참고값입니다. 신규 AX 효과 검증에서는 같은 새 코호트와 평가 조건으로 다시 기준을 고정합니다.
- 상세 가설의 기존 시설명·오차 비율을 새 Test 진단 근거로 갱신했습니다. 변수군 추가 비교는 상관적 비교이며 인과 효과를 입증하지 않습니다.
