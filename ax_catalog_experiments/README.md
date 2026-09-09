# AX 변수 사전와 범주 조합 실험

신규 구현은 이 디렉터리에 격리됩니다. 과거 FlowerA/FruitA 및 모델 결과는 수정하지 않습니다.

## 실행

작업공간 루트에서 기존 Python 3.11 환경을 사용합니다.

```bash
envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments build
envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments snapshot-db
envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments run --target tomato_total_flower --profile smoke
envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments run --target tomato_total_flower --profile full
envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments run-all --profile full
envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments report
envs/jslee_py311/bin/python -m pytest AXData/ax_catalog_experiments/tests -q
```

`build`는 로컬 후보 사전, 토마토 원천 재구축, 학습 기반 범주 동결, 기존 결과 수입 및 Excel/CSV를 생성합니다. `run`은 실제 학습입니다. smoke는 기본 범주 Poisson의 실행 검증만 수행하며 순위 대상이 아닙니다. full은 3개 모델 탐색 후 최대 6개 조합에 6개 모델·튜닝·5개 seed를 적용하므로 상당한 계산 시간이 필요합니다. 작업별 결과가 저장되어 동일 명령으로 재개할 수 있습니다. run-all은 네 타깃을 순차 실행하고 타깃별로 보고서를 갱신하며 campaign_status.json에 진행 상태를 기록합니다. 실패 기록도 보존하므로 코드·환경 수정 후에는 새 `--output` 디렉터리를 사용합니다.

`--output` 기본값은 이 디렉터리의 `outputs`입니다. 정의·데이터·설정 변경 시 기존 ID를 덮어쓰지 않고 새 출력 경로를 사용합니다. frozen definition이 달라지면 실행을 거부합니다.

## 현재 구현의 경계

- DB 연결은 환경변수 FARMSTOM_DB_HOST/PORT/USER/PASSWORD/NAME을 사용합니다. 비밀값은 출력하거나 보고서에 저장하지 않습니다.
- `snapshot-db`는 전체 컬럼 메타데이터만 읽습니다. 자동으로 모든 업무 테이블의 값 의미나 연결 가능성을 확정하지 않습니다.
- 로컬 소스 사전은 기존 스키마·센서 코드 사전의 범위입니다. 병해충·인력제약의 확인되지 않은 변수는 생성하지 않습니다.
- 신규 토마토 데이터는 보존된 raw growth/environment에서 재구축합니다. 외부환경 등 raw 캐시에 없는 항목은 train missing 판정으로 제외됩니다. 이 제외는 센서가 DB에 없다는 뜻이 아닙니다.
- 딸기는 보존된 analysis_rows.csv 감사 관측표에서 이력을 재계산합니다. 최초 조사/양수 관측은 해당 감사 코호트 내의 최초 관측이며 DB 전체 이력의 최초 시점은 아닙니다. 원본 감사의 날짜·NULL·충돌·작기 배정 필터를 manifest에 기록합니다. 시설 연결은 사용자별 유일성을 검사합니다.
- 누적 계측값, 원형 변수, 단위 미확인 생육·양액 항목은 needs_review로 남습니다. 현재 파생 엔진은 실수 입력용이며 범주형 Context 인코딩은 후속 구현이 필요합니다.
- DB 전체 값 조사와 신규 실시간 원천 추출은 로컬 코호트 재구축과 별개입니다. 전체 모집단 확정 여부와 4개 타깃 학습 완료 여부는 manifest/status로 구분합니다.

## 입력 계약

`data.save_prepared`는 row_id, facility_id, feature_date, target_date, target, 현재 타깃 및 파생변수 컬럼과 원천 provenance를 받습니다. 날짜는 datetime, row_id는 유일, target_date는 feature_date 이후여야 합니다. `manifest.json`에 시설 분할, 데이터 해시, 동결 범주 목록을 저장합니다.

파생변수의 2/3/4회 통계는 현재 포함, 충분한 이력이 있을 때만 산출, std ddof=0입니다. 센서는 [t-days,t)이며 유효 이진값은 0/1만 인정합니다. 평균 작동값은 가동시간이 아니라 관측 중 활성 비율입니다.

후보 중 train 결측률 >50%, 상수, 이전 채택 변수의 affine 결합인 항목을 제외합니다. 범주 내부 전체 포함은 이 screening 후 동결된 명시적 리스트를 의미합니다. 센서별 물리 의미 확인과 상관계수 검토는 별개의 과정입니다.

## 평가

시설 분리 고정, 조합별 평가 행 동일, 타깃/그룹/모델/seed/stage/fold/설정별 Run_ID, 평가 행 해시를 기록합니다. 내부 CV는 feature_date 순서로 분할하고 fit target_date가 검증 시작보다 이른 행만 남깁니다. 음수 예측은 0으로 제한하고 반올림하지 않습니다.

TFT는 등록 feature의 이력만 사용하고 target을 추가 설명변수로 사용하지 않습니다. sequence_id/time_idx는 시퀀스 구조이며 모델 실수 입력 목록에는 포함하지 않습니다. TFT backend의 encoder length와 padding 규칙을 사용합니다.

선택은 평균 per-seed validation RMSE → MAE → 입력 수 → 그룹 ID입니다. 최종 모델별 조합을 selected.json에 고정한 후 test를 엽니다. 결과표에서 `screening`, `internal_cv`, `validation`, `test`를 구별합니다.

Excel은 표준 OOXML로 생성해 추가 Excel 라이브러리가 필요 없습니다. 원본 CSV, manifest, 예측 Parquet가 감사 기준이며 Excel은 조회용입니다.

## 보고서의 짧은 ID

Excel·CSV의 Experiment_ID는 EXP_001, 변수군은 VG_Flower_001 / VG_Fruit_001, 실행은 RUN_001, 범주는 CAT_001 형식으로 표시합니다. 숫자는 성능 순위가 아닙니다. display_id_registry.json이 원래 ID와 표시 ID의 연결을 보존하므로 보고서를 다시 생성해도 기존 번호는 바뀌지 않습니다. 이 파일을 삭제하지 마세요. ID_Mapping 시트에서 원래 실행 폴더·manifest ID를 찾을 수 있습니다. 모델·예측 Parquet·학습 결과 JSON은 원래 ID를 유지합니다.
