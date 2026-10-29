# AXData 재현 안내

기준: 2026-10-29. 최신 범위는 토마토 20화방·딸기 5화방의 E1~E7 127개 조합과 6개 모델입니다.

## 보존한 실험 근거

- `configs/quick_settings.json`: 1시드 quick 설정·격자·선정 규칙.
- `configs/formal_settings.json`: 3시드 formal 및 내부 시간순 CV 설정·전체 격자.
- `results/quick_frozen_winners.json`: 모델별 최종 선택 조합·하이퍼파라미터·Validation 지표.
- `results/quick_validation_comparison.csv`: 1,524개 조합 비교.
- `results/quick_verification.json`: 평가 행·예측값 검증 완료 기록.
- `data_splits.json`: 시설 ID를 제외한 Train/Validation/Test 행 수·시설 수·파일 해시.
  정확한 시설 분할 명단은 비공개 manifest에서만 복원합니다.
- `configs/feature_groups.json`: 변수군과 공통 입력의 정의.

quick은 완료됐으며 전체 대표 모델은 토마토 MLP, 딸기 TabM입니다. Validation 기준 선정이고
Test를 보고 다시 선정하지 않습니다. formal은 기록 시점에 1,524개 조합 중 642개 완료된 진행 중 실험이며,
최종 성능으로 간주하면 안 됩니다. 원래 scope의 Test 분포가 사전 탐색됐으므로 새 독립 외부검증도 아닙니다.

## 환경 및 clone 검증

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r AXData/requirements.txt
python scripts/reproduce.py smoke
```

전체 설치 환경은 `environment/requirements-observed.txt`에 보존했습니다.
GPU CatBoost·PyTorch 실행에는 적절한 CUDA 드라이버와 wheel이 필요합니다.
합성 테스트는 CPU로 실행하며 원시 DB 접속·실제 학습·게시를 수행하지 않습니다.

## 비공개 데이터 복원

```bash
python scripts/reproduce.py verify-backup /backup/AXData_focused.tar
python scripts/reproduce.py restore /backup/AXData_focused.tar \
  --overlay /backup/AXData_live_delta.tar --destination "$PWD"
```

백업 및 증분 파일과 각각의 `.sha256` 파일이 필요합니다.
증분의 최신 파일을 우선 선택하므로 기존 파일을 덮어쓰지 않고 빈 clone에 한 번에 복원합니다. 복원 대상은 원시/가공 데이터와 실험 outputs입니다.
원본은 그대로 두고 새 clone에 복원하세요. 동일하지 않은 기존 파일은 덮어쓰지 않습니다.
외부 제공 관측 데이터, 시설별 분할 명단, 행별 예측, 가중치와 상세 실행 기록은 공개 Git에서 제외합니다.

기존 실험을 **재개**하려면 전체 current campaign을 복원하고 기존 code/config/data 해시를 확인하세요.
**처음부터 다시 학습**하려면 별도 실험 clone에 원천 snapshot·semantic stream 및 입력만 복원하고,
완료된 `upper_truss_full_20261020` 출력 폴더는 복사하지 마세요. 기존 결과를 삭제하여 재실행하지 않습니다.
복구 폴더와 새 실행 폴더를 구분해야 비교 기준도 남길 수 있습니다.

```bash
export PYTHONPATH="$PWD:$PWD/AXData"
python -m AXData.ax_catalog_experiments.upper_truss_full.prepare
python -m AXData.ax_catalog_experiments.upper_truss_full.verify_setup
python -m AXData.ax_catalog_experiments.upper_truss_full.run quick
python -m AXData.ax_catalog_experiments.upper_truss_full.report quick
# 정식 재개/재현은 run formal → report formal
```

`supervise`는 자동 게시도 포함하므로 단순 재현에는 사용하지 않습니다.
처음 전처리할 때에는 `outputs/exhaustive_e1_e7_v1/data_snapshot/`,
`outputs/exhaustive_e6_semantic_v2/semantic_streams/`, 원래 작물별 입력·분할 manifest가 필요합니다.
`AXData/diagnostics/20261020_upper_truss_audit/`의 동결 관측과
`AXData/diagnostics/20261020_flower_cap_audit/verified_code_names.csv`도 필요합니다.
각 입력은 백업에 보존되어 있으며 `upper_truss_screening/prepare.py`와
`upper_truss_full/prepare.py`가 실제 연결 경로를 정의합니다.
데이터가 없을 때 임의 데이터로 대체하여 논문 지표를 재현했다고 표시하지 마세요.

## 검증 범위

clone-only 핵심 테스트 59개와 데이터 복원 후 준비파일 해시 검사로 범위를 나눕니다.
이번 작업은 진행 중 학습을 중단하지 않으므로 과거 모든 모델을 다시 학습하지 않습니다.
GPU·라이브러리 변경 시 bitwise 동일성은 보장되지 않으며 저장된 seed·split·Validation 선정 규칙과
평가 지표를 기준으로 비교해야 합니다. 과거 trial별 중간 파일은 원래 작업공간에 보존하고 집중 백업에서는 제외했습니다.
