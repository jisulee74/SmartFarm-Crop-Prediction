# 안정화 재실험 v2

토마토와 딸기의 다음 조사 시점 1·2·3화방 수와 세 화방 합계를 각각 예측합니다. 합계 타깃은 독립 모델이며, 모든 관측 화방의 전체 꽃 수와 다릅니다. 기존 결과와 관측 CSV는 수정하지 않습니다.

저장소 루트에서 Python 3.11 환경을 활성화한 뒤 실행합니다. 이 작업공간에서는 `python` 대신 `envs/jslee_py311/bin/python`을 사용할 수 있습니다.

```bash
python -m AXData.ax_catalog_experiments.stability prepare
python -m AXData.ax_catalog_experiments.stability diagnose
python -m AXData.ax_catalog_experiments.stability verify
python -m AXData.ax_catalog_experiments.stability run-all
python -m AXData.ax_catalog_experiments.stability report
```

진행률 확인:

```bash
python -m AXData.ax_catalog_experiments.stability status
watch -n 60 'python -m AXData.ax_catalog_experiments.stability status'
tail -f AXData/ax_catalog_experiments/outputs/stability_v2/campaign.log
```

`run-all`은 네 실패 사례 진단이 현재 정책으로 통과한 경우에만 시작합니다. 같은 명령으로 재개하면 완료된 실행은 캐시에서 읽습니다. 중단된 실행은 다시 수행하며 실패·불안정·미수렴 실행은 별도 결과로 보존합니다. 동시 캠페인은 파일 잠금으로 막습니다. 코드·정책·데이터 해시가 달라지면 기존 결과에 이어 쓰지 않습니다.

## 고정된 정책과 근거

- 학습 fold만으로 중앙값 대체, 평균 중심화, `max(std, 1)` 스케일링, 변환값 `[-10, 10]` 제한을 적용합니다. 변수 열은 제거하지 않습니다.
- MLP·TabM·TFT는 유한한 loss/output/gradient를 확인하고 gradient norm을 1로 제한합니다. TFT는 등록 입력 4회 과거 시퀀스를 사용하며 추가 타깃·시간 설명변수는 넣지 않습니다.
- 정책 2.0의 입력 안정화만으로는 딸기 3화방 Poisson `alpha=0`의 지수 출력 폭발이 남았습니다. **정책 2.1에서는 최소 L2 alpha=0.1**을 적용합니다. 기존 그리드 `[0, 0.1, 1]`은 유효값 `[0.1, 1]`로 중복 제거합니다. 요청/유효 설정 및 실패한 2.0 진단을 보존합니다.
- 화방별 상한은 학습 현재·타깃 관측의 최대값, 합계 상한은 학습 화방별 최대값의 합입니다. 이는 생물학적 절대 상한이 아닙니다.
- 원시·제약 적용 예측과 지표를 모두 저장합니다. 비유한 값 또는 상한의 100배를 넘는 원시 출력은 선택 불가입니다. Poisson 미수렴도 선택 불가입니다.
- 모든 기존 신규/legacy 변수군을 작물별 네 타깃에 대응시킵니다. 타깃 관측 이력은 새 타깃으로 계산하며 legacy 온도도 원천 캐시에서 구합니다. 새 버전의 환경 구간은 `[t-days,t)`입니다.
- 시설 분할은 기존 배정을 유지합니다. 동일 타깃의 평가 행을 모델/변수군별로 삭제하지 않습니다. 누락된 화방은 0으로 만들지 않으며 세 화방이 모두 유효할 때 합계를 만듭니다.
- 모든 그룹 × 6개 모델을 내부 3-fold로 튜닝하고 5개 seed로 외부 검증합니다. 최종 설정을 동결한 뒤 테스트를 읽습니다. 이미 확인한 과거 holdout이므로 새로운 독립 테스트라고 해석하지 않습니다.

## 산출물

`../outputs/stability_v2/` 아래에 준비 데이터와 원천 해시, `Group_Mapping.csv`, 변수 구성·품질·제외 행, 진단, 실행별 예측 Parquet, 통합 Excel/CSV를 보관합니다. 이전 준비 시도와 실패 정책 진단은 비교를 위해 보존합니다.

`Results`는 long format입니다. `Prediction_Variant`의 `raw`/`bounded`, `Eval_Split`, 타깃, seed를 구분해 보세요. 실패 실행은 `Runs`에서 확인하고, 없는 지표를 0으로 해석하지 마세요. 보고서는 모델 하나의 전체 변수군 평가가 끝날 때와 타깃 완료 시 갱신됩니다. 중간에도 `report` 명령으로 생성할 수 있습니다.

`status`의 예정·대기 실행 수는 상한입니다. 모든 후보가 안정적으로 튜닝되지 않으면 후속 검증·테스트는 제외될 수 있습니다. 잔여시간은 모든 모델의 내부 CV와 외부 검증 시간이 관측된 후에만 거친 추정치로 제공합니다.

## 병렬 재개

기존 과학적 엔진은 수정하지 않고 타깃별 독립 프로세스로 실행합니다. 각 타깃 내부의 전체 변수군 튜닝, 5개 seed 검증, 선택 동결 후 테스트 순서는 동일합니다. 같은 타깃은 동시에 두 작업자가 처리하지 않습니다. 초기 권장 동시 작업 수는 2이며, 속도 향상 폭은 실제 처리량과 공유 GPU 상황에 따라 달라집니다.

```bash
python -m AXData.ax_catalog_experiments.stability.parallel check
python -m AXData.ax_catalog_experiments.stability.parallel run --workers 2
```

백그라운드 실행(작업공간 Python 경로 사용):

```bash
nohup envs/jslee_py311/bin/python -u -m AXData.ax_catalog_experiments.stability.parallel run --workers 2 > AXData/ax_catalog_experiments/outputs/stability_v2/parallel_campaign.log 2>&1 &
```

진행률과 작업자별 현재 학습:

```bash
watch -n 60 'envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments.stability.parallel status'
```

안전한 중단 요청:

```bash
python -m AXData.ax_catalog_experiments.stability.parallel stop
```

현재 진행 중인 한 번의 실행 결과를 저장한 뒤 각 작업자가 멈춥니다. 완전히 중단된 뒤 같은 `run --workers 2` 명령으로 재개합니다. `parallel/scheduler.json`의 `Status=paused`와 `Active={}`로 중단 완료를 확인할 수 있습니다. 작업자 로그는 `parallel/logs/<target>.log`에, 상태는 `parallel/workers/<target>.json`에 저장됩니다. 기존 순차 실행과 병렬 실행은 동시에 시작할 수 없습니다. 통합 Excel은 관리 프로세스가 타깃 종료 시 갱신하며 필요할 때 기존 `report` 명령을 사용할 수 있습니다. 병렬 실행에서는 기존 순차 실행 평균으로 계산된 잔여시간을 표시하지 않습니다.
