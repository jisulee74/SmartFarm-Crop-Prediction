# E1–E7 전수조합 실험

`C0`는 연결·분할 메타데이터로 고정하고 `E1`부터 `E7`까지의 비공집합 127개 조합을 8개 타깃과 6개 모델에 적용합니다. DB 스냅숏, 변수 선택, 시간순 CV, 3개 시드 검증, 우승 조합 테스트 및 Antigravity 전달 파일을 하나의 재개 가능한 캠페인으로 관리합니다.

저장소 루트에서 다음 한 명령을 사용합니다. 명시적인 `FARMSTOM_DB_PASSWORD`가 없으면 기존 `GEAS/GEAS3.5/.env`의 `GEAS_DB_*` 접속 설정을 자동으로 읽습니다. 셸에 지정된 `FARMSTOM_DB_*` 값은 항상 우선하며 비밀번호는 산출물이나 로그에 기록하지 않습니다.

```bash
bash run_ax_exhaustive.sh
```

같은 명령을 다시 실행하면 동일 해시의 완료 실행을 건너뛰고 중단 지점부터 재개합니다. 별도 상태 확인과 안전한 중단 요청은 다음과 같습니다.

```bash
../envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments.exhaustive status
../envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments.exhaustive stop
```

출력은 `ax_catalog_experiments/outputs/exhaustive_e1_e7_v1/`에 생성됩니다. 데이터·코드·정책 해시가 바뀌면 같은 출력 경로로 재개하지 않습니다. DB 스키마나 추출이 불완전하면 모델 학습 전에 중단합니다. DB 행은 서버 측 스트리밍 커서로 기본 100,000행씩 읽어 임시 Parquet에 순차 기록하며, `snapshot_progress.json`에 현재 테이블과 누적 행 수를 저장합니다. 측정 테이블은 `2024-01-01` 이후만 포함하고, 작기 테이블은 이 기간과 겹치는 작기만 포함합니다.

환경변수 `AX_GPU_WORKERS`는 1, 2, 4 중 하나로 자동 결정값을 덮어쓸 수 있고 `AX_CPU_WORKERS`, `AX_RF_THREADS`, `AX_GPU_HOST_THREADS`로 CPU 사용량을 조정할 수 있습니다. CUDA가 없으면 CPU로 대체하지 않습니다.
