# AXData

최신 소스 정리 기준: 2026-10-29.

공통 회귀 모델, 작물별 예측, E1~E7 조합 실험, E6 의미 통합 및 토마토 20화방·딸기
5화방 실험 코드를 포함합니다. 학습 중간 데이터와 가중치는 Git에서 제외합니다.

## 설치·실행

저장소 루트(`AXData/`의 상위 폴더)에서 실행합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r AXData/requirements.txt
export PYTHONPATH="$PWD:$PWD/AXData${PYTHONPATH:+:$PYTHONPATH}"
python -m AXData.ax_catalog_experiments --help
```

`requirements.txt`는 이번 검증 환경의 직접 의존성 버전입니다.
기존 v1.3 설치 목록은 과거 재현용으로 유지합니다.
Excel 보고서 생성에는 별도의 Excel 엔진(openpyxl 또는 XlsxWriter)이 필요하며,
이번 검증에서는 Excel 보고서를 새로 생성하지 않았습니다.

DB 추출에는 `FARMSTOM_DB_HOST`, `FARMSTOM_DB_PORT`, `FARMSTOM_DB_USER`,
`FARMSTOM_DB_PASSWORD`, `FARMSTOM_DB_NAME` 등의 로컬 환경 설정이 필요합니다.
인증정보를 코드에 넣거나 Git에 추가하지 마세요. 실제 데이터·모델 복구는
[복구 안내](../docs/RESTORE.md)를 참고하세요.

## 실험 안내

- [기본 범주 조합](ax_catalog_experiments/README.md)
- [전수실험](ax_catalog_experiments/exhaustive/README.md)
- [E6 의미 통합](ax_catalog_experiments/exhaustive_e6_semantic/README.md)
- [상위 화방 간략실험](ax_catalog_experiments/upper_truss_screening/README.md)
- [20/5화방 전수실험](ax_catalog_experiments/upper_truss_full/README.md)

기존 shell 실행기는 원래 서버 경로·환경을 전제로 합니다. 새 컨테이너에서 사용할 때에는
경로, GPU, DB 연결, 출력 폴더를 먼저 확인하고 완료된 실행을 덮어쓰지 않도록 재개하세요.

## 테스트

```bash
PYTHONPATH=.:AXData OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  AXData/common_regression/tests AXData/ax_catalog_experiments/tests \
  AXData/tomato_flower_count_prediction/tests -q
```

실험 결과는 `ax_catalog_experiments/outputs/` 등 원래 경로에 복원합니다.
대시보드 표시 파일은 저장소 루트의 각 웹 폴더에 별도로 유지합니다.
