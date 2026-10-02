# 🌿 꽃수·착과수 전수실험(127개 변수군 조합) 대시보드

토마토 꽃 수(4종)와 딸기 착과수(4종)에 대한 관측 시계열 데이터 탐색, **8개 타깃 × 6개 모델 × 127개 입력 변수군 조합(E1~E7) × 3개 시드(42, 52, 62)** 전수실험 결과 비교, 그리고 AX 방향성 분석을 통합 제공하는 반응형 스마트팜 웹 대시보드입니다.

- **온라인 배포 주소 (GitHub Pages)**: [https://jisulee74.github.io/SmartFarm-Crop-Prediction/flower_fruit_prediction_exhaustive/](https://jisulee74.github.io/SmartFarm-Crop-Prediction/flower_fruit_prediction_exhaustive/)
- **기존 예시 사이트**: [https://jisulee74.github.io/SmartFarm-Crop-Prediction/flower_fruit_prediction/](https://jisulee74.github.io/SmartFarm-Crop-Prediction/flower_fruit_prediction/)

---

## 📌 주요 화면 구성 (3개 탭)

1. **📊 관측 현황 (Observation Views)**
   - **8개 시계열 시각화 (Apache ECharts)**: 토마토 1·2·3화방 및 전체 꽃 수 / 딸기 1·2·3화방 및 전체 착과수
   - **2가지 분석 모드**: 개체당 평균 모드 및 개체별 상세 추이 모드
   - **다차원 연계 필터**: 작물, 시설, 작기, 개체번호, 날짜 범위 선택

2. **🧪 실험 결과 비교 (127개 조합 전수실험 탐색 및 비교)**
   - **127개 입력 변수군 조합 전수 탐색 ($2^7 - 1 = 127$)**:
     - **변수군 수 분류**: 단독 1개(7), 2개(21), 3개(35), 4개(35), 5개(21), 6개(7), 7개 전체(1)
     - **포함 변수군 다중 태그 필터**: E1(목표·생식), E2(영양생장), E3(시설내환경), E4(외부기상), E5(근권양액), E6(시설제어), E7(시설작기맥락)
     - **실시간 검색**: 변수군 코드, 한글 명칭, 세부 변수명 실시간 필터링
     - **127개 조합 전체 탐색기 모달**: 전체 127개 조합을 반응형 카드로 한눈에 조회하고 1-클릭 선택
     - **세부 변수 사전 모달**: 선택된 조합에 포함된 49개 변수 정의서 및 고정 메타데이터(C0) 조회
   - **8개 타깃 × 6개 모델 × 3개 시드 (42, 52, 62) + 앙상블 평균**:
     - Poisson, Random Forest, CatBoost, MLP, TabM, TFT
   - **상태 분리 및 무결성 보장**:
     - 아직 실행되지 않은 조합은 임의 수치 없이 명확히 `⏳ 결과 대기 (0/3)`로 표시
     - 실제 결과 파일이 확인된 조합만 성능 지표 및 순위 랭킹 부여
     - 시설·작기·개체별 실제값 vs 모델 예측값 시계열 추이 (관측 현황과 동일한 UI 및 조작 방식)

3. **📋 생육 예측 분석 (AX Direction & Strategy)**
   - 핵심 요약 KPI, 베이스라인 대비 개선도, 오차 취약 구간 진단, 데이터 수집 가이드 및 스키마 로드맵

---

## 🚀 로컬 실행 방법

Python 3 환경에서 즉시 실행할 수 있습니다.

```bash
# 폴더 이동
cd flower_fruit_prediction_exhaustive

# 로컬 웹서버 실행
python server.py
```

브라우저에서 **`http://localhost:8000`** 으로 접속합니다.

---

## 🔄 추후 전수실험 결과 연결 및 갱신 절차

전수실험(`run_ax_exhaustive.sh`)이 완료되면 아래 명령어로 웹 대시보드 데이터를 즉시 갱신할 수 있습니다.

### 1. 실험 결과 인덱스 자동 빌드
```bash
python sync_exhaustive_experiments.py
```
- `ax_catalog_experiments/outputs/exhaustive_e1_e7_v1/`의 `csv/` 및 `predictions/` (또는 `handoff/dashboard_bundle.json`)을 자동으로 파싱하여 `cache/experiment_index.json` 및 세부 예측 JSON을 생성합니다.
- 서버가 실행 중인 상태라면 웹 화면 상단의 `[🔄 최신 실험 결과 새로고침]` 버튼을 눌러 브라우저 캐시를 즉시 갱신할 수도 있습니다.

### 2. 무결성 검증
```bash
python test_verification.py
```
- 127개 조합 카탈로그 무결성, 8개 타깃 및 6개 모델, 3개 시드, 관측 데이터 행 수 및 API 응답을 자동 검증합니다.

---

## 📁 프로젝트 구조

```text
flower_fruit_prediction_exhaustive/
├── index.html                               # 메인 대시보드 마크업 (127개 조합 탐색기 & 모달 포함)
├── server.py                                # 로컬 정적 웹서버 및 REST API (Python 3)
├── sync_exhaustive_experiments.py           # 전수실험 산출물 파싱 및 웹 캐시 빌더
├── test_verification.py                     # 127개 조합 무결성 및 엔드포인트 검증 스크립트
├── tomato_observed_counts_by_crop_cycle.csv # 토마토 실제 관측 데이터 (2,239행)
├── strawberry_observed_counts_by_crop_cycle.csv # 딸기 실제 관측 데이터 (5,218행)
├── cache/
│   ├── combination_catalog.json             # 127개 조합 및 49개 변수 정의 사전
│   ├── experiment_index.json                # 전수실험 결과 인덱스
│   └── predictions/                         # 개별 Run 예측 시계열 캐시
├── src/
│   ├── app.js                               # 메인 앱 진입점
│   ├── parser.js                            # 관측 CSV 파서
│   ├── analytics.js                         # 관측 데이터 통계 엔진
│   ├── charts.js                            # 관측 8종 차트 렌더러
│   ├── experiment_api.js                    # 127개 조합 카탈로그 및 실험 결과 API 클라이언트
│   ├── experiment_view.js                   # 127개 조합 탐색/필터/모달/비교표 뷰 컨트롤러
│   ├── analysis_api.js                      # AX 분석 API 클라이언트
│   ├── analysis_view.js                     # AX 방향성 분석 뷰 컨트롤러
│   └── styles.css                           # 프리미엄 디자인 시스템 및 127 조합 탐색기 스타일
└── README.md
```

## 시설 제어 변수 구성 기준 비교

실험 결과 비교 화면은 내부 캠페인 ID 대신 다음 두 이름을 사용합니다.

- **시설별 제어코드 기준**: DB에 저장된 시설별 제어 코드를 그대로 구분해 수행한 최초 전수실험입니다. 기존 `exhaustive_e1_e7_v1` 결과 파일과 대시보드 캐시는 그대로 보존합니다.
- **장치 의미 통합 기준**: 방향·층수 표현을 제거하고 장치 종류와 측정값 의미를 유지해 E6 포함 유효 조합 16개를 다시 학습합니다. E6가 없는 유효 조합 15개는 기존 완료 결과를 재사용합니다.

두 실험을 나눈 이유는 완료된 최초 결과를 기준선으로 보존하면서, 시설 간 코드 표현 차이가 E6 데이터 연결률과 예측 성능에 미친 영향을 같은 분할·모델·시드 조건에서 검증하기 위해서입니다. 의미 통합 실험이 진행 중일 때는 3개 시드가 모두 끝난 결과만 순위에 포함하며 나머지는 `학습 진행 중` 또는 `실험 예정`으로 표시합니다.

현재 결과 캐시는 다음 명령으로 다시 생성합니다.

```bash
python flower_fruit_prediction_exhaustive/sync_e6_semantic_dashboard.py
```

실험 완료 후에도 같은 명령만 다시 실행하면 남아 있던 조합의 결과와 최종 상태가 기존 화면 구조에 자동 반영됩니다.
