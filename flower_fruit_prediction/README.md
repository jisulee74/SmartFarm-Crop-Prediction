# 🌿 꽃수·착과수 예측 대시보드 (SmartFarm Crop Growth & Prediction Dashboard)

토마토 꽃 수와 딸기 착과수의 실제 관측 시계열 데이터 탐색, AI 모델별 실험 결과 비교, 그리고 AX 방향성 심층 분석을 통합 제공하는 반응형 스마트팜 웹 대시보드입니다.

- **온라인 배포 주소 (GitHub Pages)**: [https://jisulee74.github.io/SmartFarm-Crop-Prediction/](https://jisulee74.github.io/SmartFarm-Crop-Prediction/)

---

## 📌 주요 화면 구성 (3개 탭)

1. **관측 현황 (Observation Views)**
   - **8개 독립 시계열 시각화 (Apache ECharts)**: 토마토 1·2·3화방 및 전체 꽃 수 / 딸기 1·2·3화방 및 전체 착과수
   - **2가지 분석 모드**: 개체당 평균 모드 및 개체별 상세 추이 모드
   - **다차원 연계 필터**: 작물, 시설, 작기, 개체번호, 날짜 범위 선택

2. **실험 결과 비교 (Experiment Comparison)**
   - **실제 실험 데이터 기반**: 안정성 검증 v2(stability_v2) 8개 타깃, 6종 모델(TabM, CatBoost, RandomForest, MLP, Poisson, TFT)
   - **입력변수군 조합별 성능 비교표**: Validation Bounded RMSE/MAE, 피처 수, 선정 그룹 랭킹
   - **예측 추이 vs 실제 관측치 시각화**: 개별 시드 및 5개 시드 앙상블 평균 지원

3. **생육 예측 분석 (AX Direction & Strategy)**
   - **핵심 요약 KPI**: 최적 모델 조합 성능, 앙상블 개선율, 오차 집중도
   - **베이스라인 vs 모델 성능 심층 비교**: 오차 분포 및 지속성 모델 대비 개선도
   - **오차 취약 구간 진단**: 생육 일수(정식 후 경과일) 및 주간 변화량 기반 오차 집중도 분석
   - **AX 전환 전략 및 데이터 수집 가이드**: 현장 조사 부하 경감, 데이터 무결성 체크, 스키마 로드맵

---

## 🚀 로컬 실행 방법

Python 3 환경에서 별도의 패키지 설치나 빌드 과정 없이 즉시 실행할 수 있습니다.

```bash
# 프로젝트 폴더 이동
cd flower_fruit_prediction

# 로컬 웹서버 실행
python server.py
```

브라우저에서 **`http://localhost:8000`** 으로 접속합니다.

---

## 📁 프로젝트 구조

```text
flower_fruit_prediction/
├── index.html                               # 메인 대시보드 마크업
├── server.py                                # 로컬 정적 웹서버 (Python 3)
├── test_verification.py                     # 엔드포인트 및 집계 무결성 검증 스크립트
├── tomato_observed_counts_by_crop_cycle.csv # 토마토 실제 관측 데이터 (2,239행)
├── strawberry_observed_counts_by_crop_cycle.csv # 딸기 실제 관측 데이터 (5,218행)
├── src/
│   ├── app.js                               # 메인 컨트롤러 및 이벤트 바인딩
│   ├── parser.js                            # UTF-8 BOM 대응 CSV 파서 및 중복 검증
│   ├── analytics.js                         # 요약 통계, 평균 집계, 개체별 추출 엔진
│   ├── charts.js                            # ECharts 8종 렌더링 및 툴팁 포맷터
│   └── styles.css                           # 프리미엄 테마 및 반응형 그리드 스타일
└── README.md
```

---

## 🔒 배포 가이드 및 데이터 보안 유의사항

1. **정적 호스팅 배포 (Static Web Hosting)**
   - 본 프로젝트는 순수 HTML5/CSS3/ES Modules 기반이므로 Nginx, Apache, AWS S3/CloudFront, Cloudflare Pages, Vercel 등에 폴더 전체를 그대로 배포할 수 있습니다.
2. **데이터 보안 유의사항**
   - 본 CSV 파일은 실제 농업 시설 및 개체의 상세 관측 데이터입니다.
   - 공개 웹서버에 배포 시 누구나 브라우저 개발자 도구나 URL 직접 접근을 통해 원본 CSV를 다운로드할 수 있습니다.
   - **사내 인트라넷, VPN 내부망, 또는 접근 제어(Basic Auth / OAuth)가 적용된 환경**에 배포하는 것을 권장합니다.
