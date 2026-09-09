# 🌿 작물 생육 관측 추이 대시보드 (Agricultural Growth Observation Trends)

토마토 꽃 수와 딸기 착과수의 실제 관측 시계열 데이터를 직관적으로 탐색하고 분석할 수 있는 반응형 웹 대시보드입니다.

---

## 📌 주요 기능 및 특징

1. **8개 독립 시계열 시각화 (Apache ECharts 기반)**
   - **토마토 (4종)**: 1화방 꽃 수, 2화방 꽃 수, 3화방 꽃 수, 전체 꽃 수
   - **딸기 (4종)**: 1화방 착과수, 2화방 착과수, 3화방 착과수, 전체 착과수
   - 실제 날짜 간격이 정확히 반영되는 **시계열 시간축(Time Axis)** 적용
   - 관측점(Point)과 추이선(Line) 동시 표시 (임의 보간/평활화 없음)
   - 마우스 휠 확대/축소 및 하단 **DataZoom 슬라이더** 지원

2. **2가지 분석 모드**
   - **개체당 평균 모드 (기본)**: 해당 조사일에 관측된 유효 개체들의 평균값 및 유효 관측 개체 수($N$)를 툴팁에 상세 표시
   - **개체별 추이 모드**: 특정 개체(시설 → 작기 → 개체번호)를 선택하여 실제 관측값을 왜곡 없이 표시 (시설/작기가 다르면 선이 연결되지 않음)

3. **엄격한 데이터 무결성 처리**
   - `split`(train/validation/test) 구분 없이 전체 실제 관측 데이터 포함
   - **결측치(빈칸)와 0의 엄격한 구분**: 빈칸은 결측으로 처리되어 표본($N$)에서 제외되며, 관측값 `0`은 정상 포함
   - **토마토 전체 꽃 수 보존**: 1·2·3화방 외 화방을 포함한 원본 CSV `tomato_flower_total` 값을 그대로 사용 (임의 재계산 방지)

4. **다차원 연계 필터링**
   - 시설(Facility) → 작기(Crop SN) → 개체 번호(Sample #) 연계 필터
   - 조회 기간(시작일 ~ 종료일) 지정 및 원클릭 초기화 지원

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
