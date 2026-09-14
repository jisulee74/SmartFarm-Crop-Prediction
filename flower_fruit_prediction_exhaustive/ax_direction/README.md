# 생육 예측 성능 원인 분석 및 AX 추가 데이터 구축 방향

분석 기준: 2026-09-14. 실험 완료: 2026-09-12 23:46 KST. 원본: `../../outputs/stability_v2/`.

## 먼저 열 파일

- **종합분석보고서.html**: 세 과제 본문, 그래프 3종, 타깃·구간별 분석표 필터를 포함한 오프라인 보고서. 브라우저에서 파일을 직접 열 수 있다. 그래프와 필터 데이터는 내장되어 있다.
- **분석표_및_AX설계.xlsx**: 분석 수치·원인 가설·AX 필드 스키마·KPI 기획표. 상세 예측 행은 별도 parquet 제공.
- **ANTIGRAVITY_PROMPT.md**: 웹사이트 반영용 전달 프롬프트.

## 과제별 산출물

| 요청 과제 | 문서 | 주요 근거 표 |
|---|---|---|
| 1. 8개 타깃 성능·오차·원인 구분 | 01_성능_오차_원인분석.md | baseline_performance, cohort_profile, model_comparison, feature_coverage, target_diagnosis |
| 2. 구간 분석·추가 수집 항목·연결 방법 | 01 및 02_AX_추가데이터_구축방향.md | error_strata, crossed_error_strata, largest_error_cases, hypotheses, ax_field_schema |
| 3. As-Is/To-Be·KPI 설계 | 03_AsIs_ToBe_검증체계_KPI.md | comparison_arms, kpi_planning_reference |

모든 CSV는 `tables/`, 그래프 PNG/PDF는 `figures/`, 회의록 추출 본문은 `sources/`에 있다. 원본 PDF는 세 단계 상위 AXData 폴더에 있다. 회의록 0909의 본문 일자는 9월 10일이다.

## 수치와 해석

- 원본 39,856회 완료. 496개 모델별 선정/유지 기준 예측 파일에서 3,968개 지표를 재계산했다. 상세 범위는 audit.json.
- 8개 타깃 각각 validation으로 선정된 모델을 고정했다. test 결과를 보고 다시 선정하지 않았다.
- 결과는 원칙적으로 bounded 예측의 seed별 지표 평균. seed SD는 초기화 변동이며 모집단 신뢰구간이 아니다.
- N은 고유 평가 행 수. 반복 seed나 8개 타깃을 독립 표본처럼 합산하지 않는다.
- 수량 변화·정답 양수 여부·다음 실제 조사 간격은 사후 진단. 해당 정보를 원래 예측 시점 입력으로 쓰면 안 된다.
- 토마토의 0 집중·초기/시설 편차, 딸기의 변화 구간 오차는 확인된 현상이다. 적과·수확 등 실제 원인과 AX 추가 입력의 효과는 아직 미검증이다.
- 10%/15% 개선 수치 및 수집 품질 KPI는 협의 전 제안. 현재 달성 실적이나 확정 계약 기준이 아니다.
- 현재 test는 이미 확인된 holdout이므로 추가 정보의 개선 효과 확정에는 새 미래 코호트가 필요하다.

## 재현

작업 공간 루트에서:

```bash
envs/jslee_py311/bin/python AXData/ax_catalog_experiments/analysis/20260914_ax_direction/analyze.py
envs/jslee_py311/bin/python AXData/ax_catalog_experiments/analysis/20260914_ax_direction/build_report.py
envs/jslee_py311/bin/python AXData/ax_catalog_experiments/analysis/20260914_ax_direction/package_outputs.py
```

스크립트는 원본을 읽고 이 분석 폴더에만 기록한다. 추가 모델 학습, 실험 재실행, 기존 웹사이트 수정은 수행하지 않는다. 새 원본 실험에는 이 버전을 덮어쓰지 말고 분석 버전을 새로 생성한다.

## 웹 연동

`dashboard_bundle.json`은 주요 표의 JSON 묶음이다. `data_dictionary.json`은 각 파일의 컬럼·데이터 타입과 주요 집계 규칙을 제공한다. `selected_predictions.parquet`에서 `target_id`는 과제 이름, `target`은 실제 개수다. 상세 계약은 ANTIGRAVITY_PROMPT.md 참조.

## 검증 및 파일 이력

- audit.json / tables/integrity_checks.csv: 원본 및 지표 정합성
- source_hashes.json: 읽은 원본 파일 및 분석 스크립트 해시
- deliverable_checks.json: 보고서·Excel·JSON·링크·웹 필터 점검 결과
- output_manifest.json: 전달 파일별 크기와 SHA-256

`AX_분석산출물_20260914.zip`은 원본 학습 모델을 제외한 전달용 묶음이다. HTML 단독으로 본문/그래프/필터를 볼 수 있으며, Excel 링크와 CSV 참조까지 이용하려면 압축을 풀고 같은 디렉터리 구조를 유지한다.
