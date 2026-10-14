# 시설별 세분 제어코드 실험 기술 보관본

이 폴더는 대시보드에서 제외한 `방향·층수 세분 코드` 기준 결과의 재현·감사용 기술 자료입니다.

- 원 데이터의 시설별 제어 코드를 그대로 구분한 기준 실험입니다.
- 방향·층수 조합이 시설마다 달라 E6 후보가 지나치게 희소해졌고, 원 코드 E6 후보 408개가 변수 선택 단계에서 모두 제외되는 문제가 확인됐습니다.
- 사용자용 대시보드는 시설 간 비교가 가능한 `장치 의미 통합` 결과만 제공합니다.
- 원 코드와 원 실험 결과는 삭제하거나 덮어쓰지 않았습니다. 대용량 run/checkpoint 파일은 서버의 고정 캠페인 출력에 보존합니다.

## 파일

- `original_campaign_manifest.json`: 최초 전수실험의 고정 설정과 해시
- `e1_e2_backfill_campaign_manifest.json`: E1·E2 보완 실험 설정
- `validation_results_compact.csv`: 8개 타깃·6개 모델·127개 조합의 세분 코드 Validation 요약
- `combination_catalog.csv`: 조합 ID와 변수군 구성
- `data_quality.csv`: 원 데이터 품질 요약
- `checksums.json`: 이 보관본의 SHA-256
