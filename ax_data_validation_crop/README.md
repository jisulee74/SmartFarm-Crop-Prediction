# 작물별 평균 성능 대시보드

새 공개 경로: https://jisulee74.github.io/SmartFarm-Crop-Prediction/ax_data_validation_crop/

기존 대시보드는 `/ax_data_validation/`에 유지한다. 이 새 화면은 기존 공개 CSS·앱·대용량 데이터 파일을 상대 base 경로로 재사용하고, 요약 HTML·CSS·JS·지표 출처는 별도로 관리한다. 기존 경로가 함께 존재해야 한다.

기본 요약은 Validation으로 선정된 각 화방 최우수 모델의 Test bounded 3시드 지표 평균을 읽어, 1·2·3화방 지표를 동일 가중치 산술평균한다. 합계 모델은 포함하지 않는다. R²/CCC의 평균은 전체 관측을 합쳐 재계산한 R²/CCC가 아니며 화면에도 명시했다. 서로 다른 화방의 모델·입력 조합이 다를 수 있다.

`python ax_data_validation_crop/build.py`로 동결된 선별 CSV에서 요약을 재생성한다. 캠페인을 변경하면 선정 모델과 지표 출처를 함께 검토해야 한다. `summary.json`은 원본 CSV 해시 및 8개 타깃 지표를 담는다. 상세 탐색의 모델 선택은 기본 요약을 바꾸지 않는다.

기본 화면에서 예측 대상 선택을 제거하고 화방별 표는 접힌 상세 영역으로 이동했다. 원래 모델·변수군·예측 시계열 탐색은 별도 고급 상세 영역에 보존했다.
