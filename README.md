# SmartFarm Crop Prediction

최신 소스 정리 기준: 2026-10-29 (GPU MIG 설정 변경 전).

AX 데이터 예측·검증 코드와 기존 공개 대시보드를 관리합니다.

- [공개 대시보드](https://jisulee74.github.io/SmartFarm-Crop-Prediction/)
- [작물별 요약](https://jisulee74.github.io/SmartFarm-Crop-Prediction/ax_data_validation_crop/)

## 구성

| 경로 | 내용 |
|---|---|
| `AXData/common_regression/` | 공통 전처리·모델·평가·검증 |
| `AXData/ax_catalog_experiments/` | 범주 조합·E6 의미 통합·상위 화방 실험 |
| `AXData/*_prediction/` | 작물·타깃별 예측 파이프라인 |
| `flower_fruit_prediction/` | 관측·예측 대시보드 |
| `flower_fruit_prediction_exhaustive/` | 전수실험 대시보드 |
| `ax_data_validation/` | AX 데이터 검증 대시보드 |
| `ax_data_validation_crop/` | 작물별 요약 대시보드 |

설치와 테스트는 [AXData 안내](AXData/README.md), 백업 복구는
[복구 안내](docs/RESTORE.md)를 참고하세요.

## 소스와 데이터 관리

원시 DB 추출물, 중간 데이터, 학습 결과·가중치·로그는 Git에서 제외하고
영구 작업공간의 별도 백업에 보존합니다. 과거 결과는 Git 이력에도 남습니다.
대시보드가 사용하는 `cache/`, 관측 CSV와 요약 JSON은 웹 표시용 배포 자산이므로 유지합니다.
표시된 결과를 갱신하려면 해당 실험의 검증·게시 절차를 사용하세요.

```bash
python flower_fruit_prediction/server.py
```

`main`에 push하면 기존 GitHub Pages workflow가 실행됩니다.

## 실험 재현

환경·입력·선정 근거와 재현 명령은 [재현 안내](reproducibility/README.md)를 확인하세요.
`python scripts/reproduce.py smoke`는 실제 데이터 없이 합성 실험을 검증합니다.
