# 토마토 화방별 꽃수 합계 예측

`sfkr_hbfm_grow`의 1~34화방 꽃수를 조사일별로 합산하고, 동일 시설·작기의
다음 조사 합계를 예측하는 독립 파이프라인이다. 데이터 분할 단위는 시설 ID이며
Test 시설은 Validation으로 구성을 확정할 때까지 읽지 않는다.

## 실행

```bash
# DB에서 원본 캐시를 생성한 뒤 전체 실험 실행
export FARMSTOM_DB_PASSWORD='...'
envs/jslee_py311/bin/python3.11 AXData/tomato_flower_count_prediction/run_pipeline.py --rebuild-cache

# 이미 생성된 raw cache를 재사용
envs/jslee_py311/bin/python3.11 AXData/tomato_flower_count_prediction/run_pipeline.py
```

DB 접속정보는 `FARMSTOM_DB_HOST`, `FARMSTOM_DB_PORT`, `FARMSTOM_DB_USER`,
`FARMSTOM_DB_NAME`, `FARMSTOM_DB_PASSWORD` 환경변수로만 받는다. 기본 실행은
`data/raw/growth.parquet`, `data/raw/environment.parquet`을 읽는다.
CatBoost, PyTorch, TabM, PyMySQL은 저장소의 `envs/jslee_py311` 환경에 설치되어 있다.

## 입력 계약

- growth: `source_sn, facility_id, user_id, crop_sn, cropping_serl_no, item_code,
  cropping_date, cropping_end_date, examin_date, growth_measure_code, flower_count`
- environment: `source_sn, facility_id, item_code, fld_code, sect_code, fatr_code,
  maker_id, sen_id, sen_val, meas_date`

`growth_measure_code=10000234`는 1화방이며 `10000267`까지 사용한다. 조사일의
1화방부터 최대 관측 화방까지 중간 번호가 빠진 행과 충돌 중복은 제외한다.

