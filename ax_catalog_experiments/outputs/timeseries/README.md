# 날짜·분할별 실제 관측 합계

- observed_counts_8_by_timestamp_split.csv: timestamp, split, 8개 관측값 열.
- timestamp는 최신 ax_catalog_experiments 실험의 target_date(다음 조사일)이며, feature_date나 예측값이 아닙니다.
- 원래 시설 기준 train/validation/test 배정을 그대로 사용합니다.
- 각 값은 그 날짜·split에서 실제 평가 대상으로 사용된 관측 개체의 합입니다. 시설 전체 생산량이나 모든 작물 개체의 총량은 아닙니다.
- tomato_flower_truss1/2/3: 개체별 해당 화방의 유효 원천 꽃 수를 합산.
- tomato_flower_total: 기존 전체 꽃 수 타깃(전체 유효 관측 화방 합)의 개체 간 합. 1~3화방만의 합이 아닙니다.
- strawberry_fruit_truss1/2/3: 해당 화방의 실제 착과수 합. strawberry_fruit_total은 세 화방의 합.
- 조사 없음/화방 미관측은 빈칸이며 실제 0과 구별합니다. 일부 개체만 해당 화방이 관측된 경우 합계는 유효 관측분의 합입니다.
- 날짜마다 관측 개체 수가 다릅니다. 합계 증가를 개체당 생육 증가로 해석하지 마세요. observation_counts.csv에 값별 유효 관측 개체 수를 함께 제공합니다.
- 토마토·딸기는 서로 다른 시설·개체입니다. 같은 행은 날짜와 split만 공유합니다.
- 학습/검증/테스트 어느 쪽에도 없는 원천 행은 포함하지 않습니다.

재생성: `envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments.export_timeseries`

작물별 파일: tomato_observed_counts_by_timestamp_split.csv, strawberry_observed_counts_by_timestamp_split.csv. 각 파일은 timestamp, split, 해당 작물의 4개 값으로 구성되며 해당 작물 관측이 전혀 없는 행은 제외합니다.

作期日期 / 작기 날짜: 원천 DB 추출본의 cropping_date, cropping_end_date를 사용합니다. 실시간 DB 재조회가 아니며 DB 종료일이 실제 종료 확정일인지 예정일인지는 별도 확인이 필요합니다. 단일 작기 행은 crop_start_date/crop_end_date에 날짜를 표시하고, 여러 작기가 섞인 행은 두 열을 빈칸으로 둡니다. crop_cycle_count와 earliest_crop_start_date/latest_crop_end_date는 집계 범위입니다. 작기별 정확한 날짜는 *_observed_counts_by_crop_cycle.csv에서 시설·작기별로 확인하세요.

현재 작기 연결: 딸기는 sfkr_pvsn_crop 추출본의 user_id + item_code(080400) + 조사일 포함으로 재매칭하며 1개만 날짜를 확정합니다. 0개/복수 매칭은 날짜 빈칸과 unmatched/ambiguous로 보존합니다. matched_crop_sn은 작기 테이블 PK이고 기존 crop_sn은 실험 추적용입니다. 토마토는 원천 cropping_number 미보존 및 DB 연결 부재로 직접 연결 미완료이며 기존 날짜를 보존하고 direct_match_pending_missing_cropping_number로 표시합니다. cropping_serl_no를 cropping_number로 복제하지 않습니다.

업데이트: 토마토 날짜는 source_snapshot/tomato_direct_crop_dates.parquet의 실시간 DB 직접 매칭 결과를 사용합니다. 위의 토마토 미연결 설명은 직접 매칭 캐시가 없는 경우에만 적용됩니다. crop_sn은 기존 실험 추적용이며 matched_crop_sn은 직접 연결된 sfkr_pvsn_crop.sn입니다.

개체별 상세 파일: *_observed_counts_by_crop_cycle.csv는 timestamp + split + facility_id + crop_sn + sample_num별 한 행입니다. 개체 간 합산하지 않습니다. 각 값은 해당 개체의 실제 관측값이며 전체 값만 해당 개체의 화방 간 합계입니다. timestamp_split 파일은 기존 날짜·분할 합계입니다. 상세 파일에서는 cropping_number, crop_match_method, crop_match_status, crop_match_count 열을 내보내지 않습니다.
