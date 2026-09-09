# All-zero fruit-count semantics analysis

## Scope and provenance

- Source tables: `sfkr_pvsn_grow`, `sfkr_pvsn_crop` (read-only `SELECT` queries).
- Date filter: `meas_date >= 2024-05-02`.
- Main analysis requires all three target columns to be non-NULL.
- `fruits_num` is excluded.
- `fruit_count_target = first_fruits_num + second_fruits_num + third_fruits_num`.
- No dataset split was performed.
- Crop assignment uses matching `user_id`, `item_code`, and inclusive crop start/end dates.

## Duplicate and crop-assignment audit

- Identical duplicate keys retained once: **0**.
- Conflicting duplicate keys: **9**.
- Facility-date groups excluded because at least one conflicting duplicate existed: **1**.
- Rows excluded by that date-level rule: **18**.
- Ambiguous crop assignments excluded from crop-cycle analyses: **840** rows.
- Unmatched crop assignments excluded from crop-cycle analyses: **0** rows.

See `duplicate_audit.csv`, `excluded_conflict_date_rows.csv`,
`ambiguous_crop_assignments.csv`, and `unmatched_crop_assignments.csv`.

## Main results

- Analyzed rows after duplicate/date and crop-assignment rules: **5477**.
- All-zero rows: **3266 (59.63%)**.
- Positive-target rows: **2211**.
- Unique user-survey dates: **648**.
- All samples simultaneously zero on a facility-survey date: **295** dates.

### User and crop-cycle all-zero rates

| user_id    |   cropping_serl_no | cropping_season_name        | cropping_date       | cropping_end_date   |   rows |   all_zero_rows |   all_zero_rate |   target_mean |   target_median |   target_max |
|:-----------|-------------------:|:----------------------------|:--------------------|:--------------------|-------:|----------------:|----------------:|--------------:|----------------:|-------------:|
| PF_0000574 |               7579 | 2024년_딸기(금실)           | 2024-09-01 00:00:00 | 2025-06-30 00:00:00 |    297 |             208 |          0.7003 |        1.2155 |               0 |           12 |
| PF_0000778 |               7578 | 2024년_딸기(금실)           | 2024-09-15 00:00:00 | 2025-06-30 00:00:00 |    288 |             179 |          0.6215 |        1.5243 |               0 |           14 |
| PF_0002037 |               7532 | 2024년_딸기(홍희)           | 2024-09-20 00:00:00 | 2025-05-31 00:00:00 |    261 |             163 |          0.6245 |        2.1111 |               0 |           12 |
| PF_0002037 |               9035 | 2025년_딸기(설향)           | 2025-09-15 00:00:00 | 2026-05-31 00:00:00 |     64 |              63 |          0.9844 |        0.0312 |               0 |            2 |
| PF_0021351 |               7540 | 2024년_딸기(설향)           | 2024-09-07 00:00:00 | 2025-05-25 00:00:00 |    247 |             117 |          0.4737 |        3.6842 |               1 |           26 |
| PF_0021351 |               9028 | 2025년_딸기(설향)           | 2025-09-10 00:00:00 | 2026-05-31 00:00:00 |     92 |              55 |          0.5978 |        2.2826 |               0 |           16 |
| PF_0022105 |               7566 | 2024년_딸기(설향)           | 2024-09-01 00:00:00 | 2025-05-31 00:00:00 |    215 |              78 |          0.3628 |        6.6372 |               7 |           27 |
| PF_0022105 |               9021 | 2025년_딸기(설향)           | 2025-09-22 00:00:00 | 2026-05-31 00:00:00 |    103 |              60 |          0.5825 |        3.1748 |               0 |           17 |
| PF_0023621 |               7565 | 2024년_딸기(설향)           | 2024-09-15 00:00:00 | 2025-05-03 00:00:00 |    205 |             149 |          0.7268 |        1.7415 |               0 |           16 |
| PF_0023732 |               7564 | 2024년_딸기(킹스베리,홍희)  | 2024-08-31 00:00:00 | 2025-06-15 00:00:00 |    283 |             149 |          0.5265 |        3.4629 |               0 |           23 |
| PF_0023732 |               9011 | 2025년_딸기(킹스베리¸ 홍희) | 2025-08-30 00:00:00 | 2026-06-10 00:00:00 |     90 |              76 |          0.8444 |        0.5556 |               0 |            7 |
| PF_0023743 |               7559 | 2024년_딸기(홍희)           | 2024-09-10 00:00:00 | 2025-05-31 00:00:00 |    278 |             124 |          0.446  |        3.4856 |               1 |           20 |
| PF_0023743 |               9008 | 2025년_딸기(홍희)           | 2025-09-07 00:00:00 | 2026-05-31 00:00:00 |    102 |              56 |          0.549  |        3.0392 |               0 |           23 |
| PF_0023923 |               7562 | 2024년_딸기(킹스베리)       | 2024-08-31 00:00:00 | 2025-06-15 00:00:00 |    227 |             132 |          0.5815 |        2.2687 |               0 |           11 |
| PF_0023923 |               9036 | 2025년_딸기(킹스베리)       | 2025-09-15 00:00:00 | 2026-06-15 00:00:00 |     45 |              42 |          0.9333 |        0.0889 |               0 |            2 |
| PF_0024357 |               7560 | 2024년_딸기(설향)           | 2024-09-09 00:00:00 | 2025-05-31 00:00:00 |    259 |              89 |          0.3436 |        5.1042 |               4 |           19 |
| PF_0024357 |               9037 | 2025년_딸기(킹스베리)       | 2025-09-17 00:00:00 | 2026-06-15 00:00:00 |     45 |              45 |          1      |        0      |               0 |            0 |
| PF_0024365 |               7561 | 2024년_딸기(킹스베리)       | 2024-09-10 00:00:00 | 2025-05-20 00:00:00 |    256 |             139 |          0.543  |        2.4961 |               0 |           13 |
| PF_0024651 |               7538 | 2024년_딸기(홍희)           | 2024-09-15 00:00:00 | 2025-06-10 00:00:00 |    288 |             167 |          0.5799 |        1.8681 |               0 |           12 |
| PF_0024654 |               7527 | 2024년_딸기(금실)           | 2024-09-15 00:00:00 | 2025-06-10 00:00:00 |    288 |             191 |          0.6632 |        1.6424 |               0 |           10 |
| PF_0024655 |               7533 | 2024년_딸기(홍희)           | 2024-09-15 00:00:00 | 2025-06-30 00:00:00 |    288 |             173 |          0.6007 |        1.4583 |               0 |            9 |
| PF_0024657 |               7543 | 2024년_딸기(설향)           | 2024-09-15 00:00:00 | 2025-06-30 00:00:00 |      9 |               9 |          1      |        0      |               0 |            0 |
| PF_0024688 |               7558 | 2024년_딸기(설향)           | 2024-09-01 00:00:00 | 2025-06-30 00:00:00 |    277 |             157 |          0.5668 |        2.9134 |               0 |           30 |
| PF_0024688 |               9042 | 2025년_딸기(설향)           | 2025-09-14 00:00:00 | 2026-05-31 00:00:00 |     79 |              77 |          0.9747 |        0.0633 |               0 |            3 |
| PF_0024700 |               7546 | 2024년_딸기(설향)           | 2024-09-10 00:00:00 | 2025-05-31 00:00:00 |    243 |             153 |          0.6296 |        3.5309 |               0 |           42 |
| PF_0024806 |               9046 | 2025년_딸기(킴스베리)       | 2025-09-08 00:00:00 | 2026-05-30 00:00:00 |     63 |              50 |          0.7937 |        0.746  |               0 |           11 |
| PF_0025103 |               7521 | 2024년_딸기(금실)           | 2024-09-15 00:00:00 | 2025-06-10 00:00:00 |    288 |             192 |          0.6667 |        1.3299 |               0 |           10 |
| PF_0025106 |               7518 | 2024년_딸기(설향)           | 2024-09-17 00:00:00 | 2025-06-30 00:00:00 |    279 |             155 |          0.5556 |        2.4158 |               0 |           17 |
| PF_0026933 |              10659 | 2025년_딸기                 | 2025-08-28 00:00:00 | 2026-06-10 00:00:00 |     18 |              18 |          1      |        0      |               0 |            0 |

### Days since crop start

| elapsed_30d_bin   |   rows |   all_zero_rows |   target_mean |   target_median |   target_q25 |   target_q75 |   users |   cycles |   all_zero_rate |
|:------------------|-------:|----------------:|--------------:|----------------:|-------------:|-------------:|--------:|---------:|----------------:|
| [0, 30)           |    225 |             225 |        0      |               0 |            0 |            0 |      13 |       13 |          1      |
| [30, 60)          |    702 |             642 |        0.2251 |               0 |            0 |            0 |      18 |       18 |          0.9145 |
| [60, 90)          |    655 |             421 |        2.5649 |               0 |            0 |            6 |      18 |       19 |          0.6427 |
| [90, 120)         |    684 |             236 |        4.3173 |               4 |            0 |            7 |      18 |       20 |          0.345  |
| [120, 150)        |    587 |              83 |        5.4514 |               5 |            2 |            8 |      18 |       20 |          0.1414 |
| [150, 180)        |    632 |             102 |        5.4604 |               4 |            2 |            7 |      18 |       21 |          0.1614 |
| [180, 210)        |    786 |             425 |        2.2748 |               0 |            0 |            3 |      19 |       26 |          0.5407 |
| [210, 240)        |    785 |             720 |        0.4    |               0 |            0 |            0 |      18 |       26 |          0.9172 |
| [240, 270)        |    360 |             351 |        0.0944 |               0 |            0 |            0 |      16 |       19 |          0.975  |
| [270, 300)        |     61 |              61 |        0      |               0 |            0 |            0 |       3 |        4 |          1      |

### Early, middle, and late crop phases

The phases are relative thirds of each crop's recorded duration, not biological stage labels.

| crop_phase   |   rows |   all_zero_rows |   target_mean |   target_median |   target_q25 |   target_q75 |   all_zero_rate |
|:-------------|-------:|----------------:|--------------:|----------------:|-------------:|-------------:|----------------:|
| early        |   1635 |            1316 |        1.2232 |               0 |            0 |            0 |          0.8049 |
| middle       |   1901 |             384 |        5.0994 |               4 |            1 |            7 |          0.202  |
| late         |   1941 |            1566 |        0.9706 |               0 |            0 |            0 |          0.8068 |

- Early all-zero rate: **80.49%**
- Middle all-zero rate: **20.20%**
- Late all-zero rate: **80.68%**

### Within-sample temporal behavior

These indicators can overlap: a sequence may contain both transitions and a positive-zero-positive subsequence.

- `0 → positive`: **162** sample-cycle sequences.
- `positive → 0`: **188** sample-cycle sequences.
- `positive → 0 → positive`: **62** sample-cycle sequences.
- Zero throughout the observed crop sequence: **58** sample-cycle sequences.

### Other growth measurements

`positive_rate` means a parsed numeric value greater than zero. This is a recording-completeness
diagnostic, not a biological normal-range test.

| ('column', '')   |   ('non_null_rate', 'all_zero') |   ('non_null_rate', 'positive') |   ('positive_rate', 'all_zero') |   ('positive_rate', 'positive') |   ('median_non_null', 'all_zero') |   ('median_non_null', 'positive') |
|:-----------------|--------------------------------:|--------------------------------:|--------------------------------:|--------------------------------:|----------------------------------:|----------------------------------:|
| flower_length    |                          0      |                               0 |                          0      |                          0      |                             nan   |                             nan   |
| grow_length      |                          0      |                               0 |                          0      |                          0      |                             nan   |                             nan   |
| leaves_length    |                          0.9991 |                               1 |                          0.9862 |                          1      |                             105   |                             109   |
| leaves_num       |                          0.9991 |                               1 |                          0.9871 |                          0.9986 |                               8   |                               9   |
| leaves_width     |                          0      |                               0 |                          0      |                          0      |                             nan   |                             nan   |
| petiole_length   |                          0      |                               0 |                          0      |                          0      |                             nan   |                             nan   |
| stem_diameter    |                          0      |                               0 |                          0      |                          0      |                             nan   |                             nan   |
| theca_diameter   |                          0.9991 |                               1 |                          0.9859 |                          1      |                              19.1 |                              22.3 |

## Interpretation

The statistical evidence is mixed and cannot prove data-entry intent:

1. Early-season concentration followed by `0 → positive` supports genuine pre-fruit-set zeros.
2. `positive → 0` can be biologically plausible after harvest/removal, but abrupt
   `positive → 0 → positive` patterns may also reflect inconsistent recording or changing
   interpretation of the three truss fields.
3. Facility-date-wide zeros are more suspicious than isolated plant zeros, especially when
   other growth fields remain populated, but synchronized biological state is still possible.
4. Rows that stay zero throughout an observed crop can be genuine non-setting plants, incomplete
   crop coverage, or a default-value convention. The database alone cannot distinguish them.

## Recommendation before dataset split

- **Do not globally convert all zeros to missing and do not globally discard all zeros.**
- Retain zeros provisionally as valid targets when they occur early in a crop and are followed by
  positive observations for the same sample, or when surrounding measurements support a coherent
  biological trajectory.
- Add flags rather than silently deleting questionable rows:
  `all_zero_target`, `all_samples_zero_on_date`, `positive_zero_positive`,
  `zero_throughout_observed_cycle`, `days_since_crop_start`, and `crop_phase`.
- Exclude every facility-date containing a conflicting duplicate, as done here, until an
  authoritative conflict-resolution rule exists.
- Treat mid/late all-zero rows, facility-wide all-zero dates, and positive-zero-positive sequences
  as sensitivity-analysis candidates. Train/evaluate once with them retained and once excluded.
- Confirm with the data owner whether zero is the UI/database default for unmeasured trusses and
  whether counts reset after harvest. This domain confirmation remains necessary before freezing
  the target contract.

## Figures

- `figures/elapsed_zero_rate_and_target.png`
- `figures/crop_phase_zero_rate.png`
- `figures/crop_phase_target_distribution.png`
- `figures/user_cycle_zero_rate.png`
- `figures/temporal_sequence_patterns.png`
