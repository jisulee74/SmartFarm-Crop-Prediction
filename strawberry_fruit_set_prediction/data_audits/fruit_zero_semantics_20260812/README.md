# Fruit-zero semantics analysis

Read-only diagnostic analysis of whether `first_fruits_num = second_fruits_num =
third_fruits_num = 0` behaves like a genuine no-fruit-set observation or a data-entry
default.

This directory is intentionally independent from `offline_dataset_preparation`. It does
not run a dataset split and does not modify the source database or pipeline code.

Run from the GEAS3.5 repository root:

```bash
PYTHONPATH=/tmp/codex_pymysql \
FARMSTOM_DB_PASSWORD='...' \
python3 analyses/fruit_zero_semantics_20260812/analyze.py
```

Connection defaults may be overridden with `FARMSTOM_DB_HOST`, `FARMSTOM_DB_PORT`,
`FARMSTOM_DB_USER`, and `FARMSTOM_DB_NAME`. Outputs are written under `artifacts/`.

