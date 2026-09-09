# Growth/environment joinability audit

Read-only AXData audit performed before dataset splitting. The analysis applies the
confirmed fruit target policy, does not interpolate or create model features, and does
not modify source tables or existing pipeline code.

Run from `GEAS/AXData`:

```bash
PYTHONPATH=/tmp/codex_pymysql \
FARMSTOM_DB_PASSWORD='...' \
python3 analyses/growth_environment_joinability_20260812/analyze.py
```

