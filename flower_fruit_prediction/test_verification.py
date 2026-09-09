import urllib.request
import pandas as pd
import numpy as np

# 1. Test HTTP server endpoints
urls = [
    'http://localhost:8000/',
    'http://localhost:8000/src/styles.css',
    'http://localhost:8000/src/app.js',
    'http://localhost:8000/src/parser.js',
    'http://localhost:8000/src/analytics.js',
    'http://localhost:8000/src/charts.js',
    'http://localhost:8000/tomato_observed_counts_by_crop_cycle.csv',
    'http://localhost:8000/strawberry_observed_counts_by_crop_cycle.csv'
]

print("=== 1. HTTP Endpoint Verification ===")
for u in urls:
    try:
        req_obj = urllib.request.Request(u, headers={'Connection': 'close'})
        with urllib.request.urlopen(req_obj, timeout=5) as req:
            content = req.read()
            print(f"[OK] {req.status} - {u} ({len(content)} bytes)")
    except Exception as e:
        print(f"[FAIL] {u} - {e}")

# 2. Aggregation & Edge Case Verification
print("\n=== 2. Aggregation & Edge Case Verification ===")
for name, crop in [('tomato_observed_counts_by_crop_cycle.csv', 'tomato'), ('strawberry_observed_counts_by_crop_cycle.csv', 'strawberry')]:
    df = pd.read_csv(name)
    print(f"\n*** Crop: {crop} ({name}) ***")
    print(f"Total rows: {len(df)}")
    print(f"Unique facilities: {df['facility_id'].nunique()}")
    print(f"Unique crop_sn: {df['crop_sn'].nunique()}")
    df['entity_id'] = df['facility_id'].astype(str) + " | " + df['crop_sn'].astype(str) + " | #" + df['sample_num'].astype(str)
    print(f"Unique entities: {df['entity_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # Check sample date average computation
    sample_date = df['timestamp'].iloc[0]
    sub = df[df['timestamp'] == sample_date]
    print(f"Sample date {sample_date} rows: {len(sub)}")
    for col in df.columns:
        if 'truss' in col or 'total' in col:
            valid_vals = sub[col].dropna()
            print(f"  {col}: count={len(valid_vals)}, mean={valid_vals.mean():.2f}, min={valid_vals.min()}, max={valid_vals.max()}")
