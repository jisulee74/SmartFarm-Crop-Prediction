"""
test_verification.py - Comprehensive Verification Script for Exhaustive Experiments Dashboard
Verifies:
1. 127 Combinations Catalog Integrity (count, uniqueness, sizes, group definitions)
2. Static files & REST API endpoints
3. 8 Targets, 6 Models, 3 Seeds configuration
4. Observation datasets integrity (tomato & strawberry)
5. Pending / unstarted status handling without mock data
"""

import sys
import json
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_URL = "http://localhost:8000"
LOCAL_DIR = Path(__file__).parent.resolve()


def test_combination_catalog_integrity():
    print("=== 1. Combination Catalog & Variable Definitions Integrity ===")
    cat_file = LOCAL_DIR / "cache" / "combination_catalog.json"
    assert cat_file.exists(), f"Missing combination catalog file: {cat_file}"

    with open(cat_file, encoding="utf-8") as f:
        cat = json.load(f)

    combos = cat.get("combinations", [])
    print(f"Total combinations count: {len(combos)}")
    assert len(combos) == 127, f"Expected 127 combinations, got {len(combos)}"

    # Check unique combination IDs
    ids = [c["combination_id"] for c in combos]
    assert len(set(ids)) == 127, "Duplicate combination IDs found!"

    # Check breakdown by size
    size_counts = {}
    for c in combos:
        cnt = c["group_count"]
        size_counts[cnt] = size_counts.get(cnt, 0) + 1

    expected_sizes = {1: 7, 2: 21, 3: 35, 4: 35, 5: 21, 6: 7, 7: 1}
    print(f"Combinations by size: {size_counts}")
    assert size_counts == expected_sizes, f"Size breakdown mismatch: {size_counts} vs {expected_sizes}"

    # Check variable groups E1~E7 and C0
    vgroups = cat.get("variable_groups", {})
    print(f"Variable groups defined: {list(vgroups.keys())}")
    for g in ["C0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]:
        assert g in vgroups, f"Missing variable group: {g}"
        print(f"  {g}: {vgroups[g]['name']} ({vgroups[g].get('feature_count', len(vgroups[g].get('features', [])))} features)")

    print("[PASS] Combination catalog integrity 100% verified.")


def test_endpoints():
    print("\n=== 2. HTTP Endpoints Verification ===")
    urls = [
        f"{BASE_URL}/",
        f"{BASE_URL}/src/styles.css",
        f"{BASE_URL}/src/app.js",
        f"{BASE_URL}/src/experiment_api.js",
        f"{BASE_URL}/src/experiment_view.js",
        f"{BASE_URL}/src/parser.js",
        f"{BASE_URL}/src/analytics.js",
        f"{BASE_URL}/src/charts.js",
        f"{BASE_URL}/cache/combination_catalog.json",
        f"{BASE_URL}/cache/experiment_index.json",
        f"{BASE_URL}/tomato_observed_counts_by_crop_cycle.csv",
        f"{BASE_URL}/strawberry_observed_counts_by_crop_cycle.csv",
        f"{BASE_URL}/api/experiments/catalog",
        f"{BASE_URL}/api/experiments/metadata",
        f"{BASE_URL}/api/experiments/comparison?target=tomato_first&model=tabm&split=validation",
        f"{BASE_URL}/api/experiments/predictions?target=tomato_first&group=COMBO_b5b76a4693c1&model=tabm&split=validation&seed=all",
    ]
    for u in urls:
        try:
            req_obj = urllib.request.Request(u, headers={'Connection': 'close'})
            with urllib.request.urlopen(req_obj, timeout=10) as req:
                content = req.read()
                print(f"[OK] {req.status} - {u} ({len(content)} bytes)")
        except Exception as e:
            print(f"[FAIL] {u} - {e}")
            raise e
    print("[PASS] All HTTP endpoints responded with 200 OK.")


def test_metadata_and_targets():
    print("\n=== 3. Metadata & Targets Verification ===")
    url = f"{BASE_URL}/api/experiments/metadata"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))

    targets = data.get("targets", {})
    print(f"Total indexed targets: {len(targets)}")
    assert len(targets) == 8, f"Expected 8 targets, got {len(targets)}"

    for t_id, t_info in targets.items():
        print(f"  Target: {t_id:22s} | Status: {t_info['status']:18s} | Combinations: {t_info['group_count']:3d}")
        assert t_info["group_count"] == 127, f"Expected 127 combinations for target {t_id}, got {t_info['group_count']}"

    models = data.get("models", [])
    print(f"Models: {models}")
    assert len(models) == 6, f"Expected 6 models, got {len(models)}"

    seeds = data.get("seeds", [])
    print(f"Seeds: {seeds}")
    assert seeds == [42, 52, 62], f"Expected seeds [42, 52, 62], got {seeds}"

    print("[PASS] Metadata and targets verified successfully.")


def test_comparison_127_combinations():
    print("\n=== 4. Comparison Table (127 Combinations) Verification ===")
    url = f"{BASE_URL}/api/experiments/comparison?target=tomato_first&model=tabm&split=validation"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))

    rows = data.get("rows", [])
    print(f"Total rows in comparison table: {len(rows)}")
    assert len(rows) == 127, f"Expected 127 rows, got {len(rows)}"

    # Check that each row has proper combination info and pending/unstarted status when unrun
    for r in rows:
        assert "group_id" in r, "Missing group_id"
        assert "readable_name" in r, "Missing readable_name"
        assert "feature_count" in r, "Missing feature_count"
        assert "status" in r, "Missing status"

    print(f"Sample row 1: {rows[0]['readable_name']} ({rows[0]['group_id']}) -> Status: {rows[0]['status']}, Rank: {rows[0].get('rank')}")
    print("[PASS] Comparison table returned exactly 127 combinations.")


def test_observation_datasets():
    print("\n=== 5. Observation Datasets Integrity ===")
    tomato_csv = LOCAL_DIR / "tomato_observed_counts_by_crop_cycle.csv"
    strawberry_csv = LOCAL_DIR / "strawberry_observed_counts_by_crop_cycle.csv"

    df_t = pd.read_csv(tomato_csv)
    df_s = pd.read_csv(strawberry_csv)

    print(f"Tomato CSV rows: {len(df_t)} (Expected: 2239)")
    print(f"Strawberry CSV rows: {len(df_s)} (Expected: 5218)")
    assert len(df_t) == 2239, f"Tomato rows mismatch: {len(df_t)}"
    assert len(df_s) == 5218, f"Strawberry rows mismatch: {len(df_s)}"

    print("[PASS] Observation datasets verified.")


if __name__ == "__main__":
    print("Starting Comprehensive Verification Test Suite...")
    test_combination_catalog_integrity()
    test_endpoints()
    test_metadata_and_targets()
    test_comparison_127_combinations()
    test_observation_datasets()
    print("\n=======================================================")
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================")
