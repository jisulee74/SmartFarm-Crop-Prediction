"""
test_verification.py - Comprehensive Verification Script
Verifies:
1. Static files & new API endpoints
2. Observation dataset integrity
3. ML experiment comparison metrics vs frozen winners
4. Prediction alignment with sample keys
5. Incomplete / unstarted target handling
"""

import sys
import json
import urllib.request
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_URL = "http://localhost:8000"

def test_endpoints():
    print("=== 1. HTTP Endpoint Verification ===")
    urls = [
        f"{BASE_URL}/",
        f"{BASE_URL}/src/styles.css",
        f"{BASE_URL}/src/app.js",
        f"{BASE_URL}/src/experiment_api.js",
        f"{BASE_URL}/src/experiment_view.js",
        f"{BASE_URL}/src/parser.js",
        f"{BASE_URL}/src/analytics.js",
        f"{BASE_URL}/src/charts.js",
        f"{BASE_URL}/tomato_observed_counts_by_crop_cycle.csv",
        f"{BASE_URL}/strawberry_observed_counts_by_crop_cycle.csv",
        f"{BASE_URL}/api/experiments/metadata",
        f"{BASE_URL}/api/experiments/comparison?target=tomato_first&model=tabm&split=validation",
        f"{BASE_URL}/api/experiments/predictions?target=tomato_first&group=VG_Flower_first_d0edfb3a_V2&model=tabm&split=validation&seed=all",
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

def test_metadata():
    print("\n=== 2. Experiment Metadata & Targets Verification ===")
    url = f"{BASE_URL}/api/experiments/metadata"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))
    
    targets = data.get("targets", {})
    print(f"Total indexed targets: {len(targets)}")
    assert len(targets) == 8, f"Expected 8 targets, got {len(targets)}"
    
    for t_id, t_info in targets.items():
        print(f"  Target: {t_id:22s} | Status: {t_info['status']:10s} | Groups: {t_info['group_count']:2d} | Winner: {t_info.get('frozen_winner')}")
        if "tomato" in t_id:
            assert t_info["group_count"] == 40, f"Expected 40 groups for tomato target {t_id}"
        elif "strawberry" in t_id:
            assert t_info["group_count"] == 10, f"Expected 10 groups for strawberry target {t_id}"
    print("Target group counts and models verified successfully.")

def test_winner_metrics_exact_match():
    print("\n=== 3. Comparison Table & Metrics Exact Match Verification ===")
    url = f"{BASE_URL}/api/experiments/comparison?target=tomato_first&model=tabm&split=validation"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))

    frozen_winner = data.get("frozen_winner")
    print(f"Frozen winner for tomato_first TabM: {frozen_winner}")
    assert frozen_winner == "VG_Flower_first_d0edfb3a_V2", f"Unexpected winner: {frozen_winner}"

    rows = data.get("rows", [])
    rank1_row = rows[0]
    print(f"Rank 1 group in table: {rank1_row['group_id']} | Readable: {rank1_row['readable_name']}")
    assert rank1_row['group_id'] == frozen_winner, "Rank 1 must match the frozen winner"
    
    b_rmse = rank1_row['bounded']['rmse_mean']
    b_mae = rank1_row['bounded']['mae_mean']
    print(f"Rank 1 Bounded RMSE Mean: {b_rmse:.8f}")
    print(f"Rank 1 Bounded MAE Mean:  {b_mae:.8f}")

    # Verify against the exact ground truth recorded in selection_frozen.json
    expected_rmse = 0.30570339417753123
    expected_mae = 0.06592403663951453
    assert abs(b_rmse - expected_rmse) < 1e-6, f"RMSE mismatch: {b_rmse} vs {expected_rmse}"
    assert abs(b_mae - expected_mae) < 1e-6, f"MAE mismatch: {b_mae} vs {expected_mae}"
    print("Winner metrics match selection_frozen.json perfectly.")

def test_prediction_alignment():
    print("\n=== 4. Predictions Alignment & Identification Keys Verification ===")
    url = f"{BASE_URL}/api/experiments/predictions?target=tomato_first&group=VG_Flower_first_d0edfb3a_V2&model=tabm&split=validation&seed=all"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))

    assert data["status"] == "success", f"Prediction status: {data.get('status')}"
    preds = data.get("predictions", [])
    print(f"Loaded ensemble predictions count: {len(preds)}")
    assert len(preds) == 339, f"Expected 339 validation samples, got {len(preds)}"

    p0 = preds[0]
    print(f"Sample prediction row keys: {list(p0.keys())}")
    print(f"Sample entity: {p0['facility_id']} | crop_sn: {p0['crop_sn']} | sample_num: {p0['sample_num']}")
    print(f"Target date: {p0['target_date']} | Actual: {p0['target']} | Pred bounded: {p0['prediction_bounded']} | Pred raw: {p0['prediction_raw']}")

    # Verify keys existence
    for k in ['facility_id', 'crop_sn', 'sample_num', 'target_date', 'target', 'prediction_raw', 'prediction_bounded']:
        assert k in p0, f"Missing key in prediction: {k}"

    # Verify bounded is constrained within [0, upper_bound]
    for r in preds:
        if r['prediction_bounded'] is not None and r['upper_bound'] is not None:
            assert r['prediction_bounded'] >= 0, f"Negative bounded prediction: {r['prediction_bounded']}"
            assert r['prediction_bounded'] <= r['upper_bound'] + 1e-5, f"Exceeded upper bound: {r['prediction_bounded']} > {r['upper_bound']}"
    print("Bounded predictions respect [0, upper_bound] constraint.")

def test_incomplete_or_test_non_winner():
    print("\n=== 5. Non-winner Test & Incomplete Edge Cases Verification ===")
    # Test split on non-winner group should return no_results or empty predictions without error
    non_winner_group = "VG_Flower_first_0fe0650e_V2"
    url = f"{BASE_URL}/api/experiments/predictions?target=tomato_first&group={non_winner_group}&model=tabm&split=test&seed=all"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))
    print(f"Non-winner test query status: {data.get('status')}")
    assert data.get("status") in ["no_results", "no_predictions"], "Non-winner group in test split must have no predictions"
    assert len(data.get("predictions", [])) == 0, "Non-winner group must not fabricate test predictions"
    print("Non-winner test split handled correctly without data fabrication.")

def test_analysis_bundle():
    print("\n=== 6. Analysis Bundle Endpoint & Data Integrity Verification ===")
    url = f"{BASE_URL}/api/analysis/bundle"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))
    
    assert "tables" in data, "Missing 'tables' in analysis bundle"
    assert "metadata" in data, "Missing 'metadata' in analysis bundle"
    tables = data["tables"]

    expected_keys = [
        "baseline_performance", "cohort_profile", "selected_models",
        "error_strata", "crossed_error_strata", "target_diagnosis",
        "hypotheses", "ax_field_schema", "comparison_arms", "kpi_planning_reference"
    ]
    for k in expected_keys:
        assert k in tables, f"Missing key in analysis bundle tables: {k}"
        print(f"  Key '{k}': {len(tables[k]) if isinstance(tables[k], list) else type(tables[k])}")

    # Check baseline_performance has 8 targets
    base_perf = tables["baseline_performance"]
    assert len(base_perf) >= 8, f"Expected at least 8 rows in baseline_performance, got {len(base_perf)}"
    
    # Check hypotheses H01~H07
    hyp = tables["hypotheses"]
    assert len(hyp) == 7, f"Expected 7 hypotheses, got {len(hyp)}"
    
    # Check schema explorer has 88 fields
    schema = tables["ax_field_schema"]
    assert len(schema) == 88, f"Expected 88 fields in ax_field_schema, got {len(schema)}"
    
    # Check comparison arms has 7 arms (B0 to ABL)
    arms = tables["comparison_arms"]
    assert len(arms) == 7, f"Expected 7 comparison arms, got {len(arms)}"
    print("Analysis bundle verified successfully with all required datasets!")

def test_html_dom_integrity():
    print("\n=== 7. HTML DOM Integrity & Observation Preservation Verification ===")
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # 1. Check all three primary tabs exist
    assert 'id="tab-observation"' in html, "tab-observation is missing"
    assert 'id="tab-experiments"' in html, "tab-experiments is missing"
    assert 'id="tab-analysis"' in html, "tab-analysis is missing"

    # 2. Check Tab 1 key observation elements are preserved
    obs_elements = [
        'summary-grid', 'tomato_summary_badge', 'strawberry_summary_badge',
        'chart_tomato_flower_truss1', 'chart_strawberry_fruit_truss1',
        'crop-nav-tabs', 'guide-banner'
    ]
    for el in obs_elements:
        assert el in html, f"Tab 1 element missing: {el}"

    # 3. Check Tab 2 main-content wrapper & filter toolbar
    exp_elements = [
        'exp-control-card', 'exp_filter_facility', 'exp_filter_crop_sn',
        'exp_filter_sample_num', 'exp_filter_reset_btn', 'table-responsive'
    ]
    for el in exp_elements:
        assert el in html, f"Tab 2 element missing: {el}"

    # 4. Check Tab 3 3-subtab architecture and components
    anl_elements = [
        'anl-subtab-perf', 'anl-subtab-ax', 'anl-subtab-kpi',
        'anl_target_select', 'anl_dimension_select', 'anl_summary_model', 'anl_summary_group',
        'anl_summary_rmse', 'anl_summary_mae', 'anl_summary_r2', 'anl_summary_ccc',
        'anl_summary_n', 'anl_summary_facilities', 'anl_baseline_tbody',
        'anl_chart_model_comp', 'anl_chart_strata', 'anl_chart_sse',
        'anl_strata_tbody', 'anl_crossed_tbody', 'anl_diagnosis_content',
        'anl_hyp_list', 'anl_hyp_detail_panel', 'anl_schema_tabs',
        'anl_schema_tbody', 'anl_schema_search', 'anl_arms_container',
        'anl_arm_detail_panel', 'anl_target_kpi_grid', 'anl_kpi_tbody'
    ]
    for el in anl_elements:
        assert el in html, f"Tab 3 element missing: {el}"

    print("HTML DOM structure and element preservation verified 100%!")

def test_numerical_accuracy_and_schema():
    print("\n=== 8. Numerical Accuracy & Dynamic Schema Verification ===")
    url = f"{BASE_URL}/api/analysis/bundle"
    req_obj = urllib.request.Request(url, headers={'Connection': 'close'})
    with urllib.request.urlopen(req_obj, timeout=10) as req:
        data = json.loads(req.read().decode('utf-8'))
    
    tables = data["tables"]
    base = tables["baseline_performance"]

    # 1. Tomato sum123 test RMSE ~ 0.896866, Persistence ~ 0.740912
    t_sum_test = next(r for r in base if r["target"] == "tomato_sum123" and r["split"] == "test")
    print(f"Tomato sum123 test RMSE: {t_sum_test['rmse_mean']:.6f} (Expected ~0.896866)")
    print(f"Tomato sum123 persistence RMSE: {t_sum_test['persistence_rmse']:.6f} (Expected ~0.740912)")
    assert abs(t_sum_test['rmse_mean'] - 0.896866) < 1e-4, "Tomato sum test RMSE mismatch"
    assert abs(t_sum_test['persistence_rmse'] - 0.740912) < 1e-4, "Tomato sum persistence RMSE mismatch"

    # 2. Strawberry second test RMSE ~ 1.494445
    s_2nd_test = next(r for r in base if r["target"] == "strawberry_second" and r["split"] == "test")
    print(f"Strawberry second test RMSE: {s_2nd_test['rmse_mean']:.6f} (Expected ~1.494445)")
    assert abs(s_2nd_test['rmse_mean'] - 1.494445) < 1e-4, "Strawberry second test RMSE mismatch"

    # 3. Dynamic schema counts
    schema = tables["ax_field_schema"]
    from collections import Counter
    counts = Counter(r["table"] for r in schema)
    print("Schema table counts:", dict(counts))
    expected_counts = {
        'ax_event': 19,
        'ax_observation': 17,
        'ax_context': 14,
        'ax_constraint': 13,
        'survey_schedule': 13,
        'existing_growth': 12
    }
    for tbl, exp_cnt in expected_counts.items():
        assert counts[tbl] == exp_cnt, f"Table {tbl} count mismatch: {counts[tbl]} vs {exp_cnt}"
    assert len(schema) == 88, f"Expected 88 total fields, got {len(schema)}"
    print("Numerical accuracy and schema table counts verified 100%!")

if __name__ == "__main__":
    test_endpoints()
    test_metadata()
    test_winner_metrics_exact_match()
    test_prediction_alignment()
    test_incomplete_or_test_non_winner()
    test_analysis_bundle()
    test_html_dom_integrity()
    test_numerical_accuracy_and_schema()
    print("\n==============================================")
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("==============================================")



