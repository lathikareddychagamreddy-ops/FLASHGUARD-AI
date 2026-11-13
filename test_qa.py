"""
FLASHGUARD AI — PRODUCTION SYSTEM QA & VALIDATION SUITE
Problem Statement 13: Hybrid Prediction & Classification for Flash Crash Risk Management
ASTRAQUANT X-2026 • KLH University Aziz Nagar Campus

Automated End-to-End Verification across all Competition Requirements:
1. Environment & ML Dependencies (XGBoost, Scikit-Learn, FastAPI, Joblib)
2. Canonical 10-Feature Pipeline & Microstructure Math
3. Liquidity Withdrawal Rate Proxy (Order-Book Depth Decreases)
4. Persisted Canonical AI Models (Multi-class Classifier, Downside Regressor, Crash Classifier)
5. Genuine Flash-Crash Prediction Pipeline & Distinctions (Feature 1A)
6. Out-of-Sample Model Performance Metrics & Confusion Matrices (Feature 1B)
7. Live Market Price Rescaling (Deribit BTC-PERPETUAL USD) & Time-Series (Feature 1C & Issue 3)
8. Real-Time Alert History Ledger & CSV Export (Feature 1D)
9. Statistical Data Drift Detection & Statuses (Feature 1E)
10. Historical Backtesting Scorecard & Advance Lead Time (Feature 1F)
11. Feature Importance Distribution & Ranking (Feature 1G)
12. Backend REST API Endpoints & Mode Management (LIVE, REPLAY, SIMULATION)
13. Non-Contradictory Multi-Factor Early Warning System
14. Frontend Static Assets (HTML5, CSS3, JS) & Reactive DOM Bindings
15. AI Model Testing Lab — Scenario Execution & Expected vs Predicted Matrix (Tasks 2, 3, 4, 5)
16. AI Model Testing Lab — Statistical Error Analysis & OOD Feature Diagnostics (Task 6)
17. AI Model Testing Lab — Historical Ground-Truth & Zero Target Leakage Verification (Task 7)
18. AI Model Testing Lab — Test Report CSV Export Endpoint (Task 8 & 10)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

def run_qa_suite():
    print("=" * 80)
    print("FLASHGUARD AI — COMPREHENSIVE QA & PRODUCTION VALIDATION SUITE")
    print("ASTRAQUANT X-2026: Problem Statement 13")
    print("=" * 80)

    passed_tests = 0
    total_tests = 18

    # -------------------------------------------------------------------------
    # TEST 1: Environment & Dependency Validation
    # -------------------------------------------------------------------------
    print("\n[QA TEST 1/18] Validating Python Runtime & Quantitative ML Libraries...")
    try:
        import xgboost as xgb
        import sklearn
        import fastapi
        import uvicorn
        import joblib
        print(f"  ✓ Python Runtime:       {sys.version.split()[0]}")
        print(f"  ✓ XGBoost Version:      {xgb.__version__}")
        print(f"  ✓ Scikit-Learn Version: {sklearn.__version__}")
        print(f"  ✓ FastAPI Version:      {fastapi.__version__}")
        print(f"  ✓ Joblib Version:       {joblib.__version__}")
        passed_tests += 1
        print("  [PASS] Test 1: Python quantitative stack verified.")
    except Exception as e:
        print(f"  [FAIL] Test 1: Dependency error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 2: Canonical Feature Pipeline & Microstructure Extraction
    # -------------------------------------------------------------------------
    print("\n[QA TEST 2/18] Validating Canonical 10-Feature Pipeline & Extraction...")
    try:
        from feature_pipeline import (
            CANONICAL_FEATURES,
            TARGET_REGIME_NAMES,
            compute_features_from_snapshot,
            RollingMicrostructureBuffer
        )
        assert len(CANONICAL_FEATURES) == 10, f"Expected 10 canonical features, got {len(CANONICAL_FEATURES)}"
        expected_features = [
            "bid_ask_spread", "relative_spread_bps", "spread_expansion_pct",
            "book_depth_top5", "volume_imbalance", "liquidity_withdrawal_rate",
            "realized_volatility_10m", "trade_intensity_10m", "cancel_pressure",
            "volatility_spread_interaction"
        ]
        for ef in expected_features:
            assert ef in CANONICAL_FEATURES, f"Missing feature: {ef}"

        bids = [[68500.0, 15.0], [68499.0, 25.0], [68498.0, 40.0], [68497.0, 60.0], [68496.0, 100.0]]
        asks = [[68501.0, 12.0], [68502.0, 20.0], [68503.0, 35.0], [68504.0, 50.0], [68505.0, 90.0]]
        feat_dict, feat_vec, meta = compute_features_from_snapshot(
            best_bid=68500.0,
            best_ask=68501.0,
            top_bids=bids,
            top_asks=asks
        )
        assert len(feat_vec) == 10, f"Expected 10-dimensional vector, got {len(feat_vec)}"
        assert feat_dict["bid_ask_spread"] == 1.0
        assert feat_dict["book_depth_top5"] == 447.0
        assert abs(feat_dict["volume_imbalance"]) <= 1.0

        print(f"  ✓ Canonical Feature Count: 10/10 Aligned")
        print(f"  ✓ Sample Spread:           ${feat_dict['bid_ask_spread']:.2f}")
        print(f"  ✓ Sample Top-5 Depth:      {feat_dict['book_depth_top5']} contracts")
        passed_tests += 1
        print("  [PASS] Test 2: Feature pipeline calculates exact canonical features.")
    except Exception as e:
        print(f"  [FAIL] Test 2: Feature pipeline error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 3: Liquidity Withdrawal Rate Proxy Verification (Issue 2 & Task 11)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 3/18] Testing Liquidity Withdrawal Rate Proxy Formula & Clean Initialization...")
    try:
        from feature_pipeline import RollingMicrostructureBuffer
        buf = RollingMicrostructureBuffer(maxlen=30)

        # Baseline resilient book
        for i in range(10):
            buf.add_tick(mid_price=68500.0, spread=0.5, total_depth=25000.0, timestamp_ms=1000 + i*100)
        exp1, r1, vol1 = buf.calculate_rolling_metrics(0.5, 25000.0)

        # Severe depth withdrawal
        for i in range(5):
            buf.add_tick(mid_price=68490.0, spread=2.5, total_depth=25000.0 - (i+1)*4000.0, timestamp_ms=2000 + i*100)
        exp2, r2, vol2 = buf.calculate_rolling_metrics(2.5, 3000.0)

        assert r2 > r1, f"Expected higher withdrawal rate on depth collapse: {r2} vs {r1}"
        assert 0.0 <= r1 <= 0.20, f"Resilient withdrawal rate should be low (<0.20): {r1}"
        assert 0.0 <= r2 <= 1.0, f"Withdrawal rate out of range [0, 1]: {r2}"
        print(f"  ✓ Resilient Depth Withdrawal Rate: {r1:.3f}")
        print(f"  ✓ Stressed Depth Withdrawal Rate:  {r2:.3f}")
        passed_tests += 1
        print("  [PASS] Test 3: Liquidity Withdrawal Rate reliably measures depth drain and avoids 100% false start.")
    except Exception as e:
        print(f"  [FAIL] Test 3: Withdrawal rate error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 4: Persisted Canonical AI Models & Serialization
    # -------------------------------------------------------------------------
    print("\n[QA TEST 4/18] Verifying Persisted Canonical Machine Learning Models...")
    try:
        models_dir = "models"
        required_model_files = [
            "xgb_liquidity_classifier.joblib",
            "rf_liquidity_classifier.joblib",
            "lr_liquidity_classifier.joblib",
            "xgb_downside_regressor.joblib",
            "xgb_crash_classifier.joblib",
            "scaler.joblib",
            "model_metadata.json"
        ]
        for mf in required_model_files:
            p = os.path.join(models_dir, mf)
            assert os.path.exists(p), f"Missing model artifact: {p}"
            sz = os.path.getsize(p)
            print(f"  ✓ Loaded {mf} ({sz:,} bytes)")

        passed_tests += 1
        print("  [PASS] Test 4: All 5 canonical models and scaler artifacts verified.")
    except Exception as e:
        print(f"  [FAIL] Test 4: Model file error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 5: Genuine Flash-Crash Prediction Pipeline (Feature 1A)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 5/18] Testing Genuine Flash-Crash Model vs 60s Downside Risk (Feature 1A)...")
    try:
        from server import MODELS, run_model_inference, load_canonical_models
        load_canonical_models()

        # Normal test vector
        normal_vec = np.array([0.5, 0.07, 0.0, 20000.0, 0.02, 0.05, 0.007, 1800, 0.01, 0.0035], dtype=np.float32)
        normal_res = run_model_inference(normal_vec, {})
        p_norm = normal_res["primary_prediction"]

        # Extreme crash vector (Spread surge > 500%, withdrawal rate 0.90, depth 400)
        crash_vec = np.array([12.0, 1.75, 550.0, 400.0, -0.85, 0.92, 0.095, 250, 0.88, 1.14], dtype=np.float32)
        crash_res = run_model_inference(crash_vec, {})
        p_crash = crash_res["primary_prediction"]

        print(f"  ✓ Normal Input -> Downside Risk: {p_norm['downside_risk_prob_60s']*100:.1f}%, Genuine Crash Prob: {p_norm['genuine_flash_crash_prob']*100:.1f}%")
        print(f"  ✓ Black Hole Input -> Downside Risk: {p_crash['downside_risk_prob_60s']*100:.1f}%, Genuine Crash Prob: {p_crash['genuine_flash_crash_prob']*100:.1f}%")

        assert p_crash["genuine_flash_crash_prob"] > p_norm["genuine_flash_crash_prob"]
        assert "60-Second Downside-Move Risk" in p_norm["downside_target_definition"]
        assert "Genuine Flash-Crash Probability" in p_norm["genuine_crash_target_definition"]

        # Probabilities sum to 1.0
        probs = list(p_norm["probabilities"].values())
        assert abs(sum(probs) - 1.0) < 0.001, f"Class probabilities must sum to 100%: {sum(probs)}"

        passed_tests += 1
        print("  [PASS] Test 5: Genuine flash-crash probability is distinct and validated.")
    except Exception as e:
        print(f"  [FAIL] Test 5: Crash prediction error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 6: Out-of-Sample Performance Dashboard Metrics (Feature 1B)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 6/18] Testing Out-of-Sample Evaluation Metrics & Matrices (Feature 1B)...")
    try:
        from server import app
        client = TestClient(app)
        res = client.get("/api/evaluation/metrics")
        assert res.status_code == 200
        metrics = res.json()

        assert "liquidity_classification" in metrics
        assert "downside_risk_prediction" in metrics
        assert "genuine_flash_crash" in metrics

        crash_m = metrics["genuine_flash_crash"]
        print(f"  ✓ Flash-Crash PR-AUC:             {crash_m['pr_auc']:.4f}")
        print(f"  ✓ Flash-Crash ROC-AUC:            {crash_m['roc_auc']:.4f}")
        print(f"  ✓ Positive Test Events:           {crash_m['positive_test_events']} events")
        print(f"  ✓ Negative Test Events:           {crash_m['negative_test_events']} events")
        print(f"  ✓ False Alarm Rate:               {crash_m['false_alarm_rate']:.2f}%")
        print(f"  ✓ Brier Score Calibration:        {crash_m['brier_score']:.4f}")

        assert crash_m["positive_test_events"] > 0, "Must have real positive test events"
        assert crash_m["pr_auc"] > 0.50, f"PR-AUC {crash_m['pr_auc']} below 0.50"
        assert crash_m["roc_auc"] > 0.80, f"ROC-AUC {crash_m['roc_auc']} below 0.80"

        passed_tests += 1
        print("  [PASS] Test 6: Model performance metrics calculated on real out-of-sample data.")
    except Exception as e:
        print(f"  [FAIL] Test 6: Performance metrics error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 7: Rescaled Historical Replay Price ($68k USD) & Timeseries (Feature 1C & Issue 3)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 7/18] Verifying Historical Replay Price Scale & Timeseries (Feature 1C)...")
    try:
        from deribit_feed import feed_manager, FeedMode
        feed_manager.set_mode(FeedMode.HISTORICAL_REPLAY)
        replay_snap = feed_manager.get_current_snapshot()
        mid = replay_snap["raw_market"]["mid_price"]
        unit = replay_snap["raw_market"]["price_unit"]
        hist_ts = replay_snap["raw_market"]["historical_timestamp"]

        print(f"  ✓ Historical Replay Mid-Price:    ${mid:,.2f}")
        print(f"  ✓ Historical Price Unit:         {unit}")
        print(f"  ✓ Historical Timestamp:          {hist_ts}")

        assert mid > 50000.0, f"Historical price appears unscaled: ${mid}"
        assert unit == "USD", f"Expected USD unit, got {unit}"
        assert "2026-10-01" in hist_ts, f"Expected realistic timestamp, got {hist_ts}"

        client = TestClient(app)
        res_ts = client.get("/api/dataset/timeseries?limit=10")
        assert res_ts.status_code == 200
        ts_data = res_ts.json()
        assert len(ts_data["mid_price"]) > 0
        assert ts_data["mid_price"][0] > 50000.0

        passed_tests += 1
        print("  [PASS] Test 7: Historical replay price correctly scaled to Deribit BTC-PERPETUAL USD.")
    except Exception as e:
        print(f"  [FAIL] Test 7: Price rescaling error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 8: Alert History Ledger & CSV Export (Feature 1D)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 8/18] Testing Alert History Logging & CSV Export (Feature 1D)...")
    try:
        client = TestClient(app)
        res_state = client.get("/api/state")
        assert res_state.status_code == 200

        res_hist = client.get("/api/alerts/history?limit=10")
        assert res_hist.status_code == 200
        alerts = res_hist.json().get("alerts", [])
        print(f"  ✓ Total Recorded Alerts in Ledger: {len(alerts)}")

        res_csv = client.get("/api/alerts/export")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers.get("content-type", "")
        csv_text = res_csv.text
        assert "Timestamp" in csv_text
        assert "Warning Level" in csv_text
        assert "Liquidity Regime" in csv_text
        assert "Downside Risk" in csv_text
        assert "Flash Crash" in csv_text

        print("  ✓ Alert History CSV Export verified with full header metadata.")
        passed_tests += 1
        print("  [PASS] Test 8: Real-time alert history and CSV export operate cleanly.")
    except Exception as e:
        print(f"  [FAIL] Test 8: Alert history error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 9: Statistical Data Drift Detection (Feature 1E)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 9/18] Testing Statistical Data Drift Detection (Feature 1E)...")
    try:
        client = TestClient(app)
        res_drift = client.get("/api/drift")
        assert res_drift.status_code == 200
        drift = res_drift.json()

        assert "overall_status" in drift
        assert "metrics" in drift
        m = drift["metrics"]
        assert "bid_ask_spread" in m
        assert "book_depth_top5" in m
        assert "volume_imbalance" in m
        assert "liquidity_withdrawal_rate" in m

        print(f"  ✓ Drift Overall Status:    {drift['overall_status']}")
        print(f"  ✓ Spread Drift Z-Score:    {m['bid_ask_spread']['z_score']:.2f}σ ({m['bid_ask_spread']['status']})")
        print(f"  ✓ Depth Drift Z-Score:     {m['book_depth_top5']['z_score']:.2f}σ ({m['book_depth_top5']['status']})")
        print(f"  ✓ Drift Interpretation:    {drift['interpretation']}")

        passed_tests += 1
        print("  [PASS] Test 9: Data Drift Monitor tracks microstructure distributions.")
    except Exception as e:
        print(f"  [FAIL] Test 9: Data drift error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 10: Historical Backtesting Scorecard & Lead Time (Feature 1F)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 10/18] Testing Historical Backtesting Scorecard & Lead Time (Feature 1F)...")
    try:
        client = TestClient(app)
        res_bt = client.get("/api/models/backtest")
        assert res_bt.status_code == 200
        bt = res_bt.json()

        print(f"  ✓ Out-of-Sample Test Windows: {bt['test_windows_count']}")
        print(f"  ✓ Correct Predictions:        {bt['correct_predictions']} ({bt['accuracy_pct']}%)")
        print(f"  ✓ Missed Events (False Neg):  {bt['false_negatives_missed_events']}")
        print(f"  ✓ False Alarms (False Pos):   {bt['false_positives_false_alarms']}")
        print(f"  ✓ Advance Warning Lead Time:  {bt['prediction_lead_time']}")

        assert bt["test_windows_count"] == 445
        assert bt["correct_predictions"] > 400
        assert "60.0 seconds" in bt["prediction_lead_time"]

        passed_tests += 1
        print("  [PASS] Test 10: Historical backtest scorecard verified.")
    except Exception as e:
        print(f"  [FAIL] Test 10: Backtest report error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 11: Feature Importance Ranking & Interpretation (Feature 1G & Task 9)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 11/18] Testing XGBoost Feature Importance Importances (Feature 1G)...")
    try:
        client = TestClient(app)
        res_fi = client.get("/api/models/feature_importance")
        assert res_fi.status_code == 200
        fi = res_fi.json()

        assert "liquidity_withdrawal_rate" in fi
        assert "trade_intensity_10m" in fi
        assert "spread_expansion_pct" in fi
        assert "book_depth_top5" in fi

        print(f"  ✓ Liquidity Withdrawal Rate: {fi['liquidity_withdrawal_rate']*100:.1f}%")
        print(f"  ✓ Trade Intensity (10m):     {fi['trade_intensity_10m']*100:.1f}%")
        print(f"  ✓ Spread Expansion %:        {fi['spread_expansion_pct']*100:.1f}%")
        print(f"  ✓ Top-5 Book Depth:          {fi['book_depth_top5']*100:.1f}%")

        assert fi["liquidity_withdrawal_rate"] > 0.15
        assert fi["spread_expansion_pct"] > 0.15

        passed_tests += 1
        print("  [PASS] Test 11: Feature importance weights from real model verified.")
    except Exception as e:
        print(f"  [FAIL] Test 11: Feature importance error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 12: Backend Mode Switching & Simulation Presets
    # -------------------------------------------------------------------------
    print("\n[QA TEST 12/18] Testing Operating Modes (LIVE, REPLAY, SIMULATION)...")
    try:
        client = TestClient(app)
        # Switch to SIMULATION
        res_sim = client.post("/api/feed/mode", json={"mode": "SIMULATION"})
        assert res_sim.status_code == 200

        # Inject Severe Black Hole scenario
        res_inject = client.post("/api/feed/simulation/scenario", json={
            "scenario_name": "Severe Liquidity Black Hole",
            "mid_price": 67900.0,
            "bid_ask_spread": 14.50,
            "spread_expansion_pct": 550.0,
            "liquidity_withdrawal_rate": 0.88,
            "volume_imbalance": -0.85,
            "book_depth_top5": 450.0,
            "realized_volatility_10m": 0.095,
            "trade_intensity_10m": 250
        })
        assert res_inject.status_code == 200

        state_sim = client.get("/api/state").json()
        assert state_sim["mode"] == "SIMULATION"
        assert state_sim["is_simulated"] is True
        assert state_sim["ai_prediction"]["primary_prediction"]["liquidity_regime"] in ["Fragmented", "Illiquid"]

        # Switch back to LIVE
        res_live = client.post("/api/feed/mode", json={"mode": "LIVE"})
        assert res_live.status_code == 200
        state_live = client.get("/api/state").json()
        assert state_live["mode"] == "LIVE"
        assert state_live["is_simulated"] is False

        passed_tests += 1
        print("  [PASS] Test 12: Mode isolation and scenario stress test operate seamlessly.")
    except Exception as e:
        print(f"  [FAIL] Test 12: Mode switching error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 13: Non-Contradictory Early Warning System
    # -------------------------------------------------------------------------
    print("\n[QA TEST 13/18] Testing Non-Contradictory Early Warning Rules & Banners...")
    try:
        from risk_engine import EarlyWarningEngine
        # Normal
        norm = EarlyWarningEngine.evaluate(
            features={"bid_ask_spread": 0.50, "spread_expansion_pct": 5.0, "book_depth_top5": 20000.0, "liquidity_withdrawal_rate": 0.05},
            downside_risk_prob=0.10,
            flash_crash_prob=0.01,
            is_simulated=False
        )
        assert norm["warning_level"] == "NORMAL"
        assert "BENIGN" in norm["banner_title"]

        # High Risk
        high = EarlyWarningEngine.evaluate(
            features={"bid_ask_spread": 15.0, "spread_expansion_pct": 600.0, "book_depth_top5": 300.0, "liquidity_withdrawal_rate": 0.90},
            downside_risk_prob=0.85,
            flash_crash_prob=0.75,
            is_simulated=False
        )
        assert high["warning_level"] == "HIGH RISK"
        assert "CRITICAL" in high["banner_title"]

        print(f"  ✓ Normal Banner:   {norm['banner_title']} -> State: {norm['warning_level']}")
        print(f"  ✓ High Risk Banner: {high['banner_title']} -> State: {high['warning_level']}")

        passed_tests += 1
        print("  [PASS] Test 13: Early Warning banners strictly agree with warning state.")
    except Exception as e:
        print(f"  [FAIL] Test 13: Early warning error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 14: Frontend Static Assets & Required DOM Bindings
    # -------------------------------------------------------------------------
    print("\n[QA TEST 14/18] Checking Frontend DOM Elements & Testing Lab Bindings...")
    try:
        html = open(os.path.join("static", "index.html"), "r", encoding="utf-8").read()
        js = open(os.path.join("static", "app.js"), "r", encoding="utf-8").read()
        css = open(os.path.join("static", "style.css"), "r", encoding="utf-8").read()

        required_ids = [
            "globalModePill", "btnModeLive", "btnModeReplay", "btnModeSim",
            "replayControls", "simAlertBanner", "earlyWarningBanner", "ewTitle",
            "heroMidPrice", "heroSpreadVal", "heroWithdrawalVal", "heroHistTime",
            "ladderBids", "ladderAsks", "predRegimeLabel", "crashProbVal",
            "genuineCrashProbVal", "priceHistoryChart", "riskHistoryChart",
            "importanceGrid", "alertCountBadge", "alertTableBody",
            "testingLabPanel", "labComparisonTableBody", "labDiagnosticsBox",
            "trTotalTests", "trPassedTests", "trFailedTests", "trCompat", "trLatency", "trHistAcc",
            "groundTruthSection", "evalXgbLoss", "evalMse", "evalCrashPrAuc"
        ]
        for rid in required_ids:
            assert f'id="{rid}"' in html, f"Missing element #{rid} in index.html"

        required_js = [
            "selectLabScenario", "runTestingLabCurrentScenario", "runTestingLabAllScenarios",
            "runTestingLabHistoricalEval", "fetchFeatureImportances", "pushDataToCharts"
        ]
        for rjs in required_js:
            assert rjs in js, f"Missing JS function {rjs} in app.js"

        passed_tests += 1
        print(f"  ✓ index.html: all {len(required_ids)} required DOM element IDs verified.")
        print(f"  ✓ app.js:     all {len(required_js)} interactive functions present.")
        print("  [PASS] Test 14: Frontend presentation layer fully aligned with specification.")
    except Exception as e:
        print(f"  [FAIL] Test 14: Frontend verification error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 15: AI Model Testing Lab — Scenario Execution & Expected vs Predicted (Tasks 2, 3, 4, 5)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 15/18] Testing AI Model Testing Lab Scenario Inference & Comparison Matrix...")
    try:
        client = TestClient(app)

        # 1. Healthy Liquid Scenario
        res_liq = client.post("/api/testing-lab/run-scenario", json={"scenario_key": "liquid"})
        assert res_liq.status_code == 200
        data_liq = res_liq.json()
        assert data_liq["comparison"]["result"] == "PASS"
        assert "Liquid" in data_liq["comparison"]["ai_prediction"]
        assert data_liq["is_simulated"] is True
        print(f"  ✓ Scenario 1 (Healthy Liquid):      Expected Liquid -> Predicted {data_liq['comparison']['ai_prediction']} [{data_liq['comparison']['result']}]")

        # 2. Liquidity Withdrawal Scenario
        res_frag = client.post("/api/testing-lab/run-scenario", json={"scenario_key": "fragmented"})
        assert res_frag.status_code == 200
        data_frag = res_frag.json()
        assert data_frag["comparison"]["result"] == "PASS"
        assert "Fragmented" in data_frag["comparison"]["ai_prediction"]
        print(f"  ✓ Scenario 2 (Liquidity Stress):    Expected Fragmented -> Predicted {data_frag['comparison']['ai_prediction']} [{data_frag['comparison']['result']}]")

        # 3. Severe Black Hole Scenario
        res_bh = client.post("/api/testing-lab/run-scenario", json={"scenario_key": "blackhole"})
        assert res_bh.status_code == 200
        data_bh = res_bh.json()
        assert data_bh["comparison"]["result"] == "PASS"
        assert "Illiquid" in data_bh["comparison"]["ai_prediction"]
        print(f"  ✓ Scenario 3 (Severe Black Hole):   Expected Illiquid -> Predicted {data_bh['comparison']['ai_prediction']} [{data_bh['comparison']['result']}]")

        # 4. Batch All Scenarios
        res_all = client.post("/api/testing-lab/run-all-scenarios")
        assert res_all.status_code == 200
        data_all = res_all.json()
        assert data_all["total_scenarios_tested"] == 3
        assert data_all["passed_count"] == 3
        assert data_all["failed_count"] == 0
        print(f"  ✓ Batch 3 Scenarios Test:          {data_all['passed_count']}/{data_all['total_scenarios_tested']} Passed (100% Pass Rate)")

        passed_tests += 1
        print("  [PASS] Test 15: AI Model Testing Lab scenario execution & comparison matrix verified.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] Test 15: Scenario testing error: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 16: AI Model Testing Lab — Statistical Error Analysis & OOD Feature Diagnostics (Task 6)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 16/18] Testing Error Analysis & Out-of-Distribution Feature Diagnostics...")
    try:
        client = TestClient(app)

        # Intentionally mismatch expectations: assert severe black hole inputs should be 'Liquid'
        res_mismatch = client.post("/api/testing-lab/run-scenario", json={
            "scenario_name": "Diagnostic Mismatch Test",
            "expected_liquidity_state": "Liquid", # Intentional mismatch
            "mid_price": 67900.0,
            "bid_ask_spread": 18.0,
            "spread_expansion_pct": 650.0,
            "liquidity_withdrawal_rate": 0.95,
            "volume_imbalance": -0.90,
            "book_depth_top5": 300.0,
            "realized_volatility_10m": 0.120,
            "trade_intensity_10m": 150
        })
        assert res_mismatch.status_code == 200
        mismatch_data = res_mismatch.json()

        assert mismatch_data["comparison"]["result"] == "FAIL"
        err = mismatch_data["error_analysis"]
        assert err["has_mismatch"] is True
        assert err["expected_label"] == "Liquid"
        assert len(err["divergent_features"]) > 0
        assert "Root diagnostic" in err["explanation"] or "Classification mismatch" in err["explanation"]

        print(f"  ✓ Honest Mismatch Flagged:        Expected: {err['expected_label']} vs Predicted: {err['predicted_label']} -> Result: FAIL")
        print(f"  ✓ Out-of-Distribution Count:      {err['out_of_distribution_count']} features outside ±2σ")
        print(f"  ✓ Diagnostic Explanation Sample:  {err['explanation'][:100]}...")

        passed_tests += 1
        print("  [PASS] Test 16: Statistical error analysis and OOD diagnostics operate without forced outcomes.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] Test 16: Error analysis failure: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 17: AI Model Testing Lab — Historical Ground-Truth & Zero Target Leakage (Task 7)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 17/18] Testing Historical Ground-Truth Evaluation & Zero Target Leakage...")
    try:
        client = TestClient(app)
        res_hist = client.get("/api/testing-lab/historical-eval")
        assert res_hist.status_code == 200
        hist_data = res_hist.json()

        assert hist_data["test_windows_count"] == 445
        assert hist_data["purge_gap_windows"] == 5

        # Check leakage verification
        leak = hist_data["leakage_verification"]
        assert leak["target_leakage_detected"] is False
        assert len(leak["features_used"]) == 10
        assert "liquidity_regime" not in leak["features_used"]
        assert "flash_crash_prob_60s" not in leak["features_used"]
        assert "is_liquidity_black_hole" not in leak["features_used"]

        # Check classification metrics
        liq = hist_data["liquidity_classification"]
        assert liq["accuracy_pct"] >= 95.0
        assert liq["macro_f1"] >= 0.90
        assert len(liq["confusion_matrix"]) == 3
        assert sum(liq["class_distribution_test"]) == 445

        # Check genuine crash metrics
        crash = hist_data["genuine_flash_crash"]
        assert crash["positive_test_events"] == 48
        assert crash["negative_test_events"] == 397
        assert crash["pr_auc"] >= 0.50
        assert crash["roc_auc"] >= 0.85
        assert crash["false_alarm_rate_pct"] <= 5.0

        print(f"  ✓ Evaluated Unseen Test Windows:   {hist_data['test_windows_count']} (Purge Gap: {hist_data['purge_gap_windows']})")
        print(f"  ✓ Target Leakage Status:          {leak['target_leakage_detected']} (Zero Target Leakage)")
        print(f"  ✓ Ground-Truth Liquidity Acc:     {liq['accuracy_pct']}% (Macro F1: {liq['macro_f1']})")
        print(f"  ✓ Ground-Truth Confusion Matrix:  {liq['confusion_matrix']}")
        print(f"  ✓ Ground-Truth Crash PR-AUC:      {crash['pr_auc']:.4f} (ROC-AUC: {crash['roc_auc']:.4f})")
        print(f"  ✓ Ground-Truth Positive Events:   {crash['positive_test_events']} events")

        passed_tests += 1
        print("  [PASS] Test 17: Ground-truth historical evaluation verified with zero target leakage.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] Test 17: Ground-truth evaluation failure: {e}")
        return False

    # -------------------------------------------------------------------------
    # TEST 18: AI Model Testing Lab — Test Report CSV Export Endpoint (Task 8 & 10)
    # -------------------------------------------------------------------------
    print("\n[QA TEST 18/18] Testing Model Testing Lab Report CSV Export Endpoint...")
    try:
        client = TestClient(app)
        res_exp = client.get("/api/testing-lab/export")
        assert res_exp.status_code == 200
        assert "text/csv" in res_exp.headers.get("content-type", "")
        csv_text = res_exp.text

        assert "FLASHGUARD AI — MODEL TESTING LAB AUDIT REPORT" in csv_text
        assert "SECTION 1: ARTIFICIAL STRESS SCENARIO EVALUATION" in csv_text
        assert "SECTION 2: GENUINE HISTORICAL GROUND-TRUTH EVALUATION" in csv_text
        assert "Healthy Liquid Market" in csv_text
        assert "Severe Liquidity Black Hole" in csv_text
        assert "Liquidity Accuracy" in csv_text
        assert "Flash-Crash PR-AUC" in csv_text

        print("  ✓ Full Testing Lab Audit CSV report generated with all scenario and historical sections.")
        passed_tests += 1
        print("  [PASS] Test 18: Testing Lab CSV report export operational.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] Test 18: Testing lab report export error: {e}")
        return False

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"ALL QA ACCEPTANCE CHECKS PASSED: {passed_tests}/{total_tests} (100% SUCCESS)")
    print("AI Model Testing Lab & Historical Ground-Truth fully verified against competition criteria.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = run_qa_suite()
    sys.exit(0 if success else 1)
