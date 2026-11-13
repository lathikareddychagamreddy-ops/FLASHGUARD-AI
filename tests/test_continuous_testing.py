"""
FlashGuard AI — Continuous AI Testing Automated Pytest Suite
Problem Statement 13: Hybrid Prediction & Classification for Flash Crash Risk Management
ASTRAQUANT X-2026

10 Comprehensive Unit & Integration Tests:
1. Test continuous prediction scheduling (timing & intervals)
2. Test start, pause, resume, and stop controls & state transitions
3. Test model feature compatibility and canonical ordering
4. Test prediction timestamp ordering and sequence monotonically increasing
5. Test 60-second outcome evaluation logic (P_min <= P_0 * 0.999)
6. Test True Positive, False Positive, True Negative, False Negative matrix calculations
7. Test handling of missing future data / data gap marking as INCOMPLETE
8. Test WebSocket disconnect / stale feed handling (inference paused when disconnected)
9. Test duplicate prediction prevention on pause/resume/restart
10. Test exported CSV test report format and contents
"""

import os
import sys
import time
import json
import uuid
import tempfile
import pytest
import numpy as np
import pandas as pd

# Add parent directory to path so server & continuous_testing can be imported cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_pipeline import CANONICAL_FEATURES, TARGET_REGIME_NAMES
from continuous_testing import (
    ContinuousTestingDatabase,
    PriceHistoryBuffer,
    ContinuousTestingEngine
)
import server


@pytest.fixture
def temp_db_path():
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, f"test_ct_{uuid.uuid4().hex}.db")
    yield db_file
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass


@pytest.fixture
def test_engine(temp_db_path):
    server.load_canonical_models()
    engine = ContinuousTestingEngine(
        models_dict=server.MODELS,
        app_metadata=server.APP_METADATA,
        db_path=temp_db_path
    )
    engine.attach_feed_manager(server.feed_manager)
    yield engine
    engine.stop()


# -----------------------------------------------------------------------------
# TEST 1: Continuous Prediction Scheduling
# -----------------------------------------------------------------------------
def test_continuous_prediction_scheduling(test_engine):
    """
    Verifies that the engine executes predictions on regular scheduled intervals.
    """
    class MockConnectedFeedManager:
        def __init__(self):
            self.tick = 0
        def get_current_snapshot(self):
            self.tick += 1
            return {
                "mode": "LIVE",
                "connected": True,
                "is_stale": False,
                "raw_market": {
                    "instrument": "Deribit BTC-PERPETUAL",
                    "mid_price": 68500.0,
                    "best_bid": 68499.5,
                    "best_ask": 68500.5,
                    "timestamp_str": f"2026-11-13 12:00:{self.tick:02d} UTC"
                },
                "features": {
                    "bid_ask_spread": 1.0,
                    "relative_spread_bps": 1.46,
                    "spread_expansion_pct": 5.0,
                    "book_depth_top5": 10000.0,
                    "volume_imbalance": 0.05,
                    "liquidity_withdrawal_rate": 0.05,
                    "realized_volatility_10m": 0.008,
                    "trade_intensity_10m": 1500,
                    "cancel_pressure": 0.0025,
                    "volatility_spread_interaction": 0.0004
                }
            }

    test_engine.feed_manager = MockConnectedFeedManager()
    test_engine.interval_sec = 0.5  # Fast 500ms test interval
    status = test_engine.start(interval_sec=0.5)
    assert status["status"] == "RUNNING"
    assert status["interval_sec"] == 0.5

    # Allow 1.6s to elapse so multiple ticks are scheduled
    time.sleep(1.6)
    test_engine.stop()

    recent = test_engine.get_recent_predictions(limit=10)
    assert len(recent) >= 2, f"Expected at least 2 scheduled predictions, got {len(recent)}"
    assert recent[0]["prediction_seq"] > recent[-1]["prediction_seq"] or len(recent) == 1


# -----------------------------------------------------------------------------
# TEST 2: Start, Pause, Resume, and Stop Controls
# -----------------------------------------------------------------------------
def test_start_pause_resume_stop_lifecycle(test_engine):
    """
    Verifies full lifecycle state transitions: STOPPED -> RUNNING -> PAUSED -> RUNNING -> STOPPED.
    """
    assert test_engine.status == "STOPPED"

    # 1. Start
    s1 = test_engine.start(interval_sec=1.0)
    assert s1["status"] == "RUNNING"
    assert test_engine.status == "RUNNING"

    # 2. Pause
    s2 = test_engine.pause()
    assert s2["status"] == "PAUSED"
    assert test_engine.status == "PAUSED"

    # Count predictions during pause to confirm no new ones are created
    time.sleep(0.6)
    preds_paused = len(test_engine.get_recent_predictions(limit=20))
    time.sleep(0.6)
    preds_paused_later = len(test_engine.get_recent_predictions(limit=20))
    assert preds_paused == preds_paused_later, "Engine generated predictions while PAUSED!"

    # 3. Resume
    s3 = test_engine.resume()
    assert s3["status"] == "RUNNING"
    assert test_engine.status == "RUNNING"

    # 4. Stop
    s4 = test_engine.stop()
    assert s4["status"] == "STOPPED"
    assert test_engine.status == "STOPPED"


# -----------------------------------------------------------------------------
# TEST 3: Model Feature Compatibility and Validation
# -----------------------------------------------------------------------------
def test_model_feature_compatibility_and_validation(test_engine):
    """
    Verifies that all 10 canonical features are required and validated before inference.
    """
    class MockFeedManager:
        def get_current_snapshot(self):
            return {
                "mode": "LIVE",
                "connected": True,
                "is_stale": False,
                "raw_market": {"mid_price": 68500.0, "best_bid": 68499.5, "best_ask": 68500.5},
                # Intentionally missing canonical features
                "features": {"bid_ask_spread": 1.0}
            }

    test_engine.feed_manager = MockFeedManager()
    test_engine._execute_single_testing_step()

    # Engine must refuse to run prediction and register a status warning
    assert test_engine.last_status_warning is not None
    assert "Missing required canonical features" in test_engine.last_status_warning
    assert test_engine.prediction_seq == 0


# -----------------------------------------------------------------------------
# TEST 4: Prediction Timestamp Ordering
# -----------------------------------------------------------------------------
def test_prediction_timestamp_ordering(temp_db_path):
    """
    Verifies that recorded predictions have monotonically non-decreasing timestamps and sequence numbers.
    """
    db = ContinuousTestingDatabase(temp_db_path)
    sess_id = "test_order_sess"
    db.create_session(sess_id, "LIVE", 5.0, 0.35)

    base_time = int(time.time() * 1000)
    for i in range(5):
        t_ms = base_time + (i * 5000)
        rec = {
            "session_id": sess_id,
            "prediction_seq": i + 1,
            "source_mode": "LIVE",
            "exchange_instrument": "Deribit BTC-PERPETUAL",
            "market_timestamp": f"2026-11-13 12:00:{i*5:02d} UTC",
            "timestamp_epoch_ms": t_ms,
            "features": {f: 1.0 for f in CANONICAL_FEATURES},
            "model_version": "FlashGuard-v2.2-XGB",
            "predicted_regime": "Liquid",
            "regime_probs": {"Liquid": 0.98, "Fragmented": 0.01, "Illiquid": 0.01},
            "rf_predicted_regime": "Liquid",
            "rf_probs": {"Liquid": 0.99, "Fragmented": 0.005, "Illiquid": 0.005},
            "downside_risk_prob": 0.15,
            "genuine_crash_prob": 0.02,
            "initial_mid_price": 68500.0,
            "evaluation_window_sec": 60.0,
            "target_eval_epoch_ms": t_ms + 60000,
            "latency_ms": 1.2,
            "alert_level": "NORMAL"
        }
        db.insert_prediction(rec)

    preds = db.get_recent_predictions(limit=10, session_id=sess_id)
    # get_recent_predictions returns desc order
    seqs = [p["prediction_seq"] for p in preds]
    assert seqs == [5, 4, 3, 2, 1], f"Expected sequence [5, 4, 3, 2, 1], got {seqs}"
    epochs = [p["timestamp_epoch_ms"] for p in preds]
    assert epochs == sorted(epochs, reverse=True), "Timestamps are not monotonically ordered!"


# -----------------------------------------------------------------------------
# TEST 5: 60-Second Outcome Evaluation Logic
# -----------------------------------------------------------------------------
def test_60_second_outcome_evaluation_logic(temp_db_path):
    """
    Verifies that the ground truth outcome identifies drops >= 0.1% correctly:
    P_min <= P_0 * 0.999 => Event 1, else Event 0.
    """
    db = ContinuousTestingDatabase(temp_db_path)
    buf = PriceHistoryBuffer()
    sess_id = "test_eval_logic"
    db.create_session(sess_id, "LIVE", 5.0, 0.35)

    t0 = 1000000
    p0 = 68500.0
    t_target = t0 + 60000

    # Insert Prediction 1: Risk prob = 0.40 (Elevated), expected to drop by 0.15% (TP)
    rec1 = {
        "session_id": sess_id,
        "prediction_seq": 1,
        "source_mode": "LIVE",
        "exchange_instrument": "Deribit BTC-PERPETUAL",
        "market_timestamp": "2026-11-13 12:00:00 UTC",
        "timestamp_epoch_ms": t0,
        "features": {},
        "predicted_regime": "Fragmented",
        "downside_risk_prob": 0.45,
        "initial_mid_price": p0,
        "target_eval_epoch_ms": t_target,
        "latency_ms": 1.5,
        "alert_level": "ELEVATED"
    }
    _, id1 = db.insert_prediction(rec1)

    # Provide price ticks with 0.15% drop (68500 * (1 - 0.0015) = 68397.25)
    for step_ms in range(0, 60001, 2000):
        price = p0 - (105.0 * (step_ms / 60000.0))  # drops to 68395.0
        buf.add_tick(t0 + step_ms, price)

    is_valid, min_p, _ = buf.evaluate_window(t0, t_target, max_allowed_gap_ms=5000)
    assert is_valid is True
    assert min_p <= p0 * 0.999

    actual_ret = round(((min_p / p0) - 1.0) * 100.0, 4)
    actual_event = 1 if actual_ret <= -0.1 else 0
    assert actual_event == 1

    # Update outcome
    db.update_prediction_outcome(
        pred_id=id1,
        outcome_status=1,
        min_future_mid_price=min_p,
        actual_return_pct=actual_ret,
        actual_downside_event=actual_event,
        evaluation_type="TP"
    )

    preds = db.get_recent_predictions(limit=1, session_id=sess_id)
    assert preds[0]["outcome_evaluated"] == 1
    assert preds[0]["evaluation_type"] == "TP"
    assert preds[0]["actual_downside_event"] == 1


# -----------------------------------------------------------------------------
# TEST 6: Confusion Matrix & Metrics Calculations
# -----------------------------------------------------------------------------
def test_confusion_matrix_calculations(temp_db_path):
    """
    Verifies quantitative accuracy of TP, FP, TN, FN, Precision, Recall, F1, FAR, and Brier Score.
    """
    db = ContinuousTestingDatabase(temp_db_path)
    sess_id = "test_metrics_sess"
    threshold = 0.35
    db.create_session(sess_id, "LIVE", 5.0, threshold)

    # We insert 4 synthetic evaluated predictions: 1 TP, 1 FP, 1 TN, 1 FN
    cases = [
        {"prob": 0.50, "actual": 1, "type": "TP"},
        {"prob": 0.40, "actual": 0, "type": "FP"},
        {"prob": 0.10, "actual": 0, "type": "TN"},
        {"prob": 0.20, "actual": 1, "type": "FN"},
    ]

    for idx, c in enumerate(cases):
        t_ms = 1000000 + idx * 5000
        rec = {
            "session_id": sess_id,
            "prediction_seq": idx + 1,
            "source_mode": "LIVE",
            "exchange_instrument": "Deribit BTC-PERPETUAL",
            "market_timestamp": f"2026-11-13 12:0{idx}:00 UTC",
            "timestamp_epoch_ms": t_ms,
            "features": {},
            "predicted_regime": "Liquid",
            "downside_risk_prob": c["prob"],
            "initial_mid_price": 68500.0,
            "target_eval_epoch_ms": t_ms + 60000,
            "latency_ms": 2.0,
            "alert_level": "NORMAL"
        }
        _, pid = db.insert_prediction(rec)
        db.update_prediction_outcome(
            pred_id=pid,
            outcome_status=1,
            min_future_mid_price=68400.0 if c["actual"] == 1 else 68500.0,
            actual_return_pct=-0.15 if c["actual"] == 1 else 0.0,
            actual_downside_event=c["actual"],
            evaluation_type=c["type"]
        )

    metrics = db.compute_metrics(session_id=sess_id, threshold=threshold)
    cm = metrics["confusion_matrix"]

    assert cm["TP"] == 1
    assert cm["FP"] == 1
    assert cm["TN"] == 1
    assert cm["FN"] == 1

    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    assert metrics["precision"] == 0.5
    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    assert metrics["recall"] == 0.5
    # F1-score = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    assert metrics["f1_score"] == 0.5
    # Accuracy = (1 + 1) / 4 = 0.5
    assert metrics["accuracy"] == 0.5
    # FAR = FP / (FP + TN) = 1 / 2 = 0.5
    assert metrics["false_alarm_rate"] == 0.5

    # Brier score = 1/4 * [(0.5-1)^2 + (0.4-0)^2 + (0.1-0)^2 + (0.2-1)^2]
    # = 1/4 * [0.25 + 0.16 + 0.01 + 0.64] = 1/4 * 1.06 = 0.265
    assert round(metrics["brier_score"], 3) == 0.265


# -----------------------------------------------------------------------------
# TEST 7: Data Gap Handling & Marking as INCOMPLETE
# -----------------------------------------------------------------------------
def test_data_gap_handling_incomplete(temp_db_path):
    """
    Verifies that if a market observation gap > 5s occurs or feed disconnects,
    the prediction outcome is strictly marked INCOMPLETE and never assumed negative.
    """
    buf = PriceHistoryBuffer()
    t0 = 2000000
    t_target = t0 + 60000

    # Add tick at start
    buf.add_tick(t0, 68500.0)
    # Gap of 10 seconds (> 5s threshold)
    buf.add_tick(t0 + 10000, 68500.0)
    # Rest of ticks
    for step in range(15000, 60001, 2000):
        buf.add_tick(t0 + step, 68500.0)

    is_valid, min_p, reason = buf.evaluate_window(t0, t_target, max_allowed_gap_ms=5000)
    assert is_valid is False
    assert min_p is None
    assert "GT_5S" in reason

    # Test explicit disconnect marker
    buf2 = PriceHistoryBuffer()
    for step in range(0, 30000, 1000):
        buf2.add_tick(t0 + step, 68500.0)
    buf2.mark_feed_gap(t0 + 31000)  # WS Disconnection logged
    for step in range(32000, 60001, 1000):
        buf2.add_tick(t0 + step, 68500.0)

    is_valid2, min_p2, reason2 = buf2.evaluate_window(t0, t_target, max_allowed_gap_ms=5000)
    assert is_valid2 is False
    assert min_p2 is None
    assert "FEED_DISCONNECTED" in reason2


# -----------------------------------------------------------------------------
# TEST 8: WebSocket Disconnect / Stale Feed Handling
# -----------------------------------------------------------------------------
def test_websocket_disconnect_and_stale_feed_handling(test_engine):
    """
    Verifies that when live feed is disconnected or stale, continuous engine
    pauses new inference and produces a clear advisory warning.
    """
    class DisconnectedFeedManager:
        def get_current_snapshot(self):
            return {
                "mode": "LIVE",
                "connected": False,  # Disconnected
                "is_stale": True,
                "raw_market": {"mid_price": 68500.0},
                "features": {f: 1.0 for f in CANONICAL_FEATURES}
            }

    test_engine.feed_manager = DisconnectedFeedManager()
    initial_seq = test_engine.prediction_seq
    test_engine._execute_single_testing_step()

    # Sequence must NOT increment
    assert test_engine.prediction_seq == initial_seq
    assert test_engine.last_status_warning is not None
    assert "disconnected" in test_engine.last_status_warning.lower()


# -----------------------------------------------------------------------------
# TEST 9: Duplicate Prediction Prevention
# -----------------------------------------------------------------------------
def test_duplicate_prediction_prevention(temp_db_path):
    """
    Verifies that identical observations with matching market timestamp are
    not inserted multiple times upon pause/resume or restart.
    """
    db = ContinuousTestingDatabase(temp_db_path)
    sess_id = "test_dup_sess"
    db.create_session(sess_id, "LIVE", 5.0, 0.35)

    rec = {
        "session_id": sess_id,
        "prediction_seq": 1,
        "source_mode": "LIVE",
        "exchange_instrument": "Deribit BTC-PERPETUAL",
        "market_timestamp": "2026-11-13 12:30:00 UTC",
        "timestamp_epoch_ms": 1794500000000,
        "features": {f: 1.0 for f in CANONICAL_FEATURES},
        "model_version": "FlashGuard-v2.2-XGB",
        "predicted_regime": "Liquid",
        "downside_risk_prob": 0.12,
        "initial_mid_price": 68500.0,
        "target_eval_epoch_ms": 1794500060000,
        "latency_ms": 1.4,
        "alert_level": "NORMAL"
    }

    # First insert -> Success
    inserted1, pid1 = db.insert_prediction(rec)
    assert inserted1 is True
    assert pid1 is not None

    # Duplicate insert -> Handled cleanly without exception or duplicate record
    inserted2, pid2 = db.insert_prediction(rec)
    assert inserted2 is False
    assert pid2 == pid1

    preds = db.get_recent_predictions(limit=10, session_id=sess_id)
    assert len(preds) == 1, f"Expected 1 record in DB, got {len(preds)}"


# -----------------------------------------------------------------------------
# TEST 10: Exported CSV Test Report Integrity
# -----------------------------------------------------------------------------
def test_exported_csv_test_report_format_and_contents(temp_db_path):
    """
    Verifies that the CSV test report contains all required headers, canonical features,
    metadata, and evaluation outcomes.
    """
    db = ContinuousTestingDatabase(temp_db_path)
    sess_id = "test_csv_sess"
    db.create_session(sess_id, "LIVE", 5.0, 0.35)

    rec = {
        "session_id": sess_id,
        "prediction_seq": 1,
        "source_mode": "LIVE",
        "exchange_instrument": "Deribit BTC-PERPETUAL",
        "market_timestamp": "2026-11-13 12:45:00 UTC",
        "timestamp_epoch_ms": 1794501000000,
        "features": {f: float(i + 1) for i, f in enumerate(CANONICAL_FEATURES)},
        "model_version": "FlashGuard-v2.2-XGB",
        "predicted_regime": "Liquid",
        "rf_predicted_regime": "Liquid",
        "downside_risk_prob": 0.22,
        "genuine_crash_prob": 0.03,
        "initial_mid_price": 68520.0,
        "target_eval_epoch_ms": 1794501060000,
        "latency_ms": 1.75,
        "alert_level": "NORMAL"
    }
    _, pid = db.insert_prediction(rec)
    db.update_prediction_outcome(
        pred_id=pid,
        outcome_status=1,
        min_future_mid_price=68510.0,
        actual_return_pct=-0.0146,
        actual_downside_event=0,
        evaluation_type="TN",
        data_quality_flags="EVALUATION_SUCCESS"
    )

    csv_str = db.export_csv_string(session_id=sess_id)
    assert len(csv_str) > 0

    lines = csv_str.strip().splitlines()
    header = lines[0].split(",")

    # Verify standard columns
    assert "prediction_id" in header
    assert "session_id" in header
    assert "source_mode" in header
    assert "initial_mid_price" in header
    assert "downside_risk_prob_60s" in header
    assert "evaluation_type" in header
    assert "actual_downside_event" in header

    # Verify all canonical features present in CSV headers
    for feat in CANONICAL_FEATURES:
        assert f"feature_{feat}" in header

    # Verify data line
    data_row = lines[1].split(",")
    assert data_row[header.index("session_id")] == sess_id
    assert data_row[header.index("evaluation_type")] == "TN"
    assert data_row[header.index("actual_downside_event")] == "0"
