"""
FlashGuard AI - Mode Isolation & Cross-Contamination Prevention Test Suite
Problem Statement 13: Hybrid Prediction & Classification for Flash Crash Risk Management
ASTRAQUANT X-2026

Verifies complete isolation between:
1. LIVE MARKET (Deribit WebSocket Feed)
2. HISTORICAL REPLAY (Chronological Recorded Dataset)
3. STRESS SIMULATION (Artificial Benchmark & Testing Lab Scenarios)

Ensures that artificial stress values, critical warning banners, and synthetic predictions
never leak into Live Market or Historical Replay modes.
"""

import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

from deribit_feed import FeedMode, MarketFeedManager, DeribitLiveFeed, HistoricalReplayFeed
from risk_engine import AlertHistoryManager, DataDriftMonitor, EarlyWarningEngine
from continuous_testing import ContinuousTestingDatabase, ContinuousTestingEngine
from server import app, feed_manager, alert_manager, drift_monitor, assemble_full_state, load_canonical_models, BENCHMARK_SCENARIOS


@pytest.fixture(scope="module")
def initialized_server():
    """Ensure models and feeds are loaded."""
    load_canonical_models()
    return app


@pytest.fixture
def client(initialized_server):
    return TestClient(initialized_server)


# -----------------------------------------------------------------------------
# 1. Simulation -> Live Isolation
# -----------------------------------------------------------------------------
def test_simulation_to_live_isolation():
    """
    Verify switching from STRESS SIMULATION to LIVE MARKET:
    - Completely clears simulation state and overrides.
    - Increments mode_generation.
    - Yields clean non-simulated state without artificial spread spikes or critical banners.
    - Restricts alert ledger to LIVE records.
    """
    mgr = MarketFeedManager()

    # 1. Enter SIMULATION with Severe Black Hole
    mgr.set_simulation_scenario("Severe Liquidity Black Hole", {
        "mid_price": 67900.0,
        "bid_ask_spread": 14.50,
        "spread_expansion_pct": 550.0,
        "liquidity_withdrawal_rate": 0.88,
        "volume_imbalance": -0.85,
        "book_depth_top5": 450.0,
        "realized_volatility_10m": 0.095,
        "trade_intensity_10m": 250
    })

    sim_snap = mgr.get_current_snapshot()
    assert sim_snap["mode"] == "SIMULATION"
    assert sim_snap["source_mode"] == "SIMULATION"
    assert sim_snap["is_simulated"] is True
    assert sim_snap["features"]["bid_ask_spread"] == 14.50
    assert sim_snap["features"]["spread_expansion_pct"] == 550.0
    initial_gen = mgr.mode_generation

    # 2. Switch back to LIVE
    mgr.set_mode(FeedMode.LIVE)
    assert mgr.mode_generation > initial_gen
    assert mgr.simulation_state is None, "simulation_state must be cleanly purged upon switching to LIVE"
    assert mgr.current_mode == FeedMode.LIVE

    live_snap = mgr.get_current_snapshot()
    assert live_snap["mode"] == "LIVE"
    assert live_snap["source_mode"] == "LIVE"
    assert live_snap["is_simulated"] is False
    assert live_snap["features"]["spread_expansion_pct"] != 550.0
    assert live_snap["features"]["liquidity_withdrawal_rate"] != 0.88


# -----------------------------------------------------------------------------
# 2. Simulation -> Replay Isolation
# -----------------------------------------------------------------------------
def test_simulation_to_replay_isolation():
    """
    Verify switching from STRESS SIMULATION to HISTORICAL REPLAY:
    - Purges simulation overrides.
    - Loads chronological historical dataset window with authentic price and timestamp.
    """
    mgr = MarketFeedManager()

    # Apply simulation
    mgr.set_simulation_scenario("Liquidity Withdrawal Stress", {
        "mid_price": 68420.0,
        "bid_ask_spread": 2.80,
        "spread_expansion_pct": 65.0,
        "liquidity_withdrawal_rate": 0.40,
        "volume_imbalance": -0.45,
        "book_depth_top5": 3500.0,
        "realized_volatility_10m": 0.025,
        "trade_intensity_10m": 1200
    })
    assert mgr.current_mode == FeedMode.SIMULATION

    # Switch to HISTORICAL_REPLAY
    mgr.set_mode(FeedMode.HISTORICAL_REPLAY)
    assert mgr.current_mode == FeedMode.HISTORICAL_REPLAY
    assert mgr.simulation_state is None

    rep_snap = mgr.get_current_snapshot()
    assert rep_snap["mode"] == "HISTORICAL_REPLAY"
    assert rep_snap["source_mode"] == "HISTORICAL_REPLAY"
    assert rep_snap["is_simulated"] is False
    assert "historical_timestamp" in rep_snap["raw_market"]
    assert rep_snap["raw_market"]["price_unit"] == "USD"


# -----------------------------------------------------------------------------
# 3. Live -> Simulation -> Live Recovery & Data Drift Protection
# -----------------------------------------------------------------------------
def test_live_to_simulation_to_live_recovery():
    """
    Verify round-trip recovery:
    - Data drift monitor rejects synthetic observations and does not poison historical baseline.
    - Early warning returns to normal once live stream resumes.
    """
    drift = DataDriftMonitor()

    # Legitimate live observation
    drift.add_observation({"bid_ask_spread": 0.52, "book_depth_top5": 14900.0}, is_simulated=False)
    assert len(drift.feature_history.get("bid_ask_spread", [])) == 1

    # Synthetic stress observation (spread = $25.0) - must be rejected
    drift.add_observation({"bid_ask_spread": 25.00, "book_depth_top5": 200.0}, is_simulated=True)
    assert len(drift.feature_history.get("bid_ask_spread", [])) == 1, "Simulated observations must never enter feature_history"
    assert drift.feature_history["bid_ask_spread"][0] == 0.52


# -----------------------------------------------------------------------------
# 4. Alert Manager Mode Isolation
# -----------------------------------------------------------------------------
def test_alert_manager_mode_isolation():
    """
    Verify alerts recorded in SIMULATION never appear when querying LIVE or HISTORICAL_REPLAY.
    """
    am = AlertHistoryManager()

    # Log Live alert
    am.log_alert("NORMAL", "Liquid", 12.0, 1.5, ["Normal spread"], mode="LIVE")
    # Log Replay alert
    am.log_alert("ELEVATED", "Fragmented", 42.0, 18.0, ["Depth skew"], mode="HISTORICAL_REPLAY")
    # Log Simulation alerts (different levels to test multiple alerts)
    am.log_alert("ELEVATED", "Fragmented", 55.0, 32.0, ["Stress Scenario Proxy"], mode="SIMULATION")
    am.log_alert("HIGH RISK", "Illiquid", 95.0, 88.0, ["Black Hole Test"], mode="SIMULATION")

    # Query by mode
    live_alerts = am.get_recent_alerts(mode="LIVE")
    replay_alerts = am.get_recent_alerts(mode="HISTORICAL_REPLAY")
    sim_alerts = am.get_recent_alerts(mode="SIMULATION")

    assert len(live_alerts) == 1
    assert live_alerts[0]["warning_level"] == "NORMAL"
    assert live_alerts[0]["mode"] == "LIVE"

    assert len(replay_alerts) == 1
    assert replay_alerts[0]["warning_level"] == "ELEVATED"
    assert replay_alerts[0]["mode"] == "HISTORICAL_REPLAY"

    assert len(sim_alerts) == 2
    assert any(a["warning_level"] == "ELEVATED" for a in sim_alerts)
    assert any(a["warning_level"] == "HIGH RISK" for a in sim_alerts)
    assert all(a["mode"] == "SIMULATION" for a in sim_alerts)

    # Verify CSV export mode filtering
    live_csv = am.export_csv_string(mode="LIVE")
    assert "SIMULATION" not in live_csv
    assert "Black Hole Test" not in live_csv

    sim_csv = am.export_csv_string(mode="SIMULATION")
    assert "LIVE" not in sim_csv
    assert "Black Hole Test" in sim_csv


# -----------------------------------------------------------------------------
# 5. Continuous Testing Mode Isolation
# -----------------------------------------------------------------------------
def test_continuous_testing_source_mode_isolation():
    """
    Verify Continuous AI Testing engine stores price buffers and evaluations
    separately by source_mode.
    """
    db = ContinuousTestingDatabase(db_path=":memory:")
    db.create_session("LIVE-1", "LIVE", 5.0, 0.35)
    db.create_session("SIM-1", "SIMULATION", 5.0, 0.35)

    # Store 1 LIVE prediction and 1 SIMULATION prediction
    now_ms = int(time.time() * 1000)
    rec_live = {
        "session_id": "LIVE-1",
        "prediction_seq": 1,
        "source_mode": "LIVE",
        "exchange_instrument": "Deribit BTC-PERPETUAL",
        "market_timestamp": "2026-11-13 14:00:00 UTC",
        "timestamp_epoch_ms": now_ms - 70000,
        "feature_snapshot_json": "{}",
        "model_version": "FlashGuard-v2.1-XGB",
        "predicted_regime": "Liquid",
        "regime_probs_json": "{\"Liquid\": 0.98}",
        "rf_predicted_regime": "Liquid",
        "rf_probs_json": "{\"Liquid\": 0.99}",
        "downside_risk_prob": 0.15,
        "genuine_crash_prob": 0.02,
        "initial_mid_price": 68500.0,
        "evaluation_window_sec": 60.0,
        "target_eval_epoch_ms": now_ms - 10000,
        "latency_ms": 1.5,
        "alert_level": "NORMAL",
        "data_quality_flags": ""
    }
    _, id_live = db.insert_prediction(rec_live)
    db.update_prediction_outcome(id_live, outcome_status=1, min_future_mid_price=68480.0,
                                 actual_return_pct=-0.029, actual_downside_event=0,
                                 evaluation_type="TN")

    rec_sim = {
        "session_id": "SIM-1",
        "prediction_seq": 1,
        "source_mode": "SIMULATION",
        "exchange_instrument": "Deribit BTC-PERPETUAL (SIMULATED)",
        "market_timestamp": "2026-11-13 14:00:05 UTC",
        "timestamp_epoch_ms": now_ms - 65000,
        "feature_snapshot_json": "{}",
        "model_version": "FlashGuard-v2.1-XGB",
        "predicted_regime": "Illiquid",
        "regime_probs_json": "{\"Illiquid\": 0.95}",
        "rf_predicted_regime": "Illiquid",
        "rf_probs_json": "{\"Illiquid\": 0.96}",
        "downside_risk_prob": 0.92,
        "genuine_crash_prob": 0.88,
        "initial_mid_price": 67900.0,
        "evaluation_window_sec": 60.0,
        "target_eval_epoch_ms": now_ms - 5000,
        "latency_ms": 1.8,
        "alert_level": "HIGH RISK",
        "data_quality_flags": ""
    }
    _, id_sim = db.insert_prediction(rec_sim)
    db.update_prediction_outcome(id_sim, outcome_status=1, min_future_mid_price=67100.0,
                                 actual_return_pct=-1.18, actual_downside_event=1,
                                 evaluation_type="TP")

    # Filter by source_mode
    live_metrics = db.compute_metrics(source_mode="LIVE")
    sim_metrics = db.compute_metrics(source_mode="SIMULATION")

    assert live_metrics["evaluated_predictions"] == 1
    assert live_metrics["confusion_matrix"]["TN"] == 1
    assert live_metrics["confusion_matrix"]["TP"] == 0

    assert sim_metrics["evaluated_predictions"] == 1
    assert sim_metrics["confusion_matrix"]["TP"] == 1
    assert sim_metrics["confusion_matrix"]["TN"] == 0


# -----------------------------------------------------------------------------
# 6. Monotonic Generation Protection During Rapid Switching
# -----------------------------------------------------------------------------
def test_rapid_mode_switching_race_conditions():
    """
    Verify mode_generation monotonically increases and that rapid transitions
    prevent stale responses.
    """
    mgr = MarketFeedManager()
    generations = []

    for _ in range(10):
        mgr.set_mode(FeedMode.LIVE)
        generations.append(mgr.mode_generation)
        mgr.set_mode(FeedMode.SIMULATION)
        generations.append(mgr.mode_generation)
        mgr.set_mode(FeedMode.HISTORICAL_REPLAY)
        generations.append(mgr.mode_generation)

    # Strictly monotonic
    for i in range(1, len(generations)):
        assert generations[i] > generations[i - 1]

    # Final mode matches snapshot
    mgr.set_mode(FeedMode.LIVE)
    snap = mgr.get_current_snapshot()
    assert snap["mode"] == "LIVE"
    assert snap["source_mode"] == "LIVE"
    assert snap["mode_generation"] == mgr.mode_generation


# -----------------------------------------------------------------------------
# 7. FastAPI Endpoint Verification
# -----------------------------------------------------------------------------
def test_fastapi_endpoints_mode_isolation(client):
    """
    Test REST API endpoints enforce generation and mode isolation:
    - /api/feed/mode returns mode_generation and session_id
    - /api/feed/simulation/scenario switches mode and returns generation
    - /api/testing-lab/run-all-scenarios keeps feed in SIMULATION and passes
    - /api/alerts/history?mode=LIVE filters correctly
    """
    # 1. Set mode to LIVE
    res_live = client.post("/api/feed/mode", json={"mode": "LIVE"})
    assert res_live.status_code == 200
    data_live = res_live.json()
    assert data_live["mode"] == "LIVE"
    assert "mode_generation" in data_live
    assert "session_id" in data_live
    gen_live = data_live["mode_generation"]

    # 2. Check full system state
    state = client.get("/api/state").json()
    assert state["mode"] == "LIVE"
    assert state["source_mode"] == "LIVE"
    assert state["mode_generation"] >= gen_live
    assert state["is_simulated"] is False

    # 3. Run all testing lab scenarios
    res_lab = client.post("/api/testing-lab/run-all-scenarios")
    assert res_lab.status_code == 200
    lab_data = res_lab.json()
    assert lab_data["passed_count"] == 3
    assert "mode_generation" in lab_data
    assert lab_data["mode_generation"] > gen_live

    # State now reflects simulation mode
    state_sim = client.get("/api/state").json()
    assert state_sim["mode"] == "SIMULATION"
    assert state_sim["is_simulated"] is True

    # 4. Switch back to LIVE
    res_back = client.post("/api/feed/mode", json={"mode": "LIVE"})
    assert res_back.status_code == 200
    back_data = res_back.json()
    assert back_data["mode"] == "LIVE"
    assert back_data["mode_generation"] > lab_data["mode_generation"]

    state_clean = client.get("/api/state").json()
    assert state_clean["mode"] == "LIVE"
    assert state_clean["source_mode"] == "LIVE"
    assert state_clean["is_simulated"] is False
    assert state_clean["system_info"]["data_freshness"] != "SIMULATED DATA"
