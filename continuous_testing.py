"""
FlashGuard AI — Continuous AI Testing & Validation Engine
Problem Statement 13: Hybrid Prediction & Classification for Flash Crash Risk Management
ASTRAQUANT X-2026

Core Capabilities:
1. Continuous Testing Controller (Start, Pause, Resume, Stop, Reset, Export)
2. Live Market Testing loop (Deribit WebSocket feed, rolling window, XGBoost & RF inference)
3. 60-Second Prediction Evaluation against actual price trajectory (min mid-price <= P_0 * 0.999)
4. Data Gap & Stale Feed Handling (marks INCOMPLETE on >5s gap or disconnect; never assumes negative)
5. Continuous Historical Replay & Synthetic Scenario Support
6. Real-Time Performance Metric Calculator (TP, FP, TN, FN, Precision, Recall, F1, FAR, Brier, PR-AUC)
7. Thread-safe SQLite Persistent Storage (flashguard_testing.db) with duplicate prevention
8. Comprehensive CSV Audit Export
"""

import os
import io
import csv
import time
import json
import sqlite3
import threading
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from feature_pipeline import (
    CANONICAL_FEATURES,
    TARGET_REGIME_NAMES,
    compute_features_from_snapshot
)
from deribit_feed import FeedMode

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "flashguard_testing.db")


# -----------------------------------------------------------------------------
# 1. Thread-Safe SQLite Persistence Layer
# -----------------------------------------------------------------------------
class ContinuousTestingDatabase:
    """
    Manages persistent SQLite storage for testing sessions, rolling predictions,
    and verified 60-second ground-truth evaluations.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        if db_path == ":memory:":
            self.db_path = f"file:mem_{id(self)}?mode=memory&cache=shared"
            self._is_uri = True
        else:
            self.db_path = db_path
            self._is_uri = db_path.startswith("file:")
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=20.0, uri=self._is_uri)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high-concurrency read/write performance
        if not self._is_uri:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS continuous_sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        source_mode TEXT NOT NULL,
                        interval_sec REAL NOT NULL,
                        decision_threshold REAL NOT NULL,
                        drop_threshold_pct REAL NOT NULL,
                        status TEXT NOT NULL,
                        total_predictions INTEGER DEFAULT 0,
                        evaluated_count INTEGER DEFAULT 0,
                        pending_count INTEGER DEFAULT 0,
                        incomplete_count INTEGER DEFAULT 0
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS continuous_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        prediction_seq INTEGER NOT NULL,
                        source_mode TEXT NOT NULL,
                        exchange_instrument TEXT NOT NULL,
                        market_timestamp TEXT NOT NULL,
                        timestamp_epoch_ms INTEGER NOT NULL,
                        feature_snapshot_json TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        predicted_regime TEXT NOT NULL,
                        regime_probs_json TEXT NOT NULL,
                        rf_predicted_regime TEXT NOT NULL,
                        rf_probs_json TEXT NOT NULL,
                        downside_risk_prob REAL NOT NULL,
                        genuine_crash_prob REAL NOT NULL,
                        initial_mid_price REAL NOT NULL,
                        evaluation_window_sec REAL DEFAULT 60.0,
                        target_eval_epoch_ms INTEGER NOT NULL,
                        outcome_evaluated INTEGER DEFAULT 0, -- 0: PENDING, 1: EVALUATED, -1: INCOMPLETE
                        min_future_mid_price REAL,
                        actual_return_pct REAL,
                        actual_downside_event INTEGER,       -- 0, 1, or NULL
                        evaluation_type TEXT DEFAULT 'PENDING', -- TP, FP, TN, FN, INCOMPLETE, PENDING
                        evaluation_timestamp TEXT,
                        latency_ms REAL NOT NULL,
                        alert_level TEXT NOT NULL,
                        data_quality_flags TEXT DEFAULT '',
                        is_duplicate INTEGER DEFAULT 0,
                        UNIQUE(session_id, market_timestamp)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_session ON continuous_predictions(session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_outcome ON continuous_predictions(outcome_evaluated);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_target_eval ON continuous_predictions(target_eval_epoch_ms);")
                conn.commit()

    def create_session(self, session_id: str, source_mode: str, interval_sec: float,
                       decision_threshold: float, drop_threshold_pct: float = 0.1):
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO continuous_sessions (
                        session_id, start_time, source_mode, interval_sec,
                        decision_threshold, drop_threshold_pct, status,
                        total_predictions, evaluated_count, pending_count, incomplete_count
                    ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', 0, 0, 0, 0)
                """, (session_id, now_str, source_mode, interval_sec, decision_threshold, drop_threshold_pct))
                conn.commit()

    def update_session_status(self, session_id: str, status: str):
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._lock:
            with self._get_connection() as conn:
                if status == "STOPPED":
                    conn.execute("UPDATE continuous_sessions SET status = ?, end_time = ? WHERE session_id = ?",
                                 (status, now_str, session_id))
                else:
                    conn.execute("UPDATE continuous_sessions SET status = ? WHERE session_id = ?",
                                 (status, session_id))
                conn.commit()

    def insert_prediction(self, record: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
        """
        Inserts a new prediction. Returns (is_inserted, prediction_id).
        If a duplicate (session_id, market_timestamp) already exists, returns (False, existing_id).
        """
        with self._lock:
            with self._get_connection() as conn:
                # Check for existing duplicate to prevent duplicate records upon pause/resume/restart
                cursor = conn.execute(
                    "SELECT id FROM continuous_predictions WHERE session_id = ? AND market_timestamp = ?",
                    (record["session_id"], record["market_timestamp"])
                )
                existing = cursor.fetchone()
                if existing:
                    return False, existing["id"]

                cursor = conn.execute("""
                    INSERT INTO continuous_predictions (
                        session_id, prediction_seq, source_mode, exchange_instrument,
                        market_timestamp, timestamp_epoch_ms, feature_snapshot_json,
                        model_version, predicted_regime, regime_probs_json,
                        rf_predicted_regime, rf_probs_json, downside_risk_prob,
                        genuine_crash_prob, initial_mid_price, evaluation_window_sec,
                        target_eval_epoch_ms, outcome_evaluated, evaluation_type,
                        latency_ms, alert_level, data_quality_flags
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'PENDING', ?, ?, ?
                    )
                """, (
                    record["session_id"],
                    record["prediction_seq"],
                    record["source_mode"],
                    record["exchange_instrument"],
                    record["market_timestamp"],
                    record["timestamp_epoch_ms"],
                    json.dumps(record.get("features", {})),
                    record.get("model_version", "FlashGuard-v2.2-XGB"),
                    record["predicted_regime"],
                    json.dumps(record.get("regime_probs", {})),
                    record.get("rf_predicted_regime", "Liquid"),
                    json.dumps(record.get("rf_probs", {})),
                    float(record["downside_risk_prob"]),
                    float(record.get("genuine_crash_prob", 0.0)),
                    float(record["initial_mid_price"]),
                    float(record.get("evaluation_window_sec", 60.0)),
                    int(record["target_eval_epoch_ms"]),
                    float(record.get("latency_ms", 1.5)),
                    record.get("alert_level", "NORMAL"),
                    record.get("data_quality_flags", "")
                ))
                new_id = cursor.lastrowid
                conn.execute(
                    "UPDATE continuous_sessions SET total_predictions = total_predictions + 1, pending_count = pending_count + 1 WHERE session_id = ?",
                    (record["session_id"],)
                )
                conn.commit()
                return True, new_id

    def get_pending_predictions(self, max_target_epoch_ms: Optional[int] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            with self._get_connection() as conn:
                query = "SELECT * FROM continuous_predictions WHERE outcome_evaluated = 0"
                params = []
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                if max_target_epoch_ms is not None:
                    query += " AND target_eval_epoch_ms <= ?"
                    params.append(max_target_epoch_ms)
                query += " ORDER BY target_eval_epoch_ms ASC"
                cursor = conn.execute(query, tuple(params))
                rows = [dict(r) for r in cursor.fetchall()]
                return rows

    def update_prediction_outcome(self, pred_id: int, outcome_status: int, min_future_mid_price: Optional[float],
                                  actual_return_pct: Optional[float], actual_downside_event: Optional[int],
                                  evaluation_type: str, data_quality_flags: str = ""):
        eval_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT session_id, outcome_evaluated FROM continuous_predictions WHERE id = ?", (pred_id,))
                row = cursor.fetchone()
                if not row:
                    return

                sess_id = row["session_id"]
                conn.execute("""
                    UPDATE continuous_predictions SET
                        outcome_evaluated = ?,
                        min_future_mid_price = ?,
                        actual_return_pct = ?,
                        actual_downside_event = ?,
                        evaluation_type = ?,
                        evaluation_timestamp = ?,
                        data_quality_flags = ?
                    WHERE id = ?
                """, (outcome_status, min_future_mid_price, actual_return_pct, actual_downside_event,
                      evaluation_type, eval_time_str, data_quality_flags, pred_id))

                # Update session counters
                if outcome_status == 1:
                    conn.execute("""
                        UPDATE continuous_sessions SET
                            pending_count = MAX(0, pending_count - 1),
                            evaluated_count = evaluated_count + 1
                        WHERE session_id = ?
                    """, (sess_id,))
                elif outcome_status == -1:
                    conn.execute("""
                        UPDATE continuous_sessions SET
                            pending_count = MAX(0, pending_count - 1),
                            incomplete_count = incomplete_count + 1
                        WHERE session_id = ?
                    """, (sess_id,))

                conn.commit()

    def get_recent_predictions(self, limit: int = 50, session_id: Optional[str] = None, source_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            with self._get_connection() as conn:
                query = "SELECT * FROM continuous_predictions"
                where_clauses = []
                params = []
                if session_id:
                    where_clauses.append("session_id = ?")
                    params.append(session_id)
                if source_mode:
                    where_clauses.append("source_mode = ?")
                    params.append(source_mode.upper())
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                query += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                cursor = conn.execute(query, tuple(params))
                rows = []
                for r in cursor.fetchall():
                    item = dict(r)
                    try:
                        item["features"] = json.loads(item.get("feature_snapshot_json", "{}"))
                    except Exception:
                        item["features"] = {}
                    try:
                        item["regime_probs"] = json.loads(item.get("regime_probs_json", "{}"))
                    except Exception:
                        item["regime_probs"] = {}
                    try:
                        item["rf_probs"] = json.loads(item.get("rf_probs_json", "{}"))
                    except Exception:
                        item["rf_probs"] = {}
                    rows.append(item)
                return rows

    def compute_metrics(self, session_id: Optional[str] = None, threshold: float = 0.35, source_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Computes real-time confusion matrix and performance statistics on all evaluated predictions.
        Strictly excludes incomplete/data-gap observations and isolates evaluations by source_mode.
        """
        with self._lock:
            with self._get_connection() as conn:
                # Total counts
                count_query = "SELECT COUNT(*) as total, " \
                              "SUM(CASE WHEN outcome_evaluated = 0 THEN 1 ELSE 0 END) as pending, " \
                              "SUM(CASE WHEN outcome_evaluated = 1 THEN 1 ELSE 0 END) as evaluated, " \
                              "SUM(CASE WHEN outcome_evaluated = -1 THEN 1 ELSE 0 END) as incomplete " \
                              "FROM continuous_predictions"
                where_clauses = []
                params = []
                if session_id:
                    where_clauses.append("session_id = ?")
                    params.append(session_id)
                if source_mode:
                    where_clauses.append("source_mode = ?")
                    params.append(source_mode.upper())
                if where_clauses:
                    count_query += " WHERE " + " AND ".join(where_clauses)
                cursor = conn.execute(count_query, tuple(params))
                c_row = cursor.fetchone()
                total_preds = c_row["total"] or 0
                pending_count = c_row["pending"] or 0
                evaluated_count = c_row["evaluated"] or 0
                incomplete_count = c_row["incomplete"] or 0

                # Fetch evaluated records
                eval_query = "SELECT downside_risk_prob, actual_downside_event, evaluation_type, latency_ms " \
                             "FROM continuous_predictions WHERE outcome_evaluated = 1"
                eval_params = []
                if session_id:
                    eval_query += " AND session_id = ?"
                    eval_params.append(session_id)
                if source_mode:
                    eval_query += " AND source_mode = ?"
                    eval_params.append(source_mode.upper())
                cursor = conn.execute(eval_query, tuple(eval_params))
                eval_rows = cursor.fetchall()

                # Get latest timestamps
                ts_query = "SELECT market_timestamp FROM continuous_predictions"
                ts_params = []
                ts_clauses = []
                if session_id:
                    ts_clauses.append("session_id = ?")
                    ts_params.append(session_id)
                if source_mode:
                    ts_clauses.append("source_mode = ?")
                    ts_params.append(source_mode.upper())
                if ts_clauses:
                    ts_query += " WHERE " + " AND ".join(ts_clauses)
                ts_query += " ORDER BY id DESC LIMIT 1"
                c_ts = conn.execute(ts_query, tuple(ts_params)).fetchone()
                latest_pred_ts = c_ts["market_timestamp"] if c_ts else "None"

                eval_ts_query = "SELECT evaluation_timestamp FROM continuous_predictions WHERE outcome_evaluated = 1"
                eval_ts_params = []
                if session_id:
                    eval_ts_query += " AND session_id = ?"
                    eval_ts_params.append(session_id)
                if source_mode:
                    eval_ts_query += " AND source_mode = ?"
                    eval_ts_params.append(source_mode.upper())
                eval_ts_query += " ORDER BY id DESC LIMIT 1"
                c_eval_ts = conn.execute(eval_ts_query, tuple(eval_ts_params)).fetchone()
                latest_eval_ts = c_eval_ts["evaluation_timestamp"] if c_eval_ts else "None"

        if not eval_rows:
            return {
                "total_predictions": total_preds,
                "pending_evaluations": pending_count,
                "evaluated_predictions": evaluated_count,
                "incomplete_predictions": incomplete_count,
                "latest_prediction_timestamp": latest_pred_ts,
                "latest_evaluated_timestamp": latest_eval_ts,
                "average_latency_ms": 0.0,
                "decision_threshold": threshold,
                "confusion_matrix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "false_alarm_rate": 0.0,
                "brier_score": 0.0,
                "pr_auc": None
            }

        probs = [r["downside_risk_prob"] for r in eval_rows]
        actuals = [r["actual_downside_event"] for r in eval_rows]
        latencies = [r["latency_ms"] for r in eval_rows]

        tp = sum(1 for p, a in zip(probs, actuals) if p >= threshold and a == 1)
        fp = sum(1 for p, a in zip(probs, actuals) if p >= threshold and a == 0)
        tn = sum(1 for p, a in zip(probs, actuals) if p < threshold and a == 0)
        fn = sum(1 for p, a in zip(probs, actuals) if p < threshold and a == 1)

        n = len(actuals)
        acc = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        try:
            brier = float(brier_score_loss(actuals, probs))
        except Exception:
            brier = float(np.mean([(p - a) ** 2 for p, a in zip(probs, actuals)]))

        pr_auc = None
        if len(set(actuals)) > 1:
            try:
                pr_auc = round(float(average_precision_score(actuals, probs)), 4)
            except Exception:
                pr_auc = None

        return {
            "total_predictions": total_preds,
            "pending_evaluations": pending_count,
            "evaluated_predictions": evaluated_count,
            "incomplete_predictions": incomplete_count,
            "latest_prediction_timestamp": latest_pred_ts,
            "latest_evaluated_timestamp": latest_eval_ts,
            "average_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "decision_threshold": threshold,
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_alarm_rate": round(far, 4),
            "brier_score": round(brier, 4),
            "pr_auc": pr_auc
        }

    def export_csv_string(self, session_id: Optional[str] = None, source_mode: Optional[str] = None) -> str:
        """
        Exports all predictions and evaluations to CSV format with full metadata and features.
        Supports filtering by source_mode for strict mode isolation.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        headers = [
            "prediction_id",
            "session_id",
            "prediction_seq",
            "source_mode",
            "exchange_instrument",
            "market_timestamp",
            "timestamp_epoch_ms",
            "model_version",
            "initial_mid_price",
            "predicted_regime",
            "rf_predicted_regime",
            "downside_risk_prob_60s",
            "genuine_crash_prob",
            "alert_level",
            "latency_ms",
            "outcome_evaluated",
            "evaluation_type",
            "evaluation_timestamp",
            "min_future_mid_price_60s",
            "actual_price_return_pct_60s",
            "actual_downside_event",
            "data_quality_flags"
        ]
        # Append canonical feature columns
        for feat in CANONICAL_FEATURES:
            headers.append(f"feature_{feat}")

        writer.writerow(headers)

        with self._lock:
            with self._get_connection() as conn:
                query = "SELECT * FROM continuous_predictions"
                where_clauses = []
                params = []
                if session_id:
                    where_clauses.append("session_id = ?")
                    params.append(session_id)
                if source_mode:
                    where_clauses.append("source_mode = ?")
                    params.append(source_mode.upper())
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                query += " ORDER BY id ASC"
                cursor = conn.execute(query, tuple(params))
                for row in cursor.fetchall():
                    item = dict(row)
                    feat_dict = {}
                    try:
                        feat_dict = json.loads(item.get("feature_snapshot_json", "{}"))
                    except Exception:
                        pass

                    row_vals = [
                        item["id"],
                        item["session_id"],
                        item["prediction_seq"],
                        item["source_mode"],
                        item["exchange_instrument"],
                        item["market_timestamp"],
                        item["timestamp_epoch_ms"],
                        item["model_version"],
                        item["initial_mid_price"],
                        item["predicted_regime"],
                        item["rf_predicted_regime"],
                        item["downside_risk_prob"],
                        item["genuine_crash_prob"],
                        item["alert_level"],
                        item["latency_ms"],
                        item["outcome_evaluated"],
                        item["evaluation_type"],
                        item.get("evaluation_timestamp") or "",
                        item.get("min_future_mid_price") if item.get("min_future_mid_price") is not None else "",
                        item.get("actual_return_pct") if item.get("actual_return_pct") is not None else "",
                        item.get("actual_downside_event") if item.get("actual_downside_event") is not None else "",
                        item.get("data_quality_flags") or ""
                    ]
                    for feat in CANONICAL_FEATURES:
                        row_vals.append(feat_dict.get(feat, ""))

                    writer.writerow(row_vals)

        return output.getvalue()

    def reset_session_data(self, session_id: str):
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM continuous_predictions WHERE session_id = ?", (session_id,))
                conn.execute("""
                    UPDATE continuous_sessions SET
                        total_predictions = 0,
                        evaluated_count = 0,
                        pending_count = 0,
                        incomplete_count = 0
                    WHERE session_id = ?
                """, (session_id,))
                conn.commit()


# -----------------------------------------------------------------------------
# 2. Chronological Market Price Buffer with Data-Gap Detection
# -----------------------------------------------------------------------------
class PriceHistoryBuffer:
    """
    Maintains a rolling window of recent market ticks with millisecond timestamps.
    Enables exact ground-truth outcome evaluation for past predictions and
    identifies gaps > 5.0 seconds or disconnection events.
    """
    def __init__(self, maxlen: int = 5000):
        self.maxlen = maxlen
        self.ticks: List[Dict[str, Any]] = []  # dict: epoch_ms, price, is_gap_marker
        self._lock = threading.Lock()

    def add_tick(self, epoch_ms: int, mid_price: float, is_gap_marker: bool = False):
        if mid_price <= 0 and not is_gap_marker:
            return
        with self._lock:
            self.ticks.append({
                "epoch_ms": int(epoch_ms),
                "mid_price": float(mid_price),
                "is_gap": bool(is_gap_marker)
            })
            if len(self.ticks) > self.maxlen:
                self.ticks.pop(0)

    def mark_feed_gap(self, epoch_ms: int):
        with self._lock:
            self.ticks.append({
                "epoch_ms": int(epoch_ms),
                "mid_price": 0.0,
                "is_gap": True
            })
            if len(self.ticks) > self.maxlen:
                self.ticks.pop(0)

    def get_latest_price(self) -> float:
        with self._lock:
            for t in reversed(self.ticks):
                if not t["is_gap"] and t["mid_price"] > 0:
                    return t["mid_price"]
            return 0.0

    def get_latest_epoch_ms(self) -> Optional[int]:
        with self._lock:
            for t in reversed(self.ticks):
                if not t["is_gap"] and t["mid_price"] > 0:
                    return t["epoch_ms"]
            return None

    def evaluate_window(self, start_epoch_ms: int, end_epoch_ms: int,
                        max_allowed_gap_ms: int = 5000) -> Tuple[bool, Optional[float], str]:
        """
        Inspects ticks in [start_epoch_ms, end_epoch_ms].
        Returns:
            (is_valid, min_price, reason)
            If there are no ticks, or if any gap between consecutive ticks exceeds max_allowed_gap_ms (5s),
            or if a gap/disconnect was explicitly logged, returns (False, None, 'DATA_GAP_GT_5S').
        """
        with self._lock:
            window_ticks = [
                t for t in self.ticks
                if start_epoch_ms <= t["epoch_ms"] <= end_epoch_ms
            ]
            trailing_ticks = [
                t for t in self.ticks
                if t["epoch_ms"] >= end_epoch_ms
            ]

        if not window_ticks:
            return False, None, "NO_TICKS_IN_WINDOW"

        # Check for explicit gap markers
        for t in window_ticks:
            if t["is_gap"]:
                return False, None, "FEED_DISCONNECTED_DURING_WINDOW"

        # Check for gap at start of window
        if (window_ticks[0]["epoch_ms"] - start_epoch_ms) > max_allowed_gap_ms:
            return False, None, "INITIAL_TICK_GAP_GT_5S"

        # Check intervals between consecutive ticks
        valid_prices = []
        for i in range(len(window_ticks)):
            valid_prices.append(window_ticks[i]["mid_price"])
            if i > 0:
                gap = window_ticks[i]["epoch_ms"] - window_ticks[i - 1]["epoch_ms"]
                if gap > max_allowed_gap_ms:
                    return False, None, f"TICK_GAP_{gap}MS_GT_5S"

        # Check gap at end of window
        trailing_gap = end_epoch_ms - window_ticks[-1]["epoch_ms"]
        if trailing_gap > max_allowed_gap_ms:
            if trailing_ticks:
                boundary_gap = trailing_ticks[0]["epoch_ms"] - window_ticks[-1]["epoch_ms"]
                if boundary_gap > max_allowed_gap_ms:
                    return False, None, "TRAILING_TICK_GAP_GT_5S"
                valid_prices.append(trailing_ticks[0]["mid_price"])
            else:
                return False, None, "TRAILING_TICK_GAP_GT_5S"

        min_price = min(valid_prices)
        return True, min_price, "VALID_WINDOW"


# -----------------------------------------------------------------------------
# 3. Continuous Testing Engine Core
# -----------------------------------------------------------------------------
class ContinuousTestingEngine:
    """
    Background engine that continuously samples market state, executes model
    inference, registers predictions, and tracks 60-second forward evaluation
    windows to calculate real-time ground truth without future data leakage.
    """
    def __init__(self, models_dict: Dict[str, Any], app_metadata: Dict[str, Any],
                 db_path: str = DEFAULT_DB_PATH):
        self.models = models_dict
        self.metadata = app_metadata
        self.db = ContinuousTestingDatabase(db_path)
        # Partitioned independent price buffers for strict mode isolation
        self.price_buffers = {
            "LIVE": PriceHistoryBuffer(maxlen=6000),
            "HISTORICAL_REPLAY": PriceHistoryBuffer(maxlen=6000),
            "SIMULATION": PriceHistoryBuffer(maxlen=6000)
        }
        self.price_buffer = self.price_buffers["LIVE"]  # alias for backward compatibility

        # Operational Parameters
        self.interval_sec: float = 5.0
        self.decision_threshold: float = 0.35
        self.drop_threshold_pct: float = 0.1     # 0.1% downside price drop
        self.eval_window_sec: float = 60.0       # 60 seconds
        self.active_mode: str = "LIVE"

        # State Control
        self.status: str = "STOPPED"            # RUNNING, PAUSED, STOPPED
        self.session_id: str = f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.prediction_seq: int = 0
        self.last_prediction_time: float = 0.0
        self.last_eval_check_time: float = 0.0

        self.last_status_warning: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Connect feed manager
        self.feed_manager = None

    def attach_feed_manager(self, feed_manager):
        self.feed_manager = feed_manager

    def start(self, interval_sec: Optional[float] = None, decision_threshold: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            if interval_sec is not None and interval_sec >= 0.5:
                self.interval_sec = float(interval_sec)
            if decision_threshold is not None and 0.0 <= decision_threshold <= 1.0:
                self.decision_threshold = float(decision_threshold)

            if self.status == "RUNNING":
                return self.get_status()

            # Start fresh session if previously stopped
            if self.status == "STOPPED":
                self.session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                self.prediction_seq = 0
                self.db.create_session(
                    session_id=self.session_id,
                    source_mode=self.active_mode,
                    interval_sec=self.interval_sec,
                    decision_threshold=self.decision_threshold,
                    drop_threshold_pct=self.drop_threshold_pct
                )

            self.status = "RUNNING"
            self.db.update_session_status(self.session_id, "RUNNING")
            self._stop_event.clear()

            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._background_loop, daemon=True)
                self._thread.start()

            print(f"[+] Continuous AI Testing Engine STARTED: Session={self.session_id}, Interval={self.interval_sec}s, Threshold={self.decision_threshold}")
            return self.get_status()

    def pause(self) -> Dict[str, Any]:
        with self._lock:
            if self.status == "RUNNING":
                self.status = "PAUSED"
                self.db.update_session_status(self.session_id, "PAUSED")
                print(f"[*] Continuous AI Testing Engine PAUSED: Session={self.session_id}")
            return self.get_status()

    def resume(self) -> Dict[str, Any]:
        with self._lock:
            if self.status == "PAUSED":
                self.status = "RUNNING"
                self.db.update_session_status(self.session_id, "RUNNING")
                print(f"[+] Continuous AI Testing Engine RESUMED: Session={self.session_id}")
            return self.get_status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self.status = "STOPPED"
            self.db.update_session_status(self.session_id, "STOPPED")
            self._stop_event.set()
            print(f"[*] Continuous AI Testing Engine STOPPED: Session={self.session_id}")
            return self.get_status()

    def reset_session(self) -> Dict[str, Any]:
        with self._lock:
            was_running = (self.status == "RUNNING")
            self.status = "STOPPED"
            self._stop_event.set()
            self.db.reset_session_data(self.session_id)
            self.session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            self.prediction_seq = 0
            self.db.create_session(
                session_id=self.session_id,
                source_mode=self.active_mode,
                interval_sec=self.interval_sec,
                decision_threshold=self.decision_threshold,
                drop_threshold_pct=self.drop_threshold_pct
            )
            if was_running:
                self.status = "RUNNING"
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._background_loop, daemon=True)
                self._thread.start()
            return self.get_status()

    def update_config(self, interval_sec: Optional[float] = None, decision_threshold: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            if interval_sec is not None and interval_sec >= 0.5:
                self.interval_sec = float(interval_sec)
            if decision_threshold is not None and 0.0 <= decision_threshold <= 1.0:
                self.decision_threshold = float(decision_threshold)
            return self.get_status()

    def record_market_tick(self, epoch_ms: int, mid_price: float, mode: str = "LIVE"):
        """Allows feeding external price ticks directly into mode-isolated price buffer."""
        norm_mode = mode.upper() if mode else "LIVE"
        buf = self.price_buffers.get(norm_mode, self.price_buffers["LIVE"])
        buf.add_tick(epoch_ms, mid_price)

    def _background_loop(self):
        """
        Background evaluation loop. Samples market observation at regular intervals,
        executes AI predictions, buffers ticks in mode-isolated buffers, and checks for mature 60-second evaluations.
        """
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                # 1. Update Market Price Buffer continuously from feed
                if self.feed_manager is not None:
                    now_ms = int(time.time() * 1000)

                    # Always feed LIVE price buffer from live_feed if active
                    if hasattr(self.feed_manager, "live_feed"):
                        live_snap = self.feed_manager.live_feed.get_snapshot()
                        live_conn = live_snap.get("connected", True)
                        live_stale = live_snap.get("is_stale", False)
                        live_raw = live_snap.get("raw_market", {})
                        live_mid = float(live_raw.get("mid_price", 0.0))
                        if not live_conn or live_stale:
                            self.price_buffers["LIVE"].mark_feed_gap(now_ms)
                        elif live_mid > 0:
                            self.price_buffers["LIVE"].add_tick(now_ms, live_mid)

                    # Also feed active mode if it is SIMULATION or HISTORICAL_REPLAY
                    snapshot = self.feed_manager.get_current_snapshot()
                    cur_mode = snapshot.get("mode", FeedMode.LIVE.value).upper()
                    if cur_mode != FeedMode.LIVE.value:
                        buf = self.price_buffers.get(cur_mode)
                        if buf is not None:
                            raw_mkt = snapshot.get("raw_market", {})
                            mid_p = float(raw_mkt.get("mid_price", 0.0))
                            if mid_p > 0:
                                buf.add_tick(now_ms, mid_p)

                # 2. Check if it is time to make a new continuous prediction
                if self.status == "RUNNING":
                    elapsed_since_pred = time.time() - self.last_prediction_time
                    if elapsed_since_pred >= self.interval_sec:
                        self._execute_single_testing_step()
                        self.last_prediction_time = time.time()

                    # 3. Evaluate mature 60-second outcomes
                    self._evaluate_mature_predictions()

            except Exception as e:
                print(f"[!] Error in continuous testing background loop: {e}")

            time.sleep(0.2)  # High resolution poll allows responsive pause/stop

    def _execute_single_testing_step(self):
        """
        Gathers current valid market observation, verifies canonical schema,
        runs XGBoost and Random Forest inference, and stores the prediction to SQLite.
        """
        if self.feed_manager is None:
            self.last_status_warning = "Feed manager not attached"
            return

        snapshot = self.feed_manager.get_current_snapshot()
        mode = snapshot.get("mode", FeedMode.LIVE.value).upper()
        self.active_mode = mode

        # Verification of live feed connectivity
        if mode == FeedMode.LIVE.value:
            if not snapshot.get("connected", False):
                self.last_status_warning = "Deribit WebSocket disconnected. Pausing prediction inference."
                return
            if snapshot.get("is_stale", False):
                self.last_status_warning = "Market data stale (>3s). Pausing prediction inference."
                return

        feat_dict = snapshot.get("features", {})
        raw_mkt = snapshot.get("raw_market", {})
        mid_price = float(raw_mkt.get("mid_price", 0.0))

        # Schema & Sanity Validations
        if mid_price <= 0:
            self.last_status_warning = "Mid price invalid (<=0)."
            return

        best_bid = float(raw_mkt.get("best_bid", 0.0))
        best_ask = float(raw_mkt.get("best_ask", 0.0))
        if best_bid > 0 and best_ask > 0 and best_ask < best_bid:
            self.last_status_warning = "Negative bid-ask spread detected. Observation rejected."
            return

        # Canonical Feature Validation
        missing_feats = [f for f in CANONICAL_FEATURES if f not in feat_dict]
        if missing_feats:
            self.last_status_warning = f"Missing required canonical features: {missing_feats}"
            return

        feature_vector = np.array([float(feat_dict[f]) for f in CANONICAL_FEATURES], dtype=np.float32)

        # Execute verified model inference
        t0 = time.perf_counter()
        X = feature_vector.reshape(1, -1)
        X_scaled = self.models["scaler"].transform(X) if "scaler" in self.models else X

        # Primary XGBoost Classifier
        xgb_probs = self.models["xgb_cls"].predict_proba(X)[0]
        xgb_idx = int(np.argmax(xgb_probs))
        predicted_regime = TARGET_REGIME_NAMES[xgb_idx]

        # Benchmark Random Forest Classifier
        rf_probs = self.models["rf_cls"].predict_proba(X)[0]
        rf_idx = int(np.argmax(rf_probs))
        rf_regime = TARGET_REGIME_NAMES[rf_idx]

        # 60-Second Downside Risk Regressor
        downside_pred = float(self.models["xgb_reg"].predict(X)[0])
        downside_prob = float(np.clip(downside_pred, 0.0, 1.0))

        # Genuine Flash Crash Classifier
        if "xgb_crash_cls" in self.models:
            crash_probs = self.models["xgb_crash_cls"].predict_proba(X)[0]
            crash_prob = float(np.clip(crash_probs[1], 0.0, 1.0))
        else:
            crash_prob = 0.02

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Determine Alert Level
        if downside_prob >= 0.65 or crash_prob >= 0.40:
            alert_lvl = "HIGH"
        elif downside_prob >= 0.35 or crash_prob >= 0.15:
            alert_lvl = "ELEVATED"
        else:
            alert_lvl = "NORMAL"

        now_ms = int(time.time() * 1000)
        ts_str = raw_mkt.get("historical_timestamp") or raw_mkt.get("timestamp_str") or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        self.prediction_seq += 1
        record = {
            "session_id": self.session_id,
            "prediction_seq": self.prediction_seq,
            "source_mode": mode,
            "exchange_instrument": raw_mkt.get("instrument", "Deribit BTC-PERPETUAL"),
            "market_timestamp": f"{ts_str} [#{self.prediction_seq}]" if mode == FeedMode.LIVE.value else f"{ts_str} [#{self.prediction_seq}]",
            "timestamp_epoch_ms": now_ms,
            "features": feat_dict,
            "model_version": self.metadata.get("model_version", "FlashGuard-v2.2-XGB"),
            "predicted_regime": predicted_regime,
            "regime_probs": {TARGET_REGIME_NAMES[i]: round(float(p), 4) for i, p in enumerate(xgb_probs)},
            "rf_predicted_regime": rf_regime,
            "rf_probs": {TARGET_REGIME_NAMES[i]: round(float(p), 4) for i, p in enumerate(rf_probs)},
            "downside_risk_prob": round(downside_prob, 4),
            "genuine_crash_prob": round(crash_prob, 4),
            "initial_mid_price": round(mid_price, 2),
            "evaluation_window_sec": self.eval_window_sec,
            "target_eval_epoch_ms": now_ms + int(self.eval_window_sec * 1000),
            "latency_ms": latency_ms,
            "alert_level": alert_lvl,
            "data_quality_flags": "VALID_OBSERVATION"
        }

        inserted, pred_id = self.db.insert_prediction(record)
        if inserted:
            self.last_status_warning = None
        else:
            self.last_status_warning = "Duplicate observation timestamp skipped."

    def _evaluate_mature_predictions(self):
        """
        Finds pending predictions where target evaluation time has elapsed.
        Checks mode-isolated price buffer for minimum price and verifies no data gaps occurred.
        Categorizes result as TP, FP, TN, FN, or INCOMPLETE.
        """
        now_ms = int(time.time() * 1000)
        pending = self.db.get_pending_predictions(max_target_epoch_ms=now_ms, session_id=self.session_id)

        for pred in pending:
            pred_id = pred["id"]
            start_ms = pred["timestamp_epoch_ms"]
            target_ms = pred["target_eval_epoch_ms"]
            p0 = pred["initial_mid_price"]
            prob = pred["downside_risk_prob"]
            pred_mode = pred.get("source_mode", "LIVE").upper()
            buf = self.price_buffers.get(pred_mode, self.price_buffers["LIVE"])

            # Check if closing tick for the 60-second window has arrived
            latest_tick_ms = buf.get_latest_epoch_ms()
            if latest_tick_ms is not None and latest_tick_ms < target_ms:
                # Still within allowable jitter (< 6s past target); wait for closing tick before marking incomplete
                if now_ms < target_ms + 6000:
                    continue

            is_valid, min_p, reason = buf.evaluate_window(start_ms, target_ms, max_allowed_gap_ms=5000)

            if not is_valid:
                # Mark as INCOMPLETE - never assume negative when data was lost!
                self.db.update_prediction_outcome(
                    pred_id=pred_id,
                    outcome_status=-1,
                    min_future_mid_price=None,
                    actual_return_pct=None,
                    actual_downside_event=None,
                    evaluation_type="INCOMPLETE",
                    data_quality_flags=reason
                )
            else:
                # Valid price path
                actual_ret_pct = round(((min_p / p0) - 1.0) * 100.0, 4)
                drop_threshold = -self.drop_threshold_pct  # -0.1%
                actual_event = 1 if actual_ret_pct <= drop_threshold else 0
                predicted_event = 1 if prob >= self.decision_threshold else 0

                if predicted_event == 1 and actual_event == 1:
                    eval_type = "TP"
                elif predicted_event == 1 and actual_event == 0:
                    eval_type = "FP"
                elif predicted_event == 0 and actual_event == 0:
                    eval_type = "TN"
                else:
                    eval_type = "FN"

                self.db.update_prediction_outcome(
                    pred_id=pred_id,
                    outcome_status=1,
                    min_future_mid_price=round(min_p, 2),
                    actual_return_pct=actual_ret_pct,
                    actual_downside_event=actual_event,
                    evaluation_type=eval_type,
                    data_quality_flags="EVALUATION_SUCCESS"
                )

    def get_status(self, source_mode: Optional[str] = None) -> Dict[str, Any]:
        """Provides status summary strictly partitioned by source_mode."""
        mode_filter = source_mode.upper() if source_mode else self.active_mode
        metrics = self.db.compute_metrics(session_id=self.session_id, threshold=self.decision_threshold, source_mode=mode_filter)
        return {
            "status": self.status,
            "session_id": self.session_id,
            "mode": mode_filter,
            "interval_sec": self.interval_sec,
            "decision_threshold": self.decision_threshold,
            "drop_threshold_pct": self.drop_threshold_pct,
            "evaluation_window_sec": self.eval_window_sec,
            "total_predictions": metrics["total_predictions"],
            "pending_evaluations": metrics["pending_evaluations"],
            "evaluated_predictions": metrics["evaluated_predictions"],
            "incomplete_predictions": metrics["incomplete_predictions"],
            "latest_prediction_timestamp": metrics["latest_prediction_timestamp"],
            "latest_evaluated_timestamp": metrics["latest_evaluated_timestamp"],
            "average_latency_ms": metrics["average_latency_ms"],
            "status_warning": self.last_status_warning
        }

    def get_metrics(self, source_mode: Optional[str] = None) -> Dict[str, Any]:
        mode_filter = source_mode.upper() if source_mode else self.active_mode
        return self.db.compute_metrics(session_id=self.session_id, threshold=self.decision_threshold, source_mode=mode_filter)

    def get_recent_predictions(self, limit: int = 25, source_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        mode_filter = source_mode.upper() if source_mode else self.active_mode
        return self.db.get_recent_predictions(limit=limit, session_id=self.session_id, source_mode=mode_filter)

    def export_csv(self, source_mode: Optional[str] = None) -> str:
        return self.db.export_csv_string(session_id=self.session_id, source_mode=source_mode)
