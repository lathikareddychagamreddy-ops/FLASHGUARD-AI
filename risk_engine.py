"""
FlashGuard AI — Early-Warning Risk Desk, Alert History & Data Drift Monitor
Problem Statement 13: Hybrid Prediction & Classification for Flash Crash Risk Management
ASTRAQUANT X-2026

Features:
1. Early-Warning Engine: Eliminates contradictory messages; dispatches NORMAL, ELEVATED, HIGH RISK.
2. Alert History Manager: In-memory rolling event ledger with CSV export capabilities.
3. Data Drift Monitor: Compares feature distributions against training baseline with exact drift states:
   - "No significant drift"
   - "Potential drift detected"
   - "Retraining recommended"
"""

import os
import io
import csv
import json
import time
import datetime
import numpy as np
from typing import Dict, Any, List, Optional

MODELS_DIR = "models"
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


class EarlyWarningEngine:
    """
    Early-Warning System providing rule-based and model-driven risk alerts.
    Guarantees non-contradictory warning states and trigger audits.
    """
    SPREAD_EXP_ELEVATED = 120.0  # %
    SPREAD_EXP_HIGH = 350.0      # %
    WITHDRAWAL_ELEVATED = 0.35   # [0, 1]
    WITHDRAWAL_HIGH = 0.55       # [0, 1]
    RISK_PROB_ELEVATED = 0.35    # [0, 1]
    RISK_PROB_HIGH = 0.65        # [0, 1]
    DEPTH_DEPLETED = 8000.0      # Contracts

    @classmethod
    def evaluate(
        cls,
        features: Dict[str, float],
        downside_risk_prob: float,
        flash_crash_prob: float = 0.0,
        is_simulated: bool = False
    ) -> Dict[str, Any]:
        spread_exp = float(features.get("spread_expansion_pct", 0.0))
        withdrawal = float(features.get("liquidity_withdrawal_rate", 0.0))
        depth = float(features.get("book_depth_top5", 100000.0))
        vol = float(features.get("realized_volatility_10m", 0.008))
        spread_val = float(features.get("bid_ask_spread", 0.5))

        triggers = []
        severity_score = 0  # 0: normal, 1: elevated, 2: high risk

        # 1. Spread Expansion Check
        if spread_exp >= cls.SPREAD_EXP_HIGH or spread_exp >= 500.0:
            triggers.append(f"Severe spread expansion (+{spread_exp:.1f}% vs baseline)")
            severity_score = max(severity_score, 2)
        elif spread_exp >= cls.SPREAD_EXP_ELEVATED:
            triggers.append(f"Moderate spread widening (+{spread_exp:.1f}%)")
            severity_score = max(severity_score, 1)

        # 2. Liquidity Withdrawal Rate Check
        if withdrawal >= cls.WITHDRAWAL_HIGH:
            triggers.append(f"Aggressive depth depletion (Withdrawal Rate: {withdrawal:.2f})")
            severity_score = max(severity_score, 2)
        elif withdrawal >= cls.WITHDRAWAL_ELEVATED:
            triggers.append(f"Elevated quote withdrawal (Rate: {withdrawal:.2f})")
            severity_score = max(severity_score, 1)

        # 3. Book Depth Depletion
        if depth < cls.DEPTH_DEPLETED:
            triggers.append(f"Displayed book depth thin ({depth:,.0f} contracts)")
            severity_score = max(severity_score, 1)

        # 4. Model Predicted 60s Downside Risk
        if downside_risk_prob >= cls.RISK_PROB_HIGH:
            triggers.append(f"Model predicts high downside risk ({downside_risk_prob*100:.1f}%)")
            severity_score = max(severity_score, 2)
        elif downside_risk_prob >= cls.RISK_PROB_ELEVATED:
            triggers.append(f"Model predicts elevated downside risk ({downside_risk_prob*100:.1f}%)")
            severity_score = max(severity_score, 1)

        # 5. Genuine Flash Crash Probability Check
        if flash_crash_prob >= 0.50:
            triggers.append(f"High flash-crash event probability ({flash_crash_prob*100:.1f}%)")
            severity_score = max(severity_score, 2)

        # Determine Non-Contradictory State and Title
        if severity_score == 2:
            state = "HIGH RISK"
            title = "EARLY-WARNING RISK DESK: CRITICAL LIQUIDITY COLLAPSE HAZARD"
            action = "Recommend restricting new aggressive orders; liquidity black-hole conditions detected."
            badge_class = "badge-illiquid"
            box_class = "alert-critical"
        elif severity_score == 1:
            state = "ELEVATED"
            title = "EARLY-WARNING RISK DESK: ELEVATED LIQUIDITY OR DOWNSIDE RISK"
            action = "Caution advised; widen execution limits and monitor order-book replenishment."
            badge_class = "badge-fragmented"
            box_class = "alert-warning"
        else:
            state = "NORMAL"
            title = "EARLY-WARNING RISK DESK: TRADING CONDITIONS BENIGN & STABLE"
            action = "Trading conditions appear stable; order book is resilient."
            badge_class = "badge-liquid"
            box_class = "alert-normal"
            triggers.append("No active liquidity depletion or spread anomalies detected.")

        return {
            "state": state,
            "warning_level": state,
            "banner_title": title,
            "badge_class": badge_class,
            "box_class": box_class,
            "severity_score": severity_score,
            "is_simulated": is_simulated,
            "triggers": triggers,
            "trigger_reasons": triggers,
            "recommendation": action,
            "recommended_action": action,
            "action_required": (severity_score > 0),
            "disclaimer": "Advisory decision support only — does NOT execute live market orders."
        }


class AlertHistoryManager:
    """
    Maintains chronological ledgers of generated alerts strictly partitioned by operating mode:
    LIVE, HISTORICAL_REPLAY, and SIMULATION.
    Guarantees that artificial simulation stress alerts never leak into live or replay audit tables.
    """
    def __init__(self, maxlen: int = 250):
        self.maxlen = maxlen
        self.mode_alerts: Dict[str, List[Dict[str, Any]]] = {
            "LIVE": [],
            "HISTORICAL_REPLAY": [],
            "SIMULATION": []
        }
        self.alerts: List[Dict[str, Any]] = []

    def log_alert(
        self,
        warning_level: str,
        liquidity_regime: str,
        downside_risk_pct: float,
        flash_crash_prob_pct: float,
        triggers: List[str],
        mode: str,
        timestamp_str: Optional[str] = None
    ):
        norm_mode = mode.upper() if mode else "LIVE"
        if norm_mode not in self.mode_alerts:
            self.mode_alerts[norm_mode] = []

        if not timestamp_str:
            timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Avoid redundant consecutive identical entries within 3 seconds in the same mode
        active_list = self.mode_alerts[norm_mode]
        if active_list and active_list[-1]["warning_level"] == warning_level:
            last_t = active_list[-1].get("_epoch", 0)
            if time.time() - last_t < 4.0:
                return

        alert_entry = {
            "id": len(self.alerts) + 1,
            "_epoch": time.time(),
            "timestamp": timestamp_str,
            "warning_level": warning_level,
            "liquidity_regime": liquidity_regime,
            "downside_risk_pct": round(downside_risk_pct, 1),
            "flash_crash_prob_pct": round(flash_crash_prob_pct, 1),
            "primary_reason": triggers[0] if triggers else "Standard risk monitoring tick",
            "all_reasons": "; ".join(triggers),
            "mode": norm_mode
        }

        # Append to mode-isolated ledger
        active_list.append(alert_entry)
        if len(active_list) > self.maxlen:
            active_list.pop(0)

        # Also append to global ledger
        self.alerts.append(alert_entry)
        if len(self.alerts) > self.maxlen * 3:
            self.alerts.pop(0)

    def get_recent_alerts(self, limit: int = 50, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns recent alerts strictly isolated to the requested mode.
        If mode is omitted, falls back to the active mode or global ledger.
        """
        if mode:
            norm_mode = mode.upper()
            target_list = self.mode_alerts.get(norm_mode, [])
            return list(reversed(target_list[-limit:]))
        return list(reversed(self.alerts[-limit:]))

    def clear_mode_alerts(self, mode: Optional[str] = None):
        if mode:
            norm_mode = mode.upper()
            self.mode_alerts[norm_mode] = []
            self.alerts = [a for a in self.alerts if a.get("mode") != norm_mode]
        else:
            for k in self.mode_alerts:
                self.mode_alerts[k] = []
            self.alerts = []

    def export_csv_string(self, mode: Optional[str] = None) -> str:
        output = io.StringIO()
        fieldnames = [
            "id", "timestamp", "mode", "warning_level",
            "liquidity_regime", "downside_risk_pct",
            "flash_crash_prob_pct", "primary_reason", "all_reasons"
        ]
        header_mapping = {
            "id": "Alert ID",
            "timestamp": "Timestamp (UTC)",
            "mode": "Market Mode",
            "warning_level": "Warning Level",
            "liquidity_regime": "Liquidity Regime",
            "downside_risk_pct": "60s Downside Risk %",
            "flash_crash_prob_pct": "Genuine Flash Crash %",
            "primary_reason": "Primary Reason",
            "all_reasons": "All Audit Triggers"
        }
        writer = csv.writer(output)
        writer.writerow([header_mapping[f] for f in fieldnames])
        source_list = self.get_recent_alerts(limit=500, mode=mode)
        for alert in source_list:
            writer.writerow([alert.get(f, "") for f in fieldnames])
        return output.getvalue()


class DataDriftMonitor:
    """
    Tracks distribution shifts in market features relative to the training baseline.
    Computes normalized divergence metrics and flags retraining recommendations.
    Strictly ignores artificial simulation observations to prevent baseline poisoning.
    """
    def __init__(self, history_maxlen: int = 80):
        self.history_maxlen = history_maxlen
        self.feature_history: Dict[str, List[float]] = {}
        self.baseline_stats: Dict[str, Any] = {}
        self.model_version = "FlashGuard-v2.1-XGB"
        self.last_training_time = "2026-11-13T00:00:00Z"
        self.load_baseline()

    def load_baseline(self):
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.baseline_stats = meta.get("baseline_feature_distributions", {})
                    self.model_version = meta.get("model_version", self.model_version)
                    self.last_training_time = meta.get("training_timestamp_utc", self.last_training_time)
            except Exception as e:
                print(f"[!] Warning reading drift baseline metadata: {e}")

    def add_observation(self, feature_dict: Dict[str, float], is_simulated: bool = False):
        # Strict isolation: Never record synthetic simulation features in drift tracking
        if is_simulated:
            return
        for key, val in feature_dict.items():
            if key not in self.feature_history:
                self.feature_history[key] = []
            self.feature_history[key].append(float(val))
            if len(self.feature_history[key]) > self.history_maxlen:
                self.feature_history[key].pop(0)

    def evaluate_drift(self) -> Dict[str, Any]:
        """
        Computes statistical divergence (Normalized Mean Difference)
        for primary microstructure features.
        """
        monitored_features = [
            "bid_ask_spread",
            "book_depth_top5",
            "volume_imbalance",
            "liquidity_withdrawal_rate",
            "realized_volatility_10m",
            "trade_intensity_10m"
        ]

        feature_drift_reports = {}
        total_divergence = 0.0
        feature_count = 0

        for feat in monitored_features:
            if feat not in self.feature_history or len(self.feature_history[feat]) < 8:
                feature_drift_reports[feat] = {
                    "divergence_score": 0.05,
                    "z_score": 0.05,
                    "status": "No significant drift"
                }
                continue

            current_vals = np.array(self.feature_history[feat])
            curr_mean = float(np.mean(current_vals))
            curr_std = float(np.std(current_vals))

            base = self.baseline_stats.get(feat, {"mean": curr_mean, "std": max(1e-4, curr_std)})
            base_mean = float(base["mean"])
            base_std = max(1e-4, float(base["std"]))

            # Standardized Mean Difference (Cohen's d proxy)
            norm_diff = abs(curr_mean - base_mean) / base_std
            norm_diff_clipped = float(np.clip(norm_diff, 0.0, 3.0))

            divergence_score = round(norm_diff_clipped / 3.0, 3)
            total_divergence += divergence_score
            feature_count += 1

            if divergence_score >= 0.45:
                status_label = "Retraining recommended"
            elif divergence_score >= 0.25:
                status_label = "Potential drift detected"
            else:
                status_label = "No significant drift"

            feature_drift_reports[feat] = {
                "current_mean": round(curr_mean, 4),
                "baseline_mean": round(base_mean, 4),
                "divergence_score": divergence_score,
                "z_score": round(norm_diff, 2),
                "status": status_label
            }

        avg_drift = (total_divergence / max(1, feature_count)) if feature_count > 0 else 0.05
        avg_drift = round(avg_drift, 3)

        if avg_drift >= 0.45:
            drift_status = "Retraining recommended"
            badge = "status-stale"
            rec_message = "Advisory: Market feature distributions have shifted significantly from training baseline. Model retraining recommended."
            retrain_rec = True
        elif avg_drift >= 0.25:
            drift_status = "Potential drift detected"
            badge = "mode-sim"
            rec_message = "Advisory: Minor distribution shift detected; continue active monitoring."
            retrain_rec = False
        else:
            drift_status = "No significant drift"
            badge = "status-healthy"
            rec_message = "Advisory: Market feature distributions align with training dataset baseline."
            retrain_rec = False

        return {
            "overall_drift_status": drift_status,
            "overall_status": drift_status,
            "badge_class": badge,
            "mean_drift_score": avg_drift,
            "model_version": self.model_version,
            "last_training_utc": self.last_training_time,
            "retraining_recommended": retrain_rec,
            "advisory_message": rec_message,
            "interpretation": rec_message,
            "metrics": feature_drift_reports,
            "features_monitored": feature_drift_reports
        }


# Global Singletons
alert_manager = AlertHistoryManager(maxlen=250)
drift_monitor = DataDriftMonitor(history_maxlen=80)
