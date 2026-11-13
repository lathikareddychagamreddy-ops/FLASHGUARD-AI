"""
FlashGuard AI — Unified Feature Engineering Pipeline
Problem Statement 13: Hybrid Prediction & Classification for Flash Crash Risk Management
ASTRAQUANT X-2026

This module provides a unified, deterministic feature extraction pipeline used
by BOTH historical model training/backtesting and live real-time inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# Canonical feature ordering expected by all trained models
CANONICAL_FEATURES = [
    "bid_ask_spread",               # Absolute spread (USD): best_ask - best_bid
    "relative_spread_bps",          # Relative spread (basis points): (spread / mid) * 10,000
    "spread_expansion_pct",         # % change vs. rolling median baseline spread
    "book_depth_top5",              # Cumulative quantity in top-5 bid + top-5 ask levels
    "volume_imbalance",             # (bid_vol - ask_vol) / (bid_vol + ask_vol), bounded [-1, 1]
    "liquidity_withdrawal_rate",    # Rate of depth reductions over rolling window, bounded [0, 1]
    "realized_volatility_10m",      # Standard deviation of mid-price returns over window
    "trade_intensity_10m",          # Trade count / tick activity per window
    "cancel_pressure",              # withdrawal_rate * abs(volume_imbalance)
    "volatility_spread_interaction" # realized_volatility * (spread_expansion_pct / 100)
]

TARGET_REGIME_NAMES = ["Liquid", "Fragmented (Depth Skew Proxy)", "Illiquid"]
MIN_WARMUP_TICKS = 20  # Minimum ticks required for robust rolling feature calibration


class RollingMicrostructureBuffer:
    """
    Maintains a rolling window of recent order book snapshots
    to compute rolling volatility, spread expansion, and liquidity withdrawal rate.
    """
    def __init__(self, maxlen: int = 120):
        self.maxlen = maxlen
        self.mid_prices: List[float] = []
        self.spreads: List[float] = []
        self.depths: List[float] = []
        self.timestamps: List[int] = []

    def add_tick(self, mid_price: float, spread: float, total_depth: float, timestamp_ms: int):
        self.mid_prices.append(float(mid_price))
        self.spreads.append(float(spread))
        self.depths.append(float(total_depth))
        self.timestamps.append(int(timestamp_ms))

        if len(self.mid_prices) > self.maxlen:
            self.mid_prices.pop(0)
            self.spreads.pop(0)
            self.depths.pop(0)
            self.timestamps.pop(0)

    @property
    def tick_count(self) -> int:
        return len(self.mid_prices)

    def is_warmed_up(self) -> bool:
        return len(self.mid_prices) >= MIN_WARMUP_TICKS

    def calculate_rolling_metrics(self, current_spread: float, current_depth: float) -> Tuple[float, float, float]:
        """
        Computes:
        1. spread_expansion_pct: % spread expansion against rolling median spread
        2. liquidity_withdrawal_rate: normalized rate of cumulative depth depletion
        3. realized_volatility_10m: standard deviation of percentage returns
        """
        # Baseline spread
        if len(self.spreads) >= 5:
            median_spread = float(np.median(self.spreads))
            spread_expansion_pct = ((current_spread - median_spread) / max(0.01, median_spread)) * 100.0
        else:
            spread_expansion_pct = 0.0

        # Liquidity Withdrawal Rate Proxy from Level-2 depth reductions
        # Measured via net depth depletion relative to recent maximum depth and negative flow dominance
        if len(self.depths) >= 4:
            recent_depths = np.array(self.depths[-20:])
            max_depth = max(100.0, float(np.max(recent_depths)))
            current_d = float(recent_depths[-1])
            depth_drawdown = max(0.0, (max_depth - current_d) / max_depth)

            depth_diffs = np.diff(recent_depths)
            neg_shifts = float(np.sum(np.maximum(0.0, -depth_diffs)))
            total_flux = float(np.sum(np.abs(depth_diffs))) + 1e-5
            neg_dominance = max(0.0, (neg_shifts / total_flux) * 2.0 - 1.0)

            raw_withdrawal = 0.65 * depth_drawdown + 0.35 * neg_dominance
            withdrawal_rate = float(np.clip(raw_withdrawal, 0.02, 0.98))
        else:
            # Clean initial baseline: un-warmed buffer starts at nominal liquid baseline (0.05), never 100%
            withdrawal_rate = 0.05

        # Realized Volatility
        if len(self.mid_prices) >= 4:
            prices = np.array(self.mid_prices)
            returns = np.diff(prices) / np.maximum(1e-4, prices[:-1])
            vol = float(np.std(returns))
        else:
            vol = 0.008

        return (
            round(float(spread_expansion_pct), 2),
            round(float(withdrawal_rate), 4),
            round(float(vol), 6)
        )


def compute_features_from_snapshot(
    best_bid: float,
    best_ask: float,
    top_bids: List[List[float]],
    top_asks: List[List[float]],
    rolling_buffer: Optional[RollingMicrostructureBuffer] = None,
    trade_intensity_10m: int = 1500,
    forced_spread_expansion: Optional[float] = None,
    forced_withdrawal_rate: Optional[float] = None,
    forced_volatility: Optional[float] = None
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
    """
    Computes canonical microstructure features from a single order book snapshot.
    Returns:
      1. feature_dict: dictionary of feature names to values
      2. feature_vector: 1D numpy array aligned with CANONICAL_FEATURES
      3. metadata: status information (warmup, depth totals, relative spread)
    """
    best_bid = float(best_bid)
    best_ask = float(best_ask)
    mid_price = (best_bid + best_ask) / 2.0
    spread = max(0.01, best_ask - best_bid)
    relative_spread_bps = (spread / max(1.0, mid_price)) * 10000.0

    # Top-5 Book Depth & Imbalance
    bid_vols = [float(b[1]) for b in top_bids[:5]] if top_bids else [5000.0]
    ask_vols = [float(a[1]) for a in top_asks[:5]] if top_asks else [5000.0]
    bid_depth = sum(bid_vols)
    ask_depth = sum(ask_vols)
    total_depth = max(10.0, bid_depth + ask_depth)

    volume_imbalance = (bid_depth - ask_depth) / total_depth
    volume_imbalance = float(np.clip(volume_imbalance, -1.0, 1.0))

    # Rolling or overridden metrics
    if rolling_buffer is not None:
        calc_expansion, calc_withdrawal, calc_vol = rolling_buffer.calculate_rolling_metrics(spread, total_depth)
        spread_expansion_pct = forced_spread_expansion if forced_spread_expansion is not None else calc_expansion
        withdrawal_rate = forced_withdrawal_rate if forced_withdrawal_rate is not None else calc_withdrawal
        realized_vol = forced_volatility if forced_volatility is not None else calc_vol
        is_warmed_up = rolling_buffer.is_warmed_up()
        current_ticks = rolling_buffer.tick_count
    else:
        spread_expansion_pct = forced_spread_expansion if forced_spread_expansion is not None else 0.0
        withdrawal_rate = forced_withdrawal_rate if forced_withdrawal_rate is not None else 0.08
        realized_vol = forced_volatility if forced_volatility is not None else 0.008
        is_warmed_up = True
        current_ticks = MIN_WARMUP_TICKS

    # Interaction terms
    cancel_pressure = float(np.clip(withdrawal_rate * abs(volume_imbalance), 0.0, 1.0))
    vol_spread_interaction = float(realized_vol * (spread_expansion_pct / 100.0))

    feature_dict = {
        "bid_ask_spread": round(spread, 4),
        "relative_spread_bps": round(relative_spread_bps, 2),
        "spread_expansion_pct": round(spread_expansion_pct, 2),
        "book_depth_top5": round(total_depth, 1),
        "volume_imbalance": round(volume_imbalance, 4),
        "liquidity_withdrawal_rate": round(withdrawal_rate, 4),
        "realized_volatility_10m": round(realized_vol, 6),
        "trade_intensity_10m": int(trade_intensity_10m),
        "cancel_pressure": round(cancel_pressure, 4),
        "volatility_spread_interaction": round(vol_spread_interaction, 6)
    }

    feature_vector = np.array([feature_dict[col] for col in CANONICAL_FEATURES], dtype=np.float32)

    metadata = {
        "mid_price": round(mid_price, 2),
        "best_bid": round(best_bid, 2),
        "best_ask": round(best_ask, 2),
        "is_warmed_up": is_warmed_up,
        "warmup_progress": f"{min(current_ticks, MIN_WARMUP_TICKS)}/{MIN_WARMUP_TICKS}",
        "warmup_status": "Ready" if is_warmed_up else f"Collecting data ({current_ticks}/{MIN_WARMUP_TICKS} ticks)",
        "feature_count": len(CANONICAL_FEATURES)
    }

    return feature_dict, feature_vector, metadata


def transform_dataset_to_canonical(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms historical training CSV data into the unified canonical feature format.
    Ensures identical feature names, definitions, and ordering.
    """
    df = df_in.copy()

    # Relative spread in basis points
    if "relative_spread_bps" not in df.columns:
        if "mid_price" in df.columns and "bid_ask_spread" in df.columns:
            df["relative_spread_bps"] = (df["bid_ask_spread"] / np.maximum(1.0, df["mid_price"])) * 10000.0
        else:
            df["relative_spread_bps"] = 1.0

    # Map legacy O2C to documented Liquidity Withdrawal Rate
    if "liquidity_withdrawal_rate" not in df.columns:
        if "order_to_cancel_ratio" in df.columns:
            # Calibrated monotonic mapping from cancellation intensity to depth withdrawal rate [0, 1]
            o2c = df["order_to_cancel_ratio"]
            df["liquidity_withdrawal_rate"] = np.clip((o2c - 1.0) / 20.0, 0.02, 0.98)
        else:
            df["liquidity_withdrawal_rate"] = 0.08

    # Ensure realized volatility column name alignment
    if "realized_volatility_10m" not in df.columns:
        if "realized_volatility" in df.columns:
            df["realized_volatility_10m"] = df["realized_volatility"]
        else:
            df["realized_volatility_10m"] = 0.008

    # Interaction terms
    df["cancel_pressure"] = df["liquidity_withdrawal_rate"] * np.abs(df["volume_imbalance"])
    df["volatility_spread_interaction"] = df["realized_volatility_10m"] * (df["spread_expansion_pct"] / 100.0)

    # Check for missing columns
    for col in CANONICAL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Required canonical feature '{col}' missing from dataset!")

    return df
