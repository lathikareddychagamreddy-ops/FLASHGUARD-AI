"""
ASTRAQUANT X-2026: Problem Statement 13
Dataset Generator & Preprocessing for Hybrid Flash Crash Risk Management

This script simulates high-frequency Limit Order Book (LOB) dynamics with:
1. 10-minute feature aggregation windows
2. Realistic Order-to-Cancel (O2C) ratios & Order Flow/Volume Imbalances (OFI)
3. Dynamic market regimes (Liquid, Fragmented, Illiquid) with concept/data drift
4. Liquidity black-hole events (spread surges > 500%) and flash crashes
5. Dual targets:
   - Target 1 (Categorical): liquidity_regime (0: Liquid, 1: Fragmented, 2: Illiquid)
   - Target 2 (Continuous): flash_crash_prob_60s (0.0 to 1.0)
"""

import numpy as np
import pandas as pd
import random
import os

def generate_astraquant_dataset(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    records = []
    
    # Base simulation state
    price = 100.0
    drift_factor = 1.0
    
    # We will simulate 10-minute window summaries (or tick-aggregated windows)
    for i in range(n_samples):
        # Simulate market regime changes (Markov-like drift)
        # 0: Liquid (70%), 1: Fragmented (20%), 2: Illiquid (10%)
        # Drift increases over time to simulate dynamic data drift
        if i > int(n_samples * 0.6):
            drift_factor = 1.8 # High volatility regime shift / market crisis
            regime_probs = [0.40, 0.35, 0.25]
        else:
            regime_probs = [0.70, 0.20, 0.10]
            
        regime = np.random.choice([0, 1, 2], p=regime_probs)
        regime_label = ["Liquid", "Fragmented", "Illiquid"][regime]
        
        # Properties based on regime
        if regime == 0:  # Liquid
            base_spread = np.random.uniform(0.01, 0.05) * drift_factor
            o2c_ratio = np.random.uniform(1.2, 3.0) # healthy order to cancel ratio
            vol_imbalance = np.random.normal(0.0, 0.15)
            depth_top5 = np.random.uniform(5000, 15000)
            realized_vol = np.random.uniform(0.005, 0.015) * drift_factor
            trade_count = np.random.randint(1000, 3000)
            is_black_hole = 0
            spread_expansion_pct = np.random.uniform(-10.0, 50.0)
            flash_crash_prob = np.clip(np.random.beta(1, 20), 0.0, 0.15)
            
        elif regime == 1:  # Fragmented
            base_spread = np.random.uniform(0.06, 0.18) * drift_factor
            o2c_ratio = np.random.uniform(3.0, 8.0) # High cancellations / spoofing signs
            vol_imbalance = np.random.normal(0.0, 0.35)
            depth_top5 = np.random.uniform(1500, 5000)
            realized_vol = np.random.uniform(0.015, 0.040) * drift_factor
            trade_count = np.random.randint(400, 1200)
            is_black_hole = 1 if np.random.rand() < 0.15 else 0
            spread_expansion_pct = np.random.uniform(50.0, 300.0)
            flash_crash_prob = np.clip(np.random.beta(2, 8), 0.15, 0.60)
            
        else:  # Illiquid
            base_spread = np.random.uniform(0.20, 0.80) * drift_factor
            o2c_ratio = np.random.uniform(8.0, 25.0) # extreme cancellations, liquidity flight
            vol_imbalance = np.random.uniform(-0.85, 0.85) # massive one-sided imbalance
            depth_top5 = np.random.uniform(100, 1500)
            realized_vol = np.random.uniform(0.040, 0.120) * drift_factor
            trade_count = np.random.randint(50, 400)
            # High probability of liquidity black-hole (>500% spread surge)
            is_black_hole = 1 if np.random.rand() < 0.50 else 0
            if is_black_hole:
                spread_expansion_pct = np.random.uniform(500.0, 1500.0) # Spread surge > 500%
                flash_crash_prob = np.clip(np.random.beta(6, 2), 0.65, 0.99)
            else:
                spread_expansion_pct = np.random.uniform(200.0, 480.0)
                flash_crash_prob = np.clip(np.random.beta(3, 4), 0.35, 0.75)

        # Microstructure feature calculations
        best_bid = price - (base_spread / 2.0)
        best_ask = price + (base_spread / 2.0)
        mid_price = (best_bid + best_ask) / 2.0
        micro_price = mid_price + (vol_imbalance * (base_spread / 2.0))
        
        # Simulating random walk for mid price
        price_return = np.random.normal(0, realized_vol)
        price = max(10.0, price * (1.0 + price_return))

        records.append({
            "window_id": i + 1,
            "timestamp_min": (i + 1) * 10,
            "mid_price": round(mid_price, 4),
            "best_bid": round(best_bid, 4),
            "best_ask": round(best_ask, 4),
            "bid_ask_spread": round(base_spread, 5),
            "spread_expansion_pct": round(spread_expansion_pct, 2), # Key: >500% indicates black hole
            "order_to_cancel_ratio": round(o2c_ratio, 3),
            "volume_imbalance": round(vol_imbalance, 4),
            "order_flow_imbalance": round(vol_imbalance * trade_count, 2),
            "book_depth_top5": round(depth_top5, 1),
            "realized_volatility": round(realized_vol, 5),
            "trade_intensity_10m": trade_count,
            "micro_price_deviation": round(micro_price - mid_price, 5),
            "drift_regime_factor": round(drift_factor, 2),
            "is_liquidity_black_hole": is_black_hole, # Spread surge > 500%
            # --- Dual Targets ---
            "liquidity_regime": regime, # 0: Liquid, 1: Fragmented, 2: Illiquid
            "liquidity_regime_label": regime_label,
            "flash_crash_prob_60s": round(flash_crash_prob, 4) # Continuous probability in next 60s
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print("Generating AstraQuant Problem Statement 13 Dataset...")
    df = generate_astraquant_dataset(n_samples=6000)
    output_path = "astraquant_problem13_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created: {output_path}")
    print(f"Shape: {df.shape}")
    print("Regime distribution:")
    print(df['liquidity_regime_label'].value_counts())
    print("Liquidity Black-Hole events (spread > 500%):", df['is_liquidity_black_hole'].sum())
