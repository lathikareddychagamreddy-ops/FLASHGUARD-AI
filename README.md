# ASTRAQUANT X-2026: Problem Statement 13
## Hybrid Prediction and Classification for Flash Crash Risk Management

### 1. Problem Overview
- **Goal:** Dual-output AI model to:
  1. Classify market liquidity into **Liquid**, **Fragmented**, or **Illiquid** (Evaluated by **Log-Loss**).
  2. Predict the probability of a flash crash within the next **60 seconds** (Evaluated by **MSE**).
- **Core Objectives:**
  - 10-minute market windows.
  - Predict liquidity black-hole events (spread increase $> 500\%$).
  - Adapt to dynamic market drift / regime shifts.

---

### 2. Ready-to-Use Files Created in Workspace
1. **`astraquant_problem13_dataset.csv`**: Ready-to-train dataset with 1,500 samples covering:
   - Microstructure features: `bid_ask_spread`, `spread_expansion_pct`, `order_to_cancel_ratio`, `volume_imbalance`, `order_flow_imbalance`, `book_depth_top5`, `realized_volatility`, `trade_intensity_10m`.
   - Dual targets: `liquidity_regime` (0, 1, 2) and `flash_crash_prob_60s` (0.0 to 1.0).
   - Dynamic data drift injection: `drift_regime_factor`.
2. **`generate_astraquant_dataset.py`**: Full Python generator script to scale up to any number of windows.
3. **`hybrid_model_baseline.py`**: Complete baseline training pipeline implementing the dual-output architecture, evaluating Log-Loss and MSE, and outputting predictions.

---

### 3. Open Benchmark Datasets (If using real market data)
- **Kaggle Optiver Realized Volatility Prediction**: 10-minute order book snapshots (`book_train.parquet`).
- **LOBSTER (lobsterdata.com)**: Free sample LOB messages & book depth for AAPL/MSFT, includes explicit order submissions and cancellations for Order-to-Cancel ratios.
- **FI-2010 Benchmark Dataset**: Standard academic Limit Order Book dataset.
- **Binance Public Data (`data.binance.vision`)**: High-frequency order book and trade data with real market regime shifts and flash crashes.
