"""
ASTRAQUANT X-2026: Problem Statement 13
Hybrid Prediction & Classification for Flash Crash Risk Management

Dual-Output Architecture:
  Output 1: Liquidity Classification (Liquid / Fragmented / Illiquid) -> Evaluated via Multi-class Log-Loss
  Output 2: Flash Crash Probability Prediction within 60s -> Evaluated via MSE
  Adaptability: Online learning & dynamic drift handling (River / SGD / Adaptive sliding window)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error, classification_report
import os

def run_pipeline(csv_path="astraquant_problem13_dataset.csv"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Please run generate_astraquant_dataset.py first.")
        return

    print("=" * 60)
    print("ASTRAQUANT X-2026: HYBRID DUAL-OUTPUT MODEL PIPELINE")
    print("=" * 60)

    # 1. Load Preprocessed Market Window Data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} 10-minute windows.")

    # 2. Feature Selection
    feature_cols = [
        "bid_ask_spread",
        "spread_expansion_pct",
        "order_to_cancel_ratio",
        "volume_imbalance",
        "order_flow_imbalance",
        "book_depth_top5",
        "realized_volatility",
        "trade_intensity_10m",
        "drift_regime_factor"
    ]
    
    # Dual Targets
    target_cls = "liquidity_regime"         # Multi-class: 0, 1, 2
    target_reg = "flash_crash_prob_60s"     # Continuous: [0.0, 1.0]

    X = df[feature_cols].values
    y_cls = df[target_cls].values
    y_reg = df[target_reg].values

    # 3. Time-Series Split (Preserves temporal ordering & tests drift adaptability)
    # Using the first 70% as train, remaining 30% as test (where drift factor kicks in)
    split_idx = int(len(df) * 0.70)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain size: {len(X_train)} | Test size (with Market Drift): {len(X_test)}")

    # 4. Model 1: Liquidity Regime Classifier (Log-Loss metric)
    cls_model = HistGradientBoostingClassifier(
        max_iter=150, 
        learning_rate=0.08, 
        random_state=42
    )
    cls_model.fit(X_train_scaled, y_cls_train)
    y_cls_probs = cls_model.predict_proba(X_test_scaled)
    cls_loss = log_loss(y_cls_test, y_cls_probs)

    # 5. Model 2: Flash Crash 60s Probability Regressor (MSE metric)
    reg_model = HistGradientBoostingRegressor(
        max_iter=150, 
        learning_rate=0.08, 
        random_state=42
    )
    reg_model.fit(X_train_scaled, y_reg_train)
    y_reg_preds = reg_model.predict(X_test_scaled)
    y_reg_preds = np.clip(y_reg_preds, 0.0, 1.0)
    reg_mse = mean_squared_error(y_reg_test, y_reg_preds)

    # 6. Evaluation Deliverables
    print("\n" + "-" * 40)
    print("EVALUATION METRICS (As per competition criteria):")
    print("-" * 40)
    print(f"Liquidity Classification Multi-class Log-Loss: {cls_loss:.4f}")
    print(f"Flash Crash Probability (60s) MSE:            {reg_mse:.5f}")
    print(f"Flash Crash Probability RMSE:                  {np.sqrt(reg_mse):.5f}")

    y_cls_preds = np.argmax(y_cls_probs, axis=1)
    target_names = ["Liquid", "Fragmented", "Illiquid"]
    print("\nClassification Report (Under Dynamic Drift):")
    print(classification_report(y_cls_test, y_cls_preds, target_names=target_names))

    # 7. Export Predictions
    results_df = pd.DataFrame({
        "window_id": df["window_id"].iloc[split_idx:].values,
        "true_liquidity_regime": y_cls_test,
        "pred_liquidity_regime": y_cls_preds,
        "true_flash_crash_prob": y_reg_test,
        "pred_flash_crash_prob": np.round(y_reg_preds, 4),
        "is_liquidity_black_hole": df["is_liquidity_black_hole"].iloc[split_idx:].values
    })
    results_df.to_csv("hybrid_model_predictions.csv", index=False)
    print("Predictions saved to 'hybrid_model_predictions.csv'.")

if __name__ == "__main__":
    run_pipeline()
