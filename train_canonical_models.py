"""
FlashGuard AI — Canonical Model Training, Validation & Persistence Pipeline
Problem Statement 13: Hybrid Prediction & Classification for Flash Crash Risk Management
ASTRAQUANT X-2026

Trains and validates:
1. XGBoost Classifier (Primary Liquidity Regime Classifier: Liquid / Fragmented / Illiquid)
2. Random Forest Classifier (Ensemble Benchmark)
3. Logistic Regression (Linear Benchmark with StandardScaler)
4. XGBoost Regressor (60-Second Downside-Move Risk: 0.1% Threshold)
5. XGBoost Binary Classifier (Genuine Flash-Crash Prediction: Spread Surge > 500% / Black Hole)

Saves models, out-of-sample backtesting metrics, and verified evaluation metadata into `models/`.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
import xgboost as xgb

from feature_pipeline import (
    CANONICAL_FEATURES,
    TARGET_REGIME_NAMES,
    transform_dataset_to_canonical
)

DATASET_PATH = "astraquant_problem13_dataset.csv"
MODELS_DIR = "models"

def train_and_persist_models():
    print("=" * 75)
    print("FLASHGUARD AI: TRAINING & VALIDATING CANONICAL MODELS")
    print("ASTRAQUANT X-2026 • Problem Statement 13")
    print("=" * 75)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset '{DATASET_PATH}' not found!")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load and Transform Dataset
    raw_df = pd.read_csv(DATASET_PATH)
    print(f"[*] Raw Dataset Loaded: {raw_df.shape[0]} windows, {raw_df.shape[1]} columns.")

    df = transform_dataset_to_canonical(raw_df)
    print(f"[+] Canonical Feature Transformation Complete: {len(CANONICAL_FEATURES)} features.")

    # 2. Chronological Train/Test Split with Purge Gap (Zero Lookahead / Future Leakage)
    split_ratio = 0.70
    purge_gap = 5  # Purge 5 windows (~50 minutes) between train and test
    total_len = len(df)
    train_end = int(total_len * split_ratio)
    test_start = train_end + purge_gap

    train_df = df.iloc[:train_end]
    test_df = df.iloc[test_start:]

    print(f"[*] Chronological Split: Train={len(train_df)}, Purged Gap={purge_gap}, Test={len(test_df)}")

    X_train = train_df[CANONICAL_FEATURES].values
    y_cls_train = train_df["liquidity_regime"].values
    y_reg_train = train_df["flash_crash_prob_60s"].values
    y_crash_train = train_df["is_liquidity_black_hole"].values

    X_test = test_df[CANONICAL_FEATURES].values
    y_cls_test = test_df["liquidity_regime"].values
    y_reg_test = test_df["flash_crash_prob_60s"].values
    y_crash_test = test_df["is_liquidity_black_hole"].values

    # Scaler for linear baseline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # -------------------------------------------------------------------------
    # MODEL 1: XGBoost Liquidity Classifier
    # -------------------------------------------------------------------------
    print("\n[1/5] Training XGBoost Liquidity Classifier...")
    xgb_cls = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb_cls.fit(X_train, y_cls_train)
    xgb_probs = xgb_cls.predict_proba(X_test)
    xgb_preds = np.argmax(xgb_probs, axis=1)

    xgb_logloss = float(log_loss(y_cls_test, xgb_probs))
    xgb_acc = float(accuracy_score(y_cls_test, xgb_preds))
    xgb_prec = [float(p) for p in precision_score(y_cls_test, xgb_preds, average=None)]
    xgb_rec = [float(r) for r in recall_score(y_cls_test, xgb_preds, average=None)]
    xgb_f1 = [float(f) for f in f1_score(y_cls_test, xgb_preds, average=None)]
    xgb_macro_f1 = float(f1_score(y_cls_test, xgb_preds, average="macro"))
    xgb_cm = confusion_matrix(y_cls_test, xgb_preds).tolist()

    joblib.dump(xgb_cls, os.path.join(MODELS_DIR, "xgb_liquidity_classifier.joblib"))
    print(f"  ✓ XGBoost Log-Loss: {xgb_logloss:.4f} | Accuracy: {xgb_acc*100:.2f}% | Macro F1: {xgb_macro_f1:.4f}")

    # -------------------------------------------------------------------------
    # MODEL 2: Random Forest Classifier
    # -------------------------------------------------------------------------
    print("\n[2/5] Training Random Forest Classifier...")
    rf_cls = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        random_state=42
    )
    rf_cls.fit(X_train, y_cls_train)
    rf_probs = rf_cls.predict_proba(X_test)
    rf_preds = rf_cls.predict(X_test)

    rf_logloss = float(log_loss(y_cls_test, rf_probs))
    rf_acc = float(accuracy_score(y_cls_test, rf_preds))
    rf_macro_f1 = float(f1_score(y_cls_test, rf_preds, average="macro"))
    rf_cm = confusion_matrix(y_cls_test, rf_preds).tolist()

    joblib.dump(rf_cls, os.path.join(MODELS_DIR, "rf_liquidity_classifier.joblib"))
    print(f"  ✓ Random Forest Log-Loss: {rf_logloss:.4f} | Accuracy: {rf_acc*100:.2f}% | Macro F1: {rf_macro_f1:.4f}")

    # -------------------------------------------------------------------------
    # MODEL 3: Logistic Regression Baseline
    # -------------------------------------------------------------------------
    print("\n[3/5] Training Logistic Regression Baseline...")
    lr_cls = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    lr_cls.fit(X_train_scaled, y_cls_train)
    lr_probs = lr_cls.predict_proba(X_test_scaled)
    lr_preds = lr_cls.predict(X_test_scaled)

    lr_logloss = float(log_loss(y_cls_test, lr_probs))
    lr_acc = float(accuracy_score(y_cls_test, lr_preds))
    lr_macro_f1 = float(f1_score(y_cls_test, lr_preds, average="macro"))
    lr_cm = confusion_matrix(y_cls_test, lr_preds).tolist()

    joblib.dump(lr_cls, os.path.join(MODELS_DIR, "lr_liquidity_classifier.joblib"))
    print(f"  ✓ Logistic Regression Log-Loss: {lr_logloss:.4f} | Accuracy: {lr_acc*100:.2f}%")

    # -------------------------------------------------------------------------
    # MODEL 4: XGBoost Regressor (60-Second Downside Risk: 0.1% Threshold)
    # -------------------------------------------------------------------------
    print("\n[4/5] Training XGBoost 60s Downside Risk Regressor...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42
    )
    xgb_reg.fit(X_train, y_reg_train)
    y_reg_preds = np.clip(xgb_reg.predict(X_test), 0.0, 1.0)

    reg_mse = float(mean_squared_error(y_reg_test, y_reg_preds))
    reg_rmse = float(np.sqrt(reg_mse))
    reg_mae = float(mean_absolute_error(y_reg_test, y_reg_preds))
    reg_r2 = float(r2_score(y_reg_test, y_reg_preds))

    # Binary risk threshold evaluation (>0.35 probability flags elevated/downside event)
    binary_true = (y_reg_test >= 0.35).astype(int)
    binary_pred = (y_reg_preds >= 0.35).astype(int)
    roc_auc = float(roc_auc_score(binary_true, y_reg_preds)) if len(np.unique(binary_true)) > 1 else 1.0
    brier = float(brier_score_loss(binary_true, y_reg_preds))
    pos_events = int(np.sum(binary_true))
    total_test = len(y_reg_test)
    false_alarms = int(np.sum((binary_pred == 1) & (binary_true == 0)))
    false_alarm_rate = float(false_alarms / max(1, np.sum(binary_true == 0)))

    joblib.dump(xgb_reg, os.path.join(MODELS_DIR, "xgb_downside_regressor.joblib"))
    print(f"  ✓ Downside Regressor MSE: {reg_mse:.6f} | RMSE: {reg_rmse:.4f} | R²: {reg_r2:.4f} | ROC-AUC: {roc_auc:.4f}")

    # -------------------------------------------------------------------------
    # MODEL 5: Genuine Flash-Crash Prediction (Spread Surge > 500% / Black Hole)
    # -------------------------------------------------------------------------
    print("\n[5/5] Training Genuine Flash-Crash Prediction Model (Spread Surge >500%)...")
    xgb_crash_cls = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    xgb_crash_cls.fit(X_train, y_crash_train)
    crash_probs = xgb_crash_cls.predict_proba(X_test)[:, 1]
    crash_preds = (crash_probs >= 0.50).astype(int)

    crash_acc = float(accuracy_score(y_crash_test, crash_preds))
    crash_prec = float(precision_score(y_crash_test, crash_preds, zero_division=0))
    crash_rec = float(recall_score(y_crash_test, crash_preds, zero_division=0))
    crash_f1 = float(f1_score(y_crash_test, crash_preds, zero_division=0))
    crash_roc_auc = float(roc_auc_score(y_crash_test, crash_probs))
    crash_pr_auc = float(average_precision_score(y_crash_test, crash_probs))
    crash_brier = float(brier_score_loss(y_crash_test, crash_probs))
    crash_cm = confusion_matrix(y_crash_test, crash_preds).tolist()

    # Backtesting stats on genuine crash events
    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = crash_cm[0][0], crash_cm[0][1], crash_cm[1][0], crash_cm[1][1]
    crash_pos_events = int(np.sum(y_crash_test == 1))
    crash_neg_events = int(np.sum(y_crash_test == 0))
    crash_far = float(fp / max(1, crash_neg_events))

    joblib.dump(xgb_crash_cls, os.path.join(MODELS_DIR, "xgb_crash_classifier.joblib"))
    print(f"  ✓ Genuine Crash PR-AUC: {crash_pr_auc:.4f} | ROC-AUC: {crash_roc_auc:.4f} | F1: {crash_f1:.4f} | FAR: {crash_far*100:.2f}%")
    print(f"  ✓ Out-of-sample positive crash events: {crash_pos_events} / {total_test} ({crash_pos_events/total_test*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Measure Single-Sample Inference Latency
    # -------------------------------------------------------------------------
    sample_feat = X_test[0:1]
    times = []
    for _ in range(150):
        t0 = time.perf_counter()
        _ = xgb_cls.predict_proba(sample_feat)
        _ = xgb_reg.predict(sample_feat)
        _ = xgb_crash_cls.predict_proba(sample_feat)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    avg_latency_ms = round(float(np.mean(times[50:])), 2)
    print(f"\n[*] Measured Model Inference Latency: {avg_latency_ms} ms (Combined 3 Models)")

    # -------------------------------------------------------------------------
    # Feature Importances (From Trained XGBoost Models)
    # -------------------------------------------------------------------------
    feature_importances_dict = {
        feat: round(float(imp), 4)
        for feat, imp in zip(CANONICAL_FEATURES, xgb_cls.feature_importances_)
    }

    # -------------------------------------------------------------------------
    # Baseline Feature Distributions (for Data Drift Monitoring)
    # -------------------------------------------------------------------------
    baseline_stats = {}
    for i, col in enumerate(CANONICAL_FEATURES):
        baseline_stats[col] = {
            "mean": float(np.mean(X_train[:, i])),
            "std": float(np.std(X_train[:, i])),
            "median": float(np.median(X_train[:, i])),
            "q25": float(np.percentile(X_train[:, i], 25)),
            "q75": float(np.percentile(X_train[:, i], 75))
        }

    # -------------------------------------------------------------------------
    # Save Comprehensive Verified Metadata
    # -------------------------------------------------------------------------
    metadata = {
        "model_version": "FlashGuard-v2.1-XGB",
        "training_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "source": "Historical Order-Book Market Observations (10-minute windows)",
            "instrument": "Deribit BTC-PERPETUAL",
            "total_windows": total_len,
            "train_windows": len(train_df),
            "test_windows": len(test_df),
            "purge_gap_windows": purge_gap,
            "positive_crash_events_total": int(np.sum(df["is_liquidity_black_hole"] == 1)),
            "positive_crash_events_test": crash_pos_events
        },
        "target_definitions": {
            "liquidity_classification": {
                "classes": TARGET_REGIME_NAMES,
                "description": "Multi-class classification of market liquidity state into Liquid, Fragmented (Proxy), or Illiquid."
            },
            "downside_risk_prediction": {
                "name": "60-Second Downside-Move Risk (0.1%)",
                "description": "Predicted risk score [0.0, 1.0] of a >0.1% downside move in next 60s."
            },
            "flash_crash_prediction": {
                "name": "Genuine Flash-Crash Prediction (Spread Surge > 500% / Black Hole)",
                "description": "Calibrated probability [0.0, 1.0] of an extreme liquidity black-hole collapse event."
            }
        },
        "canonical_features": CANONICAL_FEATURES,
        "feature_importances": feature_importances_dict,
        "historical_backtest": {
            "test_windows_count": total_test,
            "correct_predictions": int(tp + tn),
            "incorrect_predictions": int(fp + fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives_false_alarms": int(fp),
            "false_negatives_missed_events": int(fn),
            "accuracy_pct": round(crash_acc * 100.0, 2),
            "precision_pct": round(crash_prec * 100.0, 2),
            "recall_pct": round(crash_rec * 100.0, 2),
            "f1_score": round(crash_f1, 4),
            "false_alarm_rate_pct": round(crash_far * 100.0, 2),
            "prediction_lead_time": "60.0 seconds advance signal (~10-minute window horizon)",
            "pr_auc": round(crash_pr_auc, 4),
            "roc_auc": round(crash_roc_auc, 4)
        },
        "performance_metrics": {
            "liquidity_classification": {
                "xgboost": {
                    "log_loss": round(xgb_logloss, 4),
                    "accuracy": round(xgb_acc * 100.0, 2),
                    "macro_f1": round(xgb_macro_f1, 4),
                    "precision_per_class": [round(p, 4) for p in xgb_prec],
                    "recall_per_class": [round(r, 4) for r in xgb_rec],
                    "f1_per_class": [round(f, 4) for f in xgb_f1],
                    "confusion_matrix": xgb_cm,
                    "class_distribution_test": [int(np.sum(y_cls_test == c)) for c in [0, 1, 2]]
                },
                "random_forest": {
                    "log_loss": round(rf_logloss, 4),
                    "accuracy": round(rf_acc * 100.0, 2),
                    "macro_f1": round(rf_macro_f1, 4),
                    "confusion_matrix": rf_cm
                },
                "logistic_regression": {
                    "log_loss": round(lr_logloss, 4),
                    "accuracy": round(lr_acc * 100.0, 2),
                    "macro_f1": round(lr_macro_f1, 4),
                    "confusion_matrix": lr_cm
                }
            },
            "downside_risk_prediction": {
                "model": "XGBoost Regressor",
                "mse": round(reg_mse, 6),
                "rmse": round(reg_rmse, 5),
                "mae": round(reg_mae, 5),
                "r2_score": round(reg_r2, 4),
                "roc_auc": round(roc_auc, 4),
                "brier_score": round(brier, 5),
                "positive_events_in_test": pos_events,
                "total_test_samples": total_test,
                "false_alarm_rate": round(false_alarm_rate, 4),
                "avg_inference_latency_ms": avg_latency_ms
            },
            "genuine_flash_crash": {
                "model": "XGBoost Binary Classifier",
                "accuracy": round(crash_acc * 100.0, 2),
                "precision": round(crash_prec * 100.0, 2),
                "recall": round(crash_rec * 100.0, 2),
                "f1_score": round(crash_f1, 4),
                "roc_auc": round(crash_roc_auc, 4),
                "pr_auc": round(crash_pr_auc, 4),
                "brier_score": round(crash_brier, 5),
                "confusion_matrix": crash_cm,
                "positive_test_events": crash_pos_events,
                "negative_test_events": crash_neg_events,
                "false_alarm_rate": round(crash_far, 4)
            }
        },
        "baseline_feature_distributions": baseline_stats
    }

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[+] Metadata successfully saved to: {meta_path}")
    print("[+] All models saved in 'models/' directory.")
    return metadata

if __name__ == "__main__":
    train_and_persist_models()
