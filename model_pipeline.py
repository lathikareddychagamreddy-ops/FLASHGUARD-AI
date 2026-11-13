"""
ASTRAQUANT X-2026: Problem Statement 13
Full Production Modeling & Evaluation Pipeline

Components:
1. Feature Engineering & Feature Selection for Order-Book Microstructure
2. Liquidity Classification (Liquid / Fragmented / Illiquid):
   - XGBoost Classifier
   - Random Forest Classifier
   - Logistic Regression Baseline
3. Flash Crash Prediction (Probability within 60s):
   - XGBoost Regressor
4. Comprehensive Model Evaluation & Comparison (Logistic Regression vs. XGBoost)
   - Log-Loss, Accuracy, Precision, Recall, Macro F1, Confusion Matrix
   - Mean Squared Error (MSE), RMSE, MAE, R2 for Flash Crash
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import xgboost as xgb


def load_and_engineer_features(csv_path="astraquant_problem13_dataset.csv"):
    print(f"[*] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[*] Raw Dataset Shape: {df.shape}")

    # Microstructure Feature Engineering
    # 1. Kyle's Lambda proxy (Price impact per unit volume)
    df["spread_to_depth_ratio"] = df["bid_ask_spread"] / (df["book_depth_top5"] + 1e-5)
    
    # 2. Cancel Pressure: Interaction between cancellations and volume skew (Spoofing indicator)
    df["cancel_pressure"] = df["order_to_cancel_ratio"] * np.abs(df["volume_imbalance"])
    
    # 3. Volatility-Spread Stress Interaction
    df["volatility_spread_interaction"] = df["realized_volatility"] * (df["spread_expansion_pct"] / 100.0)
    
    # 4. Flow Toxicity / Order Intensity proxy
    df["flow_toxicity"] = df["order_flow_imbalance"] / (df["trade_intensity_10m"] + 1e-5)

    # 5. Black hole hazard score (non-linear spread surge indicator)
    df["black_hole_hazard"] = np.where(df["spread_expansion_pct"] > 500.0, 1.0, df["spread_expansion_pct"] / 500.0)

    return df


def select_features(df, target_col="liquidity_regime", k=8):
    candidate_features = [
        "bid_ask_spread",
        "spread_expansion_pct",
        "order_to_cancel_ratio",
        "volume_imbalance",
        "order_flow_imbalance",
        "book_depth_top5",
        "realized_volatility",
        "trade_intensity_10m",
        "spread_to_depth_ratio",
        "cancel_pressure",
        "volatility_spread_interaction",
        "flow_toxicity",
        "black_hole_hazard"
    ]

    X_cand = df[candidate_features]
    y = df[target_col]

    # Use ANOVA F-value & Mutual Info to score feature relevance
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_cand, y)
    
    feature_scores = pd.DataFrame({
        "Feature": candidate_features,
        "F_Score": selector.scores_,
        "P_Value": selector.pvalues_
    }).sort_values(by="F_Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE & SELECTION RANKING (ANOVA F-Score):")
    print("=" * 50)
    print(feature_scores.to_string(index=False))

    # Pick top k attributes
    selected_cols = feature_scores["Feature"].iloc[:k].tolist()
    print(f"\n[+] Selected Top {k} Features for Training:")
    for i, col in enumerate(selected_cols, 1):
        print(f"    {i}. {col}")

    return selected_cols, feature_scores


def train_and_evaluate_all():
    # 1. Load data
    df = load_and_engineer_features("astraquant_problem13_dataset.csv")

    # 2. Feature selection
    selected_features, feature_scores = select_features(df, target_col="liquidity_regime", k=9)

    X = df[selected_features].values
    y_cls = df["liquidity_regime"].values
    y_reg = df["flash_crash_prob_60s"].values

    # 3. Time-Series Preserving Split (Train 70%, Test 30% containing dynamic regime drift)
    split_idx = int(len(df) * 0.70)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

    print(f"\nTrain set: {len(X_train)} windows | Test set (Market Drift Evaluation): {len(X_test)} windows")

    # Scaler for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================================
    # TASK 1: LIQUIDITY CLASSIFICATION (Liquid=0, Fragmented=1, Illiquid=2)
    # Models: XGBoost, Random Forest, Logistic Regression
    # Metric: Log-Loss, Accuracy, Precision, Recall, F1
    # =========================================================================
    print("\n" + "=" * 60)
    print("TASK 1: LIQUIDITY CLASSIFICATION (XGBoost vs. Random Forest vs. LogReg)")
    print("=" * 60)

    # 1A. XGBoost Classifier
    print("[*] Training XGBoost Classifier...")
    xgb_cls = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
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

    xgb_logloss = log_loss(y_cls_test, xgb_probs)
    xgb_acc = accuracy_score(y_cls_test, xgb_preds)
    xgb_f1 = f1_score(y_cls_test, xgb_preds, average="macro")

    # 1B. Random Forest Classifier
    print("[*] Training Random Forest Classifier...")
    rf_cls = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        random_state=42
    )
    rf_cls.fit(X_train, y_cls_train)
    rf_probs = rf_cls.predict_proba(X_test)
    rf_preds = rf_cls.predict(X_test)

    rf_logloss = log_loss(y_cls_test, rf_probs)
    rf_acc = accuracy_score(y_cls_test, rf_preds)
    rf_f1 = f1_score(y_cls_test, rf_preds, average="macro")

    # 1C. Logistic Regression Baseline
    print("[*] Training Logistic Regression Baseline...")
    lr_cls = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    lr_cls.fit(X_train_scaled, y_cls_train)
    lr_probs = lr_cls.predict_proba(X_test_scaled)
    lr_preds = lr_cls.predict(X_test_scaled)

    lr_logloss = log_loss(y_cls_test, lr_probs)
    lr_acc = accuracy_score(y_cls_test, lr_preds)
    lr_f1 = f1_score(y_cls_test, lr_preds, average="macro")

    # Comparison Table
    cls_summary = pd.DataFrame([
        {
            "Model": "Logistic Regression (Linear Baseline)",
            "Log-Loss (Lower=Better)": round(lr_logloss, 4),
            "Accuracy": f"{lr_acc * 100:.2f}%",
            "Macro F1": round(lr_f1, 4),
            "Illiquid Recall": round(recall_score(y_cls_test, lr_preds, average=None)[2], 4)
        },
        {
            "Model": "Random Forest Classifier",
            "Log-Loss (Lower=Better)": round(rf_logloss, 4),
            "Accuracy": f"{rf_acc * 100:.2f}%",
            "Macro F1": round(rf_f1, 4),
            "Illiquid Recall": round(recall_score(y_cls_test, rf_preds, average=None)[2], 4)
        },
        {
            "Model": "XGBoost Classifier",
            "Log-Loss (Lower=Better)": round(xgb_logloss, 4),
            "Accuracy": f"{xgb_acc * 100:.2f}%",
            "Macro F1": round(xgb_f1, 4),
            "Illiquid Recall": round(recall_score(y_cls_test, xgb_preds, average=None)[2], 4)
        }
    ])

    print("\n--- LIQUIDITY CLASSIFICATION BENCHMARK ---")
    print(cls_summary.to_string(index=False))

    # =========================================================================
    # TASK 2: FLASH CRASH PREDICTION (Continuous Probability in Next 60s)
    # Model: XGBoost Regressor
    # Metric: Mean Squared Error (MSE), RMSE, MAE, R2
    # =========================================================================
    print("\n" + "=" * 60)
    print("TASK 2: FLASH CRASH RISK PREDICTION WITHIN 60 SECONDS (XGBoost Regressor)")
    print("=" * 60)

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

    reg_mse = mean_squared_error(y_reg_test, y_reg_preds)
    reg_rmse = np.sqrt(reg_mse)
    reg_mae = mean_absolute_error(y_reg_test, y_reg_preds)
    reg_r2 = r2_score(y_reg_test, y_reg_preds)

    print(f"[+] Flash Crash Probability Prediction (60s horizon):")
    print(f"    - Mean Squared Error (MSE, Competition Metric): {reg_mse:.6f}")
    print(f"    - Root Mean Squared Error (RMSE):               {reg_rmse:.6f}")
    print(f"    - Mean Absolute Error (MAE):                    {reg_mae:.6f}")
    print(f"    - R-squared (R2 Score):                         {reg_r2:.4f}")

    # =========================================================================
    # TASK 3: DETAILED MODEL EVALUATION: LOGISTIC REGRESSION VS. XGBOOST
    # =========================================================================
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON & EVALUATION: LOGISTIC REGRESSION VS. XGBOOST")
    print("=" * 60)

    target_labels = ["Liquid", "Fragmented", "Illiquid"]

    print("\n[A] Logistic Regression Classification Report:")
    print(classification_report(y_cls_test, lr_preds, target_names=target_labels, digits=4))

    print("\n[B] XGBoost Classification Report:")
    print(classification_report(y_cls_test, xgb_preds, target_names=target_labels, digits=4))

    # Confusion Matrices
    lr_cm = confusion_matrix(y_cls_test, lr_preds)
    xgb_cm = confusion_matrix(y_cls_test, xgb_preds)

    print("[C] Confusion Matrix - Logistic Regression:")
    print(pd.DataFrame(lr_cm, index=[f"True {c}" for c in target_labels], columns=[f"Pred {c}" for c in target_labels]))

    print("\n[D] Confusion Matrix - XGBoost:")
    print(pd.DataFrame(xgb_cm, index=[f"True {c}" for c in target_labels], columns=[f"Pred {c}" for c in target_labels]))

    # Export Test Predictions
    eval_df = pd.DataFrame({
        "window_id": df["window_id"].iloc[split_idx:].values,
        "true_liquidity_regime": [target_labels[i] for i in y_cls_test],
        "logreg_pred_regime": [target_labels[i] for i in lr_preds],
        "rf_pred_regime": [target_labels[i] for i in rf_preds],
        "xgb_pred_regime": [target_labels[i] for i in xgb_preds],
        "true_flash_crash_prob": y_reg_test,
        "xgb_pred_flash_crash_prob": np.round(y_reg_preds, 4),
        "is_liquidity_black_hole": df["is_liquidity_black_hole"].iloc[split_idx:].values
    })
    eval_df.to_csv("model_evaluation_results.csv", index=False)
    print("\n[+] Full test evaluation results exported to 'model_evaluation_results.csv'.")

    # Generate Performance & Comparison Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Feature Importances (XGBoost Classifier)
    feat_imp = pd.Series(xgb_cls.feature_importances_, index=selected_features).sort_values()
    feat_imp.plot(kind="barh", ax=axes[0, 0], color="#2b5c8f")
    axes[0, 0].set_title("XGBoost Liquidity Classification - Feature Importances", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Relative Importance (Gain)")

    # Plot 2: Confusion Matrix - Logistic Regression
    sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 1], xticklabels=target_labels, yticklabels=target_labels)
    axes[0, 1].set_title(f"Logistic Regression Confusion Matrix (Log-Loss: {lr_logloss:.4f})", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("True Regime")
    axes[0, 1].set_xlabel("Predicted Regime")

    # Plot 3: Confusion Matrix - XGBoost
    sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1, 0], xticklabels=target_labels, yticklabels=target_labels)
    axes[1, 0].set_title(f"XGBoost Confusion Matrix (Log-Loss: {xgb_logloss:.4f})", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("True Regime")
    axes[1, 0].set_xlabel("Predicted Regime")

    # Plot 4: Flash Crash Risk: Predicted vs True Probabilities
    axes[1, 1].scatter(y_reg_test, y_reg_preds, alpha=0.4, color="#d95f02", edgecolors="none", s=25)
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=2, label="Perfect Calibration")
    axes[1, 1].set_title(f"XGBoost Flash Crash Prediction (MSE: {reg_mse:.5f}, R2: {reg_r2:.3f})", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("True Flash Crash Probability (60s)")
    axes[1, 1].set_ylabel("Predicted Flash Crash Probability")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("astraquant_model_evaluation.png", dpi=200)
    print("[+] Visualization plot saved to 'astraquant_model_evaluation.png'.")

    return {
        "lr_logloss": lr_logloss,
        "rf_logloss": rf_logloss,
        "xgb_logloss": xgb_logloss,
        "lr_acc": lr_acc,
        "rf_acc": rf_acc,
        "xgb_acc": xgb_acc,
        "reg_mse": reg_mse,
        "reg_r2": reg_r2
    }


if __name__ == "__main__":
    train_and_evaluate_all()
