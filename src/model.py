"""
Modeling for NOAA climate data.

Models included:
    - Linear Regression 
    - Ridge Regression 
    - Random Forest Regressor

Outputs:
    - best model predictions
    - metrics for all models
    - feature importance (RF only)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _train_one_model(model, X_train, y_train, X_test, y_test):
    """Train a single model and compute metrics."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return pred, {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_predict(df: pd.DataFrame, target: str = "TMEAN"):
    """
    Train 3 models and select the best-performing one based on RMSE.

    Returns:
        pred_df: DataFrame with test predictions
        best_model: best fitted model
        metrics_all: metrics for all models
    """

    df = df.copy()

    # Feature columns
    base_cols = ["year", "month", "PRCP_sum", "TMAX_mean", "TMIN_mean"]
    lag_cols = [c for c in df.columns if c.startswith(f"{target}_lag")]
    feature_cols = base_cols + lag_cols

    df_model = df.dropna(subset=[target] + feature_cols)
    X = df_model[feature_cols].to_numpy()
    y = df_model[target].to_numpy()

    # 80/20 chronological split
    n = len(df_model)
    split = int(n * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_test) == 0:
        raise ValueError("Not enough data for train/test split.")

    # Models to evaluate
    model_dict = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=42, max_depth=6
        )
    }

    metrics_all = {}
    preds_all = {}

    # Train each model
    for name, model in model_dict.items():
        pred, metrics = _train_one_model(model, X_train, y_train, X_test, y_test)
        metrics_all[name] = metrics
        preds_all[name] = (pred, model)

    # Select best model by RMSE
    best_name = min(metrics_all, key=lambda m: metrics_all[m]["rmse"])
    best_pred, best_model = preds_all[best_name]

    print("\n=== MODEL COMPARISON ===")
    for m, v in metrics_all.items():
        print(f"{m}: RMSE={v['rmse']:.3f}, MAE={v['mae']:.3f}, R2={v['r2']:.3f}")
    print(f"\n>> Best model selected: {best_name}")

    # Create prediction dataframe
    pred_df = df_model.iloc[split:].copy()
    pred_df[f"predicted_{target}"] = best_pred
    pred_df["model_used"] = best_name

    # Feature importance for RandomForest
    if best_name == "RandomForest":
        importances = best_model.feature_importances_
        pred_df["feature_importance"] = str(
            dict(zip(feature_cols, importances.round(4)))
        )

    return pred_df, best_model, metrics_all
