"""
Model training utilities that follow the exact workflow in notebooks/modeling.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Use 1996–2020 as training years and 2021+ as held-out test years.
TRAIN_THRESHOLD = 2021

# 90th percentile for defining “relative heat extremes”
REL_EXTREME_Q = 0.90

# Physical temperature thresholds used to augment features
HEAT_THRESHOLD_C = 32.0
COLD_THRESHOLD_C = 0.0

# Full feature list (mirrors the notebook)
FEATURE_COLUMNS = [
    "PRCP_sum",
    "TMAX_mean",
    "TMIN_mean",
    "TMEAN",
    "temp_range",
    "prcp_roll3_mean",
    "prcp_roll6_mean",
    "tmean_roll3_mean",
    "tmean_roll6_std",
    "month_sin",
    "month_cos",
    "TMEAN_lag1",
    "TMEAN_lag3",
    "TMEAN_lag6",
    "TMEAN_lag12",
    "PRCP_lag1",
    "PRCP_lag3",
    "PRCP_lag12",
    "TMEAN_roll3",
    "TMEAN_roll6",
    "heat_extreme",
    "cold_extreme",
    "PRCP_anom",
    "PRCP_to_month_mean",
    "TMEAN_station_norm",
    "PRCP_station_norm",
    "PRCP_log",
]

# Four regression targets (predict next month's climate variables)
REGRESSION_TARGETS = ["TMEAN_next", "TMAX_next", "TMIN_next", "PRCP_next"]
PRCP_TARGET = "PRCP_next"


@dataclass
class ModelResult:
    """
    Container for all outputs produced by train_and_predict().
    """
    model_used: str
    predictions: pd.DataFrame
    metrics: Dict[str, Dict[str, Dict[str, float]]]
    forecast: Optional[pd.DataFrame] = None
    classification_metrics: Optional[Dict[str, float]] = None
    classification_predictions: Optional[pd.DataFrame] = None



### ------------------------
### Utility transforms
### ------------------------

def _log1p_prcp(values: pd.Series) -> pd.Series:
    # Log-transform to stabilize variance in precipitation targets
    return np.log1p(values.clip(lower=0))


def _inv_log1p_prcp(values: pd.Series) -> pd.Series:
    # Inverse transform back to physical precipitation units
    return np.expm1(values)


### ------------------------
### Feature engineering
### ------------------------

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all rolling statistics, lags, anomalies, and periodic features.
    This replicates exactly what was done in the notebook.
    """
    df = df.sort_values(["STATION", "year", "month"]).copy()
    df["month"] = df["month"].astype(int)
    station_group = df.groupby("STATION")

    # Basic physical features
    df["temp_range"] = df["TMAX_mean"] - df["TMIN_mean"]

    # Rolling means and variability
    df["prcp_roll3_mean"] = station_group["PRCP_sum"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    df["prcp_roll6_mean"] = station_group["PRCP_sum"].transform(
        lambda s: s.rolling(window=6, min_periods=1).mean()
    )
    df["tmean_roll3_mean"] = station_group["TMEAN"].transform(
        lambda s: s.rolling(window=3, min_periods=2).mean()
    )
    df["tmean_roll6_std"] = station_group["TMEAN"].transform(
        lambda s: s.rolling(window=6, min_periods=2).std()
    )

    # Seasonal periodicity
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Temperature & precipitation lags
    for lag in (1, 3, 6, 12):
        df[f"TMEAN_lag{lag}"] = station_group["TMEAN"].shift(lag)
    for lag in (1, 3, 12):
        df[f"PRCP_lag{lag}"] = station_group["PRCP_sum"].shift(lag)

    # Rolling averages for additional smoothing
    df["TMEAN_roll3"] = station_group["TMEAN"].transform(
        lambda s: s.rolling(window=3, min_periods=2).mean()
    )
    df["TMEAN_roll6"] = station_group["TMEAN"].transform(
        lambda s: s.rolling(window=6, min_periods=3).mean()
    )

    # Simple absolute heat/cold extremes
    df["heat_extreme"] = (df["TMAX_mean"] >= HEAT_THRESHOLD_C).astype(int)
    df["cold_extreme"] = (df["TMIN_mean"] <= COLD_THRESHOLD_C).astype(int)

    # Monthly precipitation anomalies
    monthly_station_mean = df.groupby(["STATION", "month"])["PRCP_sum"].transform("mean")
    df["PRCP_anom"] = df["PRCP_sum"] - monthly_station_mean
    df["PRCP_to_month_mean"] = df["PRCP_sum"] / (monthly_station_mean + 1e-3)

    # Station-normalized features
    df["TMEAN_station_norm"] = df["TMEAN"] - station_group["TMEAN"].transform("mean")
    df["PRCP_station_norm"] = df["PRCP_sum"] - station_group["PRCP_sum"].transform("mean")

    # Log-transform precipitation
    df["PRCP_log"] = np.log1p(df["PRCP_sum"].clip(lower=0))

    return df


def _fill_station_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill within each station to ensure continuity of lag features.
    """
    return (
        df.groupby("STATION", group_keys=False)
        .apply(lambda grp: grp.ffill())
        .reset_index(drop=True)
    )


def _prepare_feature_frame(
    monthly_df: pd.DataFrame, train_threshold: int
) -> pd.DataFrame:
    """
    Apply feature engineering, drop invalid rows, and compute
    relative heat extremes + next-month targets.
    """
    df = _add_engineered_features(monthly_df)
    df = _fill_station_missing(df)
    df = df.dropna(subset=["TMEAN", "PRCP_sum"]).reset_index(drop=True)

    # Compute station-month relative heat extremes (90th percentile within training years)
    train_mask = df["year"] < train_threshold
    quantiles = (
        df[train_mask]
        .groupby(["STATION", "month"])["TMEAN"]
        .quantile(REL_EXTREME_Q)
    )
    df = df.merge(
        quantiles.rename("tmean_quantile"), on=["STATION", "month"], how="left"
    )
    df["heat_extreme_rel"] = (df["TMEAN"] >= df["tmean_quantile"]).astype(int)
    df["heat_extreme_next_rel"] = df.groupby("STATION")["heat_extreme_rel"].shift(-1)
    df = df.drop(columns=["tmean_quantile"])

    # Next-month regression targets
    df["TMEAN_next"] = df.groupby("STATION")["TMEAN"].shift(-1)
    df["TMAX_next"] = df.groupby("STATION")["TMAX_mean"].shift(-1)
    df["TMIN_next"] = df.groupby("STATION")["TMIN_mean"].shift(-1)
    df["PRCP_next"] = df.groupby("STATION")["PRCP_sum"].shift(-1)

    return df


def _build_model_dataframe(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows that lack any required feature or target.
    """
    return feature_df.dropna(subset=FEATURE_COLUMNS + REGRESSION_TARGETS).reset_index(
        drop=True
    )


def _calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² for a single target.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _convert_predictions(pred_array: np.ndarray) -> pd.DataFrame:
    """
    Convert raw model outputs back to physical units (inverse log for PRCP).
    """
    pred_df = pd.DataFrame(pred_array, columns=REGRESSION_TARGETS)
    pred_df[PRCP_TARGET] = _inv_log1p_prcp(pred_df[PRCP_TARGET]).clip(lower=0)
    return pred_df


### ------------------------
### Build next-month forecast
### ------------------------

def _prepare_future_inputs(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the last observed month for each station and construct a valid
    feature row for predicting the next month.
    """
    last_rows = (
        feature_df.sort_values(["STATION", "year", "month"])
        .groupby("STATION")
        .tail(1)
        .reset_index(drop=True)
    )

    future_df = last_rows.copy()
    future_df["month"] += 1

    # Year-month rollover
    rollover = future_df["month"] == 13
    future_df.loc[rollover, "month"] = 1
    future_df.loc[rollover, "year"] += 1

    # Recompute seasonal encoding
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12.0)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12.0)

    # Update basic lags
    future_df["TMEAN_lag1"] = last_rows["TMEAN"]
    future_df["PRCP_lag1"] = last_rows["PRCP_sum"]

    # Carry over higher-order lags and rolling values
    for col in ["TMEAN_lag3", "TMEAN_lag6", "TMEAN_lag12", "PRCP_lag3", "PRCP_lag12"]:
        if col in future_df.columns:
            future_df[col] = last_rows[col]

    # Persistence features (unchanged for next-month inference)
    persistence_cols = [
        "temp_range",
        "prcp_roll3_mean",
        "prcp_roll6_mean",
        "tmean_roll3_mean",
        "tmean_roll6_std",
        "TMEAN_roll3",
        "TMEAN_roll6",
        "heat_extreme",
        "cold_extreme",
        "PRCP_anom",
        "PRCP_to_month_mean",
        "TMEAN_station_norm",
        "PRCP_station_norm",
    ]
    for col in persistence_cols:
        if col in future_df.columns:
            future_df[col] = last_rows[col]

    # Recompute log precipitation
    future_df["PRCP_log"] = np.log1p(future_df["PRCP_sum"].clip(lower=0))

    return future_df


### ------------------------
### Main training + prediction pipeline
### ------------------------

def train_and_predict(
    monthly_df: pd.DataFrame, train_threshold: int = TRAIN_THRESHOLD
) -> ModelResult:
    """
    Main function used by main.py.
    Performs:
        - feature engineering
        - train/test split by year
        - fit Linear Regression and Random Forest
        - compute evaluation metrics
        - fit logistic classifier for heat-extreme prediction
        - generate next-month forecast
    """
    # Prepare engineered features and usable modeling frame
    feature_df = _prepare_feature_frame(monthly_df, train_threshold)
    model_df = _build_model_dataframe(feature_df)

    # Design matrices
    X = model_df[FEATURE_COLUMNS]
    y = model_df[REGRESSION_TARGETS]
    metadata = model_df[["STATION", "year", "month"]]

    # Temporal split
    train_mask = model_df["year"] < train_threshold
    test_mask = ~train_mask
    if not test_mask.any():
        raise ValueError("No test samples available. Adjust the train_threshold.")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    meta_test = metadata[test_mask].reset_index(drop=True)

    # Log-transform precipitation target before regression
    y_train_transformed = y_train.copy()
    y_train_transformed[PRCP_TARGET] = _log1p_prcp(y_train_transformed[PRCP_TARGET])

    # Standardize features for all models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Two regression baselines
    # I kept the original linear/RF baselines and tossed in ridge for a third perspective.
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=1
        ),
    }

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    predictions_by_model: Dict[str, pd.DataFrame] = {}
    fitted_models: Dict[str, object] = {}

    # Train each model and compute per-target metrics
    for name, estimator in reg_models.items():
        estimator.fit(X_train_scaled, y_train_transformed)
        preds_df = _convert_predictions(estimator.predict(X_test_scaled))
        predictions_by_model[name] = preds_df
        fitted_models[name] = estimator

        per_target_metrics: Dict[str, Dict[str, float]] = {}
        for target in REGRESSION_TARGETS:
            per_target_metrics[target] = _calc_metrics(
                y_test[target].reset_index(drop=True), preds_df[target]
            )
        metrics[name] = per_target_metrics

    # Choose best regression model by average RMSE across four targets
    def _avg_rmse(metric_dict: Dict[str, Dict[str, float]]) -> float:
        return float(np.mean([values["rmse"] for values in metric_dict.values()]))

    best_model_name = min(metrics.keys(), key=lambda key: _avg_rmse(metrics[key]))
    best_predictions = predictions_by_model[best_model_name]

    # Attach metadata and sort for consistency
    predictions = pd.concat([meta_test, best_predictions], axis=1)
    predictions["model_used"] = best_model_name.replace(" ", "")
    predictions = predictions.sort_values(["STATION", "year", "month"]).reset_index(
        drop=True
    )

    ### --------------------------------------
    ### Logistic regression for heat extremes
    ### --------------------------------------

    classification_metrics: Optional[Dict[str, float]] = None
    classification_predictions: Optional[pd.DataFrame] = None

    # Use May–Sep + rows with valid next-month extreme label
    clf_df = model_df[
        model_df["month"].between(5, 9) & model_df["heat_extreme_next_rel"].notna()
    ].copy()

    if not clf_df.empty:
        clf_df = clf_df.dropna(
            subset=FEATURE_COLUMNS + ["heat_extreme_next_rel"]
        ).reset_index(drop=True)

    if not clf_df.empty:
        X_clf = clf_df[FEATURE_COLUMNS]
        y_clf = clf_df["heat_extreme_next_rel"].astype(int)

        clf_train_mask = clf_df["year"] < train_threshold
        clf_test_mask = ~clf_train_mask

        if clf_train_mask.any() and clf_test_mask.any():
            # Standardize independently for classifier
            clf_scaler = StandardScaler()
            X_clf_train = clf_scaler.fit_transform(X_clf[clf_train_mask])
            X_clf_test = clf_scaler.transform(X_clf[clf_test_mask])
            y_clf_train = y_clf[clf_train_mask]
            y_clf_test = y_clf[clf_test_mask]

            # Balanced logistic regression 
            log_clf = LogisticRegression(
                max_iter=1000, solver="liblinear", class_weight="balanced"
            )
            log_clf.fit(X_clf_train, y_clf_train)

            # Probability scores
            y_prob = log_clf.predict_proba(X_clf_test)[:, 1]
            default_pred = (y_prob >= 0.5).astype(int)

            # Compute F1-optimal threshold
            precision, recall, thresholds = precision_recall_curve(y_clf_test, y_prob)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            candidate_scores = f1_scores[:-1]

            if len(thresholds) == 0 or np.all(~np.isfinite(candidate_scores)):
                best_threshold = 0.5
            else:
                best_idx = int(np.nanargmax(candidate_scores))
                best_threshold = float(thresholds[best_idx])

            optimized_pred = (y_prob >= best_threshold).astype(int)

            # Store evaluation metrics
            classification_metrics = {
                "roc_auc": float(roc_auc_score(y_clf_test, y_prob)),
                "f1_default": float(f1_score(y_clf_test, default_pred, zero_division=0)),
                "precision_default": float(
                    precision_score(y_clf_test, default_pred, zero_division=0)
                ),
                "recall_default": float(
                    recall_score(y_clf_test, default_pred, zero_division=0)
                ),
                "f1_optimized": float(
                    f1_score(y_clf_test, optimized_pred, zero_division=0)
                ),
                "best_threshold": best_threshold,
            }

            # Prepare classification output rows
            clf_meta = clf_df.loc[clf_test_mask, ["STATION", "year", "month"]].reset_index(
                drop=True
            )
            classification_predictions = clf_meta.assign(
                probability=y_prob,
                pred_default=default_pred,
                pred_best=optimized_pred,
                observed=y_clf_test.reset_index(drop=True),
            )

    ### ------------------------
    ### Final next-month forecast
    ### ------------------------

    forecast_df: Optional[pd.DataFrame] = None
    future_inputs = _prepare_future_inputs(feature_df)

    if not future_inputs.empty:
        X_future = future_inputs[FEATURE_COLUMNS]
        X_future_scaled = scaler.transform(X_future)
        future_preds = _convert_predictions(
            fitted_models[best_model_name].predict(X_future_scaled)
        )

        forecast_df = pd.concat(
            [future_inputs[["STATION", "year", "month"]].reset_index(drop=True), future_preds],
            axis=1,
        )
        forecast_df["model_used"] = best_model_name.replace(" ", "")

    # Return all structured outputs
    return ModelResult(
        model_used=best_model_name.replace(" ", ""),
        predictions=predictions,
        metrics=metrics,
        forecast=forecast_df,
        classification_metrics=classification_metrics,
        classification_predictions=classification_predictions,
    )
