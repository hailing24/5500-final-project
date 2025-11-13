"""
Modeling functions

This module provides:
- add_lags(): create lag features for monthly climate data
- train_and_predict(): train simple regression models per station
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def add_lags(monthly):
    """
    Add lag features (lag1_TMEAN, lag1_PRCP) for monthly climate data.
    Each station is handled separately.
    """

    def _add_lags(g):
        g = g.sort_values("ym").copy()
        g["lag1_TMEAN"] = g["TMEAN"].shift(1)
        g["lag1_PRCP"] = g["PRCP_sum"].shift(1)
        return g

    return monthly.groupby("STATION", group_keys=False).apply(_add_lags)


def train_and_predict(df):
    """
    Train simple Linear Regression models for each station.
    Predict:
    - Monthly temperature (TMEAN)
    - Monthly precipitation (PRCP_sum)

    Last 12 months serve as a test set.

    Returns:
    - A combined DataFrame with predictions.
    """

    results = []

    for station, g in df.groupby("STATION"):

        g = g.sort_values("ym").copy()

        # Drop rows with missing required values (fixes NaN error)
        g = g.dropna(subset=["TMEAN", "PRCP_sum", "lag1_TMEAN", "lag1_PRCP"]).copy()

        if g.empty:
            continue

        # Last 12 months = test split
        split_date = g["ym"].max() - pd.DateOffset(months=12)
        g["split"] = np.where(g["ym"] > split_date, "test", "train")

        # ----------------------
        # Temperature model
        # ----------------------
        feat_temp = ["year", "month", "lag1_TMEAN", "lag1_PRCP"]

        model_temp = LinearRegression()
        model_temp.fit(
            g.loc[g["split"] == "train", feat_temp],
            g.loc[g["split"] == "train", "TMEAN"]
        )

        g["pred_TMEAN"] = model_temp.predict(g[feat_temp])

        # ----------------------
        # Precipitation model
        # ----------------------
        feat_prcp = ["year", "month", "lag1_PRCP"]

        model_prcp = LinearRegression()
        model_prcp.fit(
            g.loc[g["split"] == "train", feat_prcp],
            g.loc[g["split"] == "train", "PRCP_sum"]
        )

        g["pred_PRCP"] = model_prcp.predict(g[feat_prcp])

        results.append(g)

    # Combine all station results
    return pd.concat(results, ignore_index=True)
