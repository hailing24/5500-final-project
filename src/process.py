"""
Preprocessing functions for NOAA climate data.
"""

import os
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load raw NOAA data from CSV."""
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep essential columns."""
    keep = ["STATION", "DATE", "PRCP", "TMAX", "TMIN"]
    return df[keep].copy()


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily observations to monthly level per station.
    Creates:
        - PRCP_sum: monthly total precipitation
        - TMAX_mean / TMIN_mean: monthly mean max/min temperature
        - TMEAN: monthly mean temperature
        - ym: month anchor date (first day of month)
    """
    df = df.copy()
    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month

    monthly = (
        df.groupby(["STATION", "year", "month"])
        .agg(
            PRCP_sum=("PRCP", "sum"),
            TMAX_mean=("TMAX", "mean"),
            TMIN_mean=("TMIN", "mean"),
        )
        .reset_index()
    )

    monthly["TMEAN"] = monthly[["TMAX_mean", "TMIN_mean"]].mean(axis=1)

    # Month anchor
    monthly["ym"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )

    return monthly


def add_lags(
    df: pd.DataFrame,
    target_col: str = "TMEAN",
    lags: int = 1,
    group_col: str = "STATION",
) -> pd.DataFrame:
    """
    Add lagged versions of the target column per station.
    """
    df = df.sort_values([group_col, "year", "month"]).copy()
    for lag in range(1, lags + 1):
        df[f"{target_col}_lag{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def process_monthly_data(
    raw_path: str,
    monthly_output_path: str | None = None,
    lags: int = 1,
) -> pd.DataFrame:

    # Load and clean
    raw = load_data(raw_path)
    clean = clean_data(raw)

    # Aggregate to monthly
    monthly = aggregate_monthly(clean)

    #  Save monthly data
    if monthly_output_path is not None:
        os.makedirs(os.path.dirname(monthly_output_path), exist_ok=True)
        monthly.to_csv(monthly_output_path, index=False)

    # Add lag features for modeling
    monthly_with_lags = add_lags(monthly, target_col="TMEAN", lags=lags)

    return monthly_with_lags
