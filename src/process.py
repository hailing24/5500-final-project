"""
Preprocessing functions
"""

import pandas as pd

def load_data(path):
    """Load raw NOAA data."""
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df

def clean_data(df):
    """Keep essential columns."""
    keep = ["STATION", "DATE", "PRCP", "TMAX", "TMIN"]
    return df[keep].copy()

def aggregate_monthly(df):
    """Aggregate daily observations to monthly per station."""
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

    # Month anchor for plotting
    monthly["ym"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )

    return monthly
