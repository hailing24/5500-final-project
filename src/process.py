"""
Data preprocessing helpers for turning raw NOAA daily observations into the
monthly dataset consumed by the modeling pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# All numeric measurement fields expected in NOAA GHCN-D data
NUMERIC_COLUMNS: Iterable[str] = (
    "PRCP",
    "SNOW",
    "SNWD",
    "TAVG",
    "TMAX",
    "TMIN",
)


def _load_raw_data(raw_path: str | Path) -> pd.DataFrame:
    """
    Load the raw CSV exactly as provided.
    This function stays intentionally simple so that all downstream cleaning
    occurs in dedicated preprocessing steps.
    """
    raw_df = pd.read_csv(raw_path)
    return raw_df


def _prepare_daily_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the daily NOAA GHCN-D format:

    - Convert numeric columns to floats (coercing invalid entries to NaN).
    - Convert DATE to datetime.
    - Extract year and month for later grouping.
    - Construct daily mean temperature (TMEAN_daily). If TAVG is missing,
      approximate it using (TMAX + TMIN) / 2 — a standard NOAA fallback.
    """
    df = raw_df.copy()

    # Ensure consistent numeric types for all measurement columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamp parsing and basic temporal decomposition
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month

    # Construct daily mean temperature
    df["TMEAN_daily"] = df["TAVG"]
    missing_tavg = df["TMEAN_daily"].isna()

    # Fallback for missing TAVG
    df.loc[missing_tavg, "TMEAN_daily"] = df.loc[missing_tavg, ["TMAX", "TMIN"]].mean(
        axis=1
    )

    return df


def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily data into monthly summaries per station:

    - Sum daily precipitation
    - Take monthly means for TMAX, TMIN, and mean temperature
    - Create a monthly timestamp (ym) for downstream time-series handling

    This produces the exact monthly-level inputs required by model.py.
    """
    monthly = (
        df.groupby(["STATION", "year", "month"], as_index=False)
        .agg(
            PRCP_sum=("PRCP", "sum"),
            TMAX_mean=("TMAX", "mean"),
            TMIN_mean=("TMIN", "mean"),
            TMEAN=("TMEAN_daily", "mean"),
        )
        .sort_values(["STATION", "year", "month"])
        .reset_index(drop=True)
    )

    # Create a monthly timestamp for consistency with the notebook version
    monthly["ym"] = pd.to_datetime(
        {
            "year": monthly["year"].astype(int),
            "month": monthly["month"].astype(int),
            "day": 1,
        }
    )

    return monthly


def process_monthly_data(
    raw_path: str | Path,
    monthly_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:

    1. Read the raw NOAA daily CSV
    2. Convert daily observations into a cleaned daily frame
    3. Aggregate to monthly climate features
    4. Optionally save the monthly output

    The resulting dataframe matches the `monthly` dataframe used in the
    modeling.ipynb notebook before feature engineering.
    """
    raw_df = _load_raw_data(raw_path)
    daily_df = _prepare_daily_frame(raw_df)
    monthly = _aggregate_monthly(daily_df)

    # Persist monthly CSV for reproducibility and debugging
    if monthly_output_path is not None:
        output_path = Path(monthly_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(output_path, index=False)

    return monthly
