"""
Data preprocessing helpers for turning raw NOAA daily observations into the
monthly dataset consumed by the modeling pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

NUMERIC_COLUMNS: Iterable[str] = (
    "PRCP",
    "SNOW",
    "SNWD",
    "TAVG",
    "TMAX",
    "TMIN",
)


def _load_raw_data(raw_path: str | Path) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_path)
    return raw_df


def _prepare_daily_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["TMEAN_daily"] = df["TAVG"]
    missing_tavg = df["TMEAN_daily"].isna()
    df.loc[missing_tavg, "TMEAN_daily"] = df.loc[missing_tavg, ["TMAX", "TMIN"]].mean(
        axis=1
    )
    return df


def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
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
    Convert daily raw measurements into monthly aggregates persisted at
    `monthly_output_path`. The returned dataframe matches the notebook's
    `monthly` dataframe prior to feature engineering.
    """
    raw_df = _load_raw_data(raw_path)
    daily_df = _prepare_daily_frame(raw_df)
    monthly = _aggregate_monthly(daily_df)

    if monthly_output_path is not None:
        output_path = Path(monthly_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(output_path, index=False)

    return monthly
