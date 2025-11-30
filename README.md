# Climate Forecasting

End‑to‑end pipeline that converts raw NOAA daily observations into monthly aggregates, engineers features, fits next‑month temperature/precipitation regressors, and classifies relative (May–Sep) heat extremes. The repository contains both a reproducible CLI entry point (`main.py`) and exploratory notebooks (most notably `notebooks/modeling.ipynb`).

---

## Data & Sources

- **Input**: `data/rawdata.csv`, concatenated NOAA Integrated Surface Dataset extracts (multiple stations from 1996–2024).
- **Intermediate**: `data/monthly_data.csv`, rebuilt every run from the raw file (see `src/process.py`).
- **Outputs** (written by `main.py`):
  - `outputs/predictions.csv` – test-set predictions for each station/month.
  - `outputs/forecast_next_month.csv` – one-step-ahead forecast per station using the latest month of data.
  - `outputs/extreme_heat_classification.csv` – logistic regression probabilities/labels for May–Sep relative extremes (test years only).

---

## Repository Layout

```
5550-final-project/
├─ data/                         # Raw + monthly aggregated datasets
├─ notebooks/
│  └─ modeling.ipynb             # Mirrors the CLI workflow with extra analysis/plots
├─ outputs/                      # Predictions, forecasts, classification tables
├─ src/
│  ├─ process.py                 # Daily to monthly aggregation utilities
│  ├─ model.py                   # Regression + classification pipeline
├─ main.py                       # CLI entry point (rebuilds data, trains models, saves outputs)
└─ README.md
```

---

## Environment

- Python 3.10+
- Key libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- Install with `pip install -r requirements.txt` (or mirror the Anaconda env used in development).

---

## Pipeline Overview

1. **Preprocessing (`src/process.py`)**
   - Parse daily NOAA csv, coerce numeric fields, derive `year`, `month`, and `TMEAN`.
   - Aggregate to station × year × month totals/means.
   - Persist to `data/monthly_data.csv`.

2. **Feature Engineering (`src/model.py`)**
   - Rolling precip/temperature stats, sine/cosine seasonal encodings, multiple lags (1/3/6/12 months), anomalies, and log precipitation.
   - Relative extreme labels: station–month 90th-percentile threshold computed using pre-2021 data only.
   - Targets: next-month TMEAN/TMAX/TMIN/PRCP; classification target is next-month relative heat flag.

3. **Regression models**
   - Train/test split by year (`train < 2021`, `test ≥ 2021`).
   - Models: Linear Regression, Ridge Regression, Random Forest (300 estimators).
   - Metrics per target: RMSE / MAE / R². Best model chosen by average RMSE across four targets.

4. **Classification (May–Sep)**
   - Balanced Logistic Regression (liblinear) with standardized features.
   - Reports ROC AUC, F1 at default 0.5 threshold, and F1 at an optimized probability threshold (maximizing PR curve F1).
   - Saves per-sample probabilities + predictions.

5. **Forecasting**
   - Last observation per station is advanced one month to create inputs for a one-step forecast.
   - Uses whichever regressor had the lowest average RMSE.

6. **Carbon Estimate**
   - Simple runtime-based electricity/carbon estimate printed after each CLI run.

---

## Usage

```bash
python main.py
```

The script:

1. Rebuilds `data/monthly_data.csv` from daily raw data.
2. Trains regressors + classifier.
3. Writes the three csv outputs listed above.
4. Prints evaluation tables similar to the modeling notebook (regression metrics per target, classification reports/confusion matrices, best threshold, carbon estimate).

To inspect or visualize interactively, open `notebooks/modeling.ipynb`. The notebook reproduces the CLI outputs and adds:

- Gradient Boosting baseline (tuned) comparisons.
- Time-series cross-validation snippets.
- Permutation importance for TMEAN_next.
- ROC curves and per-station diagnostic plots.

---

## Interpretation Guide

- **Regression metrics**: TMEAN/TMAX/TMIN RMSE ≈ 3 °F with R² ≈ 0.97 (random forest). PRCP is inherently noisier (RMSE ≈ 2 in, R² ≈ 0.54).
- **Classification**: ROC AUC ≈ 0.65 for 2021–2024 May–Sep test months. Default threshold maximizes recall for rare heat events; tuned threshold balances F1 around 0.50.
- **Best model**: Random Forest almost always wins; Ridge provides a linear-but-regularized reference; Gradient Boosting illustrates a third approach.

---

