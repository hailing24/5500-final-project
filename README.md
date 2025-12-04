# Forecasting Monthly Temperature, Precipitation, and Extreme Heat Events

End‑to‑end pipeline that converts raw NOAA daily observations into monthly aggregates, engineers features, fits next‑month temperature and precipitation regressors, and classifies relative (May–Sep) heat extremes. The repository contains both a reproducible CLI entry point (`main.py`) and exploratory notebooks (most notably `notebooks/modeling.ipynb`).

---

## Problem Definition

Climate variability directly affects **infrastructure planning, agriculture, public health, and extreme-heat preparedness**.  
To support localized early-warning and resilience planning, this project investigates:

### Goals
1. **Forecast next-month temperature** (TMAX, TMIN, TMEAN)  
2. **Forecast next-month precipitation (PRCP)**  
3. **Classify relative extreme-heat months** (May–Sep), defined as:
   - `TMAX_mean_next ≥` station-specific **90th percentile threshold**  
     (computed using pre-2021 history only)

### Why it matters
Extreme-heat events are becoming **more frequent and more persistent** in major U.S. cities.  
Accurate monthly-scale predictions are critical for **operational planning**, **public-health preparedness**, and **local climate resilience**.

---

## Data & Sources

- **Input**: `data/rawdata.csv`, concatenated NOAA Integrated Surface Dataset extracts (multiple stations from 1996–2024).
- **Intermediate**: `data/monthly_data.csv`, rebuilt every run from the raw file (see `src/process.py`).
- **Outputs** (written by `main.py`):
  - `outputs/predictions.csv` – test-set predictions for each station/month.
  - `outputs/forecast_next_month.csv` – one-step-ahead forecast per station using the latest month of data.
  - `outputs/extreme_heat_classification.csv` – logistic regression probabilities/labels for May–Sep relative extremes (test years only).
  - `outputs/codecarbon/emissions.csv` – CodeCarbon log with the measured energy use and kg CO2e for each CLI run.

---

## Repository Layout

```
5550-final-project/
├─ data/                         # Raw + monthly aggregated datasets
├─ notebooks/
│  └─ modeling.ipynb             # The CLI workflow with extra analysis/plots
├─ outputs/                      # Predictions, forecasts, classification tables
├─ src/
│  ├─ process.py                 # Daily to monthly aggregation utilities
│  ├─ model.py                   # Regression + classification pipeline
├─ main.py                       # CLI entry point (rebuilds data, trains models, saves outputs)
└─ README.md
```

---

### Environment

- Python 3.10+
- Key libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `codecarbon`
- Install with `pip install -r requirements.txt` (or mirror the Anaconda env used in development).

---

### Pipeline Overview

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
   - Uses the `codecarbon` tracker to measure real-time energy consumption and kg CO2e; falls back to the previous runtime-based approximation if the tracker is unavailable.

---

### Usage

```bash
python main.py
```

The script:

1. Rebuilds `data/monthly_data.csv` from daily raw data.
2. Trains regressors + classifier.
3. Writes the three csv outputs listed above.
4. Prints evaluation tables similar to the modeling notebook (regression metrics per target, classification reports and confusion matrices, best threshold, carbon estimate).

---


## Evaluation & Results

### Temperature (TMAX, TMIN, TMEAN)

- RMSE ≈ 3 °F, R² ≈ 0.96–0.97
- Random Forest consistently best

### Precipitation (PRCP)

- RMSE ≈ 2 in, R² ≈ 0.54
- Expected due to high variance and storm-driven spikes

### Extreme Heat Classification

- ROC AUC ≈ 0.65
- Rare-event nature makes linear classification difficult
- Optimized threshold yields balanced F1 ≈ 0.50

---

## Carbon Footprint Tracking

- Every invocation of `python main.py` wraps preprocessing, model training, inference, and file exports in a [CodeCarbon](https://mlco2.github.io/codecarbon/) tracker. The tracker logs energy usage and estimated kg CO2e into `outputs/codecarbon/emissions.csv`.
- The log file records both the aggregate run statistics (duration, CPU/GPU/RAM power draw, kg CO2e, hardware info) and a unique `run_id` so we can compare different experiments or machines.
- Typical local runs on an Apple M2 laptop consume ≈ 9 s wall-clock time and ≈ 2 × 10⁻⁶ kg CO2e for the full training + inference pass (see the sample row currently in `outputs/codecarbon/emissions.csv`).

---


## Negative Results & Limitations

This project intentionally documents realistic limitations:

### Precipitation forecasting is difficult

- High variance, heavy-tailed distribution
- Driven by localized storm events

### Extreme-heat classification

- Rare-event → class imbalance
- Limited set of predictors (no ENSO/PDO, humidity, etc.)

### Dataset limitations

- Only 7 stations (major cities with different climates)
- Monthly resolution cannot capture fine-scale extremes

---