# 5500-final-project

### Climate Data Preprocessing Pipeline

This repository contains a full end-to-end workflow for preparing, modeling, and visualizing NOAA climate data.
The goal is to take raw daily weather observations, clean and organize them, aggregate them to the monthly level, and build simple predictive models.

The project is organized into modular components so that preprocessing, modeling, and visualization can be run independently or as a single pipeline.

This project uses **NOAA daily weather observations** and prepares them for later climate analysis.

---

## Repository Structure

```plaintext
5550-final-project/
│
├── data/
│   ├── rawdata.csv              # Original NOAA daily dataset
│   └── monthly_data.csv         # Cleaned and aggregated monthly data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb   # Explore and clean raw data
│   ├── 02_modeling.ipynb        # Train simple climate prediction models
│   └── 03_visualization.ipynb   # Monthly trends and forecast plots
│
├── src/
│   ├── preprocess.py            # Functions for loading + cleaning + monthly aggregation
│   ├── model.py                 # Baseline regression models
│   └── utils.py                 # Helper functions
│
├── outputs/
│   ├── results.csv              # Model predictions / aggregated outputs
│   └── figures/                 # Saved plots
│
├── requirements.txt             # Python dependencies
│
├── README.md                    # Project documentation
│
└── main.ipynb                   # Runs the full pipeline end-to-end
```

---

## Preprocessing Pipeline

Raw NOAA daily observations are converted into structured monthly climate summaries.  
The preprocessing workflow includes:

- Reading the raw `rawdata.csv`
- Selecting key variables such as PRCP, TMAX, and TMIN
- Converting and standardizing date formats
- Computing monthly precipitation totals and temperature averages
- Saving the processed output to `data/monthly_data.csv`

All preprocessing logic is implemented inside `src/preprocess.py`.

---

##  Modeling

The modeling component trains **three regression models** and automatically selects the best-performing one.

### **Included Models**
- **Linear Regression**
- **Ridge Regression**
- **Random Forest Regressor**

### **Pipeline Features**
- Chronological train/test split (80% train / 20% test)
- Per-model evaluation metrics:
  - RMSE
  - MAE
  - R²
- Automatic model selection (best model = lowest RMSE)
- Predictions saved to:
outputs/predictions.csv

---

## Visualization(To be done)


---

## Running the Pipeline

To execute the full workflow, simply run:
python main.py

This will:

1. Load the raw NOAA dataset
2. Process and aggregate monthly climate statistics
3. Add lag features
4. Train and compare three regression models
5. Select the best model
6. Save predictions to outputs/predictions.csv
---


