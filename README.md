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

The project includes simple baseline regression models for demonstration.  
These models are stored in `src/model.py` and include:

- Linear regression for monthly mean temperature
- Linear regression for monthly precipitation totals
- Lagged features such as previous-month temperature or precipitation

Model predictions can be exported to `outputs/results.csv`.

---

## Visualization 


---

## Running the Pipeline

To execute the entire workflow:

1. Open `main.ipynb`
2. Run all cells

This will:

- Load the raw dataset  
- Run preprocessing  
- Generate monthly data  
- Run modeling and visualization  
---


