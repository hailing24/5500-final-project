"""
Steps:
  1. Read raw NOAA data from data/rawdata.csv
  2. Clean and aggregate to monthly_data.csv
  3. Add lag features
  4. Train models and generate predictions
  5. Save predictions to outputs/predictions.csv
"""

import os

from src.process import process_monthly_data
from src.model import train_and_predict


def main() -> None:
    # Base directory of this project 
    base_dir = os.path.dirname(os.path.abspath(__file__))

    raw_path = os.path.join(base_dir, "data", "rawdata.csv")
    monthly_output_path = os.path.join(base_dir, "data", "monthly_data.csv")
    predictions_path = os.path.join(base_dir, "outputs", "predictions.csv")

    # Preprocessing Data
    print("=== Preprocessing raw data ===")
    monthly_with_lags = process_monthly_data(
        raw_path=raw_path,
        monthly_output_path=monthly_output_path,
        lags=1,
    )

    # Modeling
    print("=== Training model and predicting ===")
    pred_df, model, metrics = train_and_predict(monthly_with_lags, target="TMEAN")

    # Save predictions
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    pred_df.to_csv(predictions_path, index=False)

    print("=== Done ===")
    print(f"Predictions saved to: {predictions_path}")
    print("=== Metrics for all models ===")
    for model_name, m in metrics.items():
        print(f"{model_name}:")
        print(f"   RMSE = {m['rmse']:.4f}")
        print(f"   MAE  = {m['mae']:.4f}")
        print(f"   R2   = {m['r2']:.4f}")

if __name__ == "__main__":
    main()
