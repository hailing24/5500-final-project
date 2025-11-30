from __future__ import annotations

import os
import time

from src.model import ModelResult, train_and_predict
from src.process import process_monthly_data

CARBON_POWER_KW = 0.15  # Approximate laptop draw
CARBON_INTENSITY_KG_PER_KWH = 0.475  # Global average grid intensity


def _print_regression_metrics(results: ModelResult) -> None:
    print("=== Regression Metrics (test set) ===")
    for model_name, per_target in results.metrics.items():
        print(f"\n{model_name}:")
        for target, stats in per_target.items():
            print(
                f"  {target}: RMSE={stats['rmse']:.3f}  "
                f"MAE={stats['mae']:.3f}  R2={stats['r2']:.3f}"
            )


def _print_classification_metrics(results: ModelResult) -> None:
    if not results.classification_metrics:
        return
    metrics = results.classification_metrics
    print("\n=== Classification (May-Sep relative heat extremes) ===")
    print(
        f"ROC AUC={metrics['roc_auc']:.3f}  "
        f"F1@0.5={metrics['f1_default']:.3f}  "
        f"F1@best={metrics['f1_optimized']:.3f}  "
        f"best_threshold={metrics['best_threshold']:.2f}"
    )


def _estimate_carbon_cost(duration_seconds: float) -> float:
    """
    Convert runtime into kg CO2e using a constant-power approximation.
    """
    hours = duration_seconds / 3600.0
    return hours * CARBON_POWER_KW * CARBON_INTENSITY_KG_PER_KWH


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "rawdata.csv")
    monthly_output_path = os.path.join(base_dir, "data", "monthly_data.csv")
    outputs_dir = os.path.join(base_dir, "outputs")
    predictions_path = os.path.join(outputs_dir, "predictions.csv")
    forecast_path = os.path.join(outputs_dir, "forecast_next_month.csv")
    classification_path = os.path.join(outputs_dir, "extreme_heat_classification.csv")

    start_time = time.perf_counter()

    print("=== Preprocessing raw data ===")
    monthly = process_monthly_data(
        raw_path=raw_path,
        monthly_output_path=monthly_output_path,
    )

    print("=== Training models and generating predictions ===")
    results = train_and_predict(monthly)

    os.makedirs(outputs_dir, exist_ok=True)
    results.predictions.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

    if results.forecast is not None:
        results.forecast.to_csv(forecast_path, index=False)
        print(f"One-month-ahead forecast saved to: {forecast_path}")

    if results.classification_predictions is not None:
        results.classification_predictions.to_csv(classification_path, index=False)
        print(
            "Extreme-heat classification predictions saved to: "
            f"{classification_path}"
        )

    print("=== Done ===")
    print(f"Best regression model: {results.model_used}")
    _print_regression_metrics(results)
    _print_classification_metrics(results)

    duration = time.perf_counter() - start_time
    carbon_cost = _estimate_carbon_cost(duration)
    print(
        f"\nEstimated carbon cost: {carbon_cost:.4f} kg CO2e "
        f"(runtime {duration:.1f}s, power {CARBON_POWER_KW:.3f} kW, "
        f"intensity {CARBON_INTENSITY_KG_PER_KWH:.3f} kg/kWh)"
    )


if __name__ == "__main__":
    main()
