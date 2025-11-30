from __future__ import annotations

import os
import time

from src.model import ModelResult, train_and_predict
from src.process import process_monthly_data
from sklearn.metrics import classification_report, confusion_matrix

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

# This script recreates the modeling workflow from the Jupyter notebook,
# but in a single reproducible command-line entry point.
# It performs:
#   (1) raw to monthly preprocessing
#   (2) model training (regression + classification)
#   (3) prediction export
#   (4) next-month forecasting
#   (5) carbon-cost estimation

CARBON_POWER_KW = 0.15  # Approximate laptop power draw (kW)
CARBON_INTENSITY_KG_PER_KWH = 0.475  # Global average grid carbon intensity


def _print_regression_metrics(results: ModelResult) -> None:
    """
    Pretty-print regression results for each model and each target variable.
    Mirrors the notebook output so results are easy to compare.
    """
    print("=== Regression Metrics (test set) ===")
    for model_name, per_target in results.metrics.items():
        print(f"\n{model_name}:")
        for target, stats in per_target.items():
            print(
                f"  {target}: RMSE={stats['rmse']:.3f}  "
                f"MAE={stats['mae']:.3f}  R2={stats['r2']:.3f}"
            )


def _print_classification_metrics(results: ModelResult) -> None:
    """
    Print classification performance for the relative heat-extreme model.
    Includes ROC AUC + F1 (default vs. optimized threshold),
    plus classification reports and confusion matrices to mirror the notebook.
    """
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

    # If prediction rows exist, also print full classification reports
    if results.classification_predictions is not None:
        preds = results.classification_predictions
        y_true = preds["observed"].astype(int)

        # Print reports for both decision thresholds
        for label, column in (
            ("Default 0.5 threshold", "pred_default"),
            ("F1-optimized threshold", "pred_best"),
        ):
            y_pred = preds[column].astype(int)
            print(f"\nLogistic Regression ({label})")
            print(
                classification_report(
                    y_true,
                    y_pred,
                    digits=3,
                    zero_division=0,
                )
            )
            print("Confusion matrix:")
            print(confusion_matrix(y_true, y_pred))


def _estimate_carbon_cost(duration_seconds: float) -> float:
    """
    Convert runtime into estimated kg CO2e using a constant-power model:
        carbon = hours * power(kW) * carbon_intensity(kg/kWh)
    This is a simplified but widely accepted approximation for small workloads.
    """
    hours = duration_seconds / 3600.0
    return hours * CARBON_POWER_KW * CARBON_INTENSITY_KG_PER_KWH


def _start_emissions_tracker(outputs_dir: str):
    """
    Kick off CodeCarbon tracking if the library is available.
    Returns the started tracker and the directory where logs are written.
    """
    if EmissionsTracker is None:
        print(
            "CodeCarbon is not installed. Run `pip install codecarbon` to enable "
            "hardware-based emissions tracking."
        )
        return None, None

    tracker_output_dir = os.path.join(outputs_dir, "codecarbon")
    os.makedirs(tracker_output_dir, exist_ok=True)

    try:
        tracker = EmissionsTracker(
            project_name="climate_forecasting",
            output_dir=tracker_output_dir,
            save_to_file=True,
            log_level="warning",
        )
        tracker.start()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Warning: CodeCarbon tracker could not be started ({exc}).")
        return None, None

    return tracker, tracker_output_dir


def _stop_emissions_tracker(tracker):
    """
    Stop CodeCarbon tracker and return measured kg CO2e, if tracking was enabled.
    """
    if tracker is None:
        return None

    try:
        return tracker.stop()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Warning: Failed to stop CodeCarbon tracker ({exc}).")
        return None


def main() -> None:
    """
    End-to-end pipeline entry point:
        - Load raw daily data
        - Build monthly aggregates
        - Train regression + classification models
        - Save predictions and next-month forecasts
        - Print all evaluation metrics
        - Estimate carbon emissions

    This guarantees full reproducibility independent of the notebook environment.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Input/output paths for data and results
    raw_path = os.path.join(base_dir, "data", "rawdata.csv")
    monthly_output_path = os.path.join(base_dir, "data", "monthly_data.csv")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    predictions_path = os.path.join(outputs_dir, "predictions.csv")
    forecast_path = os.path.join(outputs_dir, "forecast_next_month.csv")
    classification_path = os.path.join(outputs_dir, "extreme_heat_classification.csv")

    tracker, tracker_output_dir = _start_emissions_tracker(outputs_dir)
    start_time = time.perf_counter()

    try:
        # Preprocess raw daily to monthly 
        print("=== Preprocessing raw data ===")
        monthly = process_monthly_data(
            raw_path=raw_path,
            monthly_output_path=monthly_output_path,
        )
        # Rebuilding monthly_data.csv each run guarantees notebook + script consistency.

        # Train all models + generate predictions
        print("=== Training models and generating predictions ===")
        results = train_and_predict(monthly)

        # Save regression predictions
        results.predictions.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")

        # Save next-month forecast
        if results.forecast is not None:
            results.forecast.to_csv(forecast_path, index=False)
            print(f"One-month-ahead forecast saved to: {forecast_path}")

        # Save classification outputs
        if results.classification_predictions is not None:
            results.classification_predictions.to_csv(classification_path, index=False)
            print(
                "Extreme-heat classification predictions saved to: "
                f"{classification_path}"
            )

        # Print evaluation summaries 
        print("=== Done ===")
        print(f"Best regression model: {results.model_used}")
        _print_regression_metrics(results)
        _print_classification_metrics(results)

    finally:
        # Compute carbon footprint 
        duration = time.perf_counter() - start_time
        emissions_kg = _stop_emissions_tracker(tracker)

        if emissions_kg is not None:
            log_hint = (
                f" (detailed log in {os.path.join(tracker_output_dir, 'emissions.csv')})"
                if tracker_output_dir
                else ""
            )
            print(
                f"\nCodeCarbon measured emissions: {emissions_kg:.4f} kg CO2e "
                f"(runtime {duration:.1f}s){log_hint}"
            )
        else:
            carbon_cost = _estimate_carbon_cost(duration)
            print(
                f"\nEstimated carbon cost: {carbon_cost:.4f} kg CO2e "
                f"(runtime {duration:.1f}s, power {CARBON_POWER_KW:.3f} kW, "
                f"intensity {CARBON_INTENSITY_KG_PER_KWH:.3f} kg/kWh)"
            )


if __name__ == "__main__":
    main()
