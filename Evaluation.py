from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def evaluate_forecast(true_values, predicted_values, model_name="Model"):
    """
    Evaluates forecast performance using standard error metrics.

    Parameters:
    - true_values: array-like of actual observed values
    - predicted_values: array-like of model forecasts
    - model_name: str, name of the model for reporting

    Returns:
    - dict with model name and error metrics
    """
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)

    print(f"\n📊 Evaluation for {model_name}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse
    }


def summarize_model_comparisons(results_list):
    """
    Prints a comparison summary from a list of evaluation results.

    Parameters:
    - results_list: list of dictionaries from evaluate_forecast()
    """
    print("\n📋 Model Comparison Summary")
    for res in results_list:
        print(f"- {res['Model']}: MAE={res['MAE']:.6f}, RMSE={res['RMSE']:.6f}, MSE={res['MSE']:.6f}")
