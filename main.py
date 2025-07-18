import pandas as pd
from Data_Loader import load_and_process_aapl
from ARIMA import check_stationarity, fit_arima_model, forecast_arima
from GARCH import fit_garch_model, forecast_volatility
from Evaluation import evaluate_forecast, summarize_model_comparisons
from Residual_Analysis import (
    plot_residuals,
    plot_residual_distribution,
    plot_acf_of_residuals,
    ljung_box_test
)

# 1. Load Data
prices, returns = load_and_process_aapl()

# 2. Train-Test Split
train = returns[:-30]
test = returns[-30:]

# 3. ARIMA
check_stationarity(train)
arima_model = fit_arima_model(train, order=(1, 0, 1))
arima_forecast, _ = forecast_arima(arima_model, train, n_periods=30)

# 4. GARCH
garch_model = fit_garch_model(train)
garch_forecast = forecast_volatility(garch_model, horizon=30)

# 5. Evaluate Models (note: GARCH forecasts volatility, not returns)
arima_eval = evaluate_forecast(test.values, arima_forecast, model_name="ARIMA")

# Optional: If you have a GARCH-based return estimate or approximation:
# garch_eval = evaluate_forecast(test.values, garch_return_approx, model_name="GARCH")

summarize_model_comparisons([arima_eval])

# 6. Residual Analysis (ARIMA)
arima_resid = arima_model.resid
plot_residuals(arima_resid, title="ARIMA Residuals")
plot_residual_distribution(arima_resid)
plot_acf_of_residuals(arima_resid)
ljung_box_test(arima_resid)
