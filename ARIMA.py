import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


def check_stationarity(returns):
    """
    Performs Augmented Dickey-Fuller test on returns.
    """
    result = adfuller(returns)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("✅ Series is stationary.")
    else:
        print("⚠️ Series may not be stationary.")


def fit_arima_model(returns, order=(1, 0, 1)):
    """
    Fits ARIMA(p,d,q) model to the returns.
    """
    model = ARIMA(returns, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model


def forecast_arima(fitted_model, returns, n_periods=30):
    """
    Forecasts future returns using the fitted ARIMA model.
    """
    forecast = fitted_model.get_forecast(steps=n_periods)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(returns.values, label="Historical Returns")
    plt.plot(range(len(returns), len(returns) + n_periods), mean_forecast, color='orange', label="Forecast")
    plt.fill_between(range(len(returns), len(returns) + n_periods),
                     conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color='orange', alpha=0.3)
    plt.title("ARIMA Forecast of AAPL Log Returns")
    plt.xlabel("Time Step")
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mean_forecast, conf_int
