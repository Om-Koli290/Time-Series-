from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np

def fit_garch_model(returns):
    """
    Fits a GARCH(1,1) model to the return series.
    Returns the fitted model.
    """
    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    fitted_model = model.fit(update_freq=10, disp="off")
    print(fitted_model.summary())
    return fitted_model


def plot_volatility(fitted_model):
    """
    Plots conditional volatility estimated by the GARCH model.
    """
    cond_vol = fitted_model.conditional_volatility

    plt.figure(figsize=(12, 6))
    plt.plot(cond_vol, color='firebrick', linewidth=1)
    plt.title("Conditional Volatility (GARCH) of AAPL Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.show()


def forecast_volatility(fitted_model, horizon=30):
    """
    Forecasts volatility for a given number of days.
    Returns forecasted standard deviation.
    """
    forecasts = fitted_model.forecast(horizon=horizon)
    variance_forecast = forecasts.variance.values[-1]
    stddev_forecast = np.sqrt(variance_forecast)
    return stddev_forecast
