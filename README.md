# 📉 AAPL Return Forecasting with ARIMA & GARCH

This project performs **time series forecasting** and **volatility modeling** on Apple's stock data. It uses ARIMA to forecast log returns and GARCH to model volatility, and includes evaluation metrics and residual diagnostics.


---

## 📊 Key Features

- ✅ Downloads AAPL stock data via `yfinance`
- ✅ Computes daily **log returns**
- ✅ Checks stationarity via **ADF test**
- ✅ Fits **ARIMA(1,0,1)** to returns
- ✅ Forecasts next 30 return values
- ✅ Fits **GARCH(1,1)** to volatility
- ✅ Forecasts 30-step volatility horizon
- ✅ Evaluates ARIMA forecasts (MAE, RMSE, MSE)
- ✅ Performs residual diagnostics: ACF, histograms, Ljung-Box

---

## ⚙️ Setup Instructions

### 🔧 Dependencies

Install all requirements via pip:

pip install yfinance pandas numpy matplotlib seaborn statsmodels arch scikit-learn

▶️ How to Run

python main.py

This will:

Download & preprocess AAPL stock data

Fit ARIMA and GARCH models

Forecast future returns & volatility

Evaluate model performance

Plot residuals, ACF, and conduct Ljung-Box tests

