import yfinance as yf
import pandas as pd
import numpy as np


def download_aapl_data(start, end):
    ticker = "AAPL"
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if "Adj Close" not in data.columns:
        raise ValueError(f"'Adj Close' not found. Columns returned: {list(data.columns)}")

    return data["Adj Close"]


def compute_log_returns(prices):
    """
    Computes daily log returns from adjusted close prices.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns

def load_and_process_aapl(start="2019-01-01", end=None):
    """
    Full pipeline: download AAPL, clean, and compute log returns.
    Returns both the price series and log returns.
    """
    prices = download_aapl_data(start, end)
    returns = compute_log_returns(prices)
    return prices, returns
