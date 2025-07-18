import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_price_series(prices, save_path=None):
    """
    Plots the historical adjusted close price.
    """
    plt.figure(figsize=(12, 6))
    prices.plot(color="blue", linewidth=1.5)
    plt.title("AAPL Adjusted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_log_returns(returns, save_path=None):
    """
    Plots the daily log returns.
    """
    plt.figure(figsize=(12, 6))
    returns.plot(color="purple", linewidth=1)
    plt.title("AAPL Daily Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_return_distribution(returns, save_path=None):
    """
    Plots the histogram of log returns.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(returns, bins=50, kde=True, color="darkorange")
    plt.title("Distribution of AAPL Log Returns")
    plt.xlabel("Log Return")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
