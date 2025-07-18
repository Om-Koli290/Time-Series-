import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


def plot_residuals(residuals, title="Model Residuals"):
    plt.figure(figsize=(12, 4))
    plt.plot(residuals, label="Residuals")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.axhline(0, linestyle="--", color="black")
    plt.tight_layout()
    plt.show()


def plot_residual_distribution(residuals):
    plt.figure(figsize=(10, 4))
    sns.histplot(residuals, kde=True, color="purple")
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.show()

    sm.qqplot(residuals, line="s")
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.show()


def plot_acf_of_residuals(residuals, lags=20):
    sm.graphics.tsa.plot_acf(residuals, lags=lags)
    plt.title("ACF of Residuals")
    plt.tight_layout()
    plt.show()


def ljung_box_test(residuals, lags=[10, 20]):
    lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)
    print("\n📊 Ljung-Box Test for Autocorrelation:")
    print(lb_test)
