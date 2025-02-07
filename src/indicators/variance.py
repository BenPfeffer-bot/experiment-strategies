# File: src/indicators/variance.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR


def compute_variance(
    df: pd.DataFrame, length: int = 20, src: str = "close"
) -> pd.Series:
    """
    Compute the rolling variance of the specified source column.

    Parameters:
      df: pd.DataFrame
         DataFrame that must contain the source column (e.g., 'close').
      length: int, default=20
         Rolling window length.
      src: str, default='close'
         Column name of the data source.

    Returns:
      pd.Series containing the rolling variance.
    """
    if src not in df.columns:
        raise ValueError(f"Column '{src}' not found in DataFrame.")
    variance_series = df[src].rolling(window=length, min_periods=length).var()
    return variance_series


def plot_variance(
    df: pd.DataFrame, variance_series: pd.Series, filename: str = "variance.png"
):
    """
    Plot the price series and its rolling variance.

    Parameters:
      df: pd.DataFrame
         DataFrame containing the price series in the column used for variance (default 'close').
      variance_series: pd.Series
         The rolling variance computed from compute_variance.
      filename: str, default='variance.png'
         The file name for saving the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.style.use("dark_background")
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["close"], label="Close Price", color="cyan", linewidth=1.5)
    plt.title("Price Series")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(
        variance_series.index,
        variance_series,
        label=f"Rolling Variance (Length={len(variance_series.dropna())})",
        color="magenta",
        linewidth=1.5,
    )
    plt.title("Rolling Variance")
    plt.xlabel("Date")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Variance plot saved as {filename}")


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    try:
        for ticker in TICKERS:
            # Assume each ticker has a CSV file with at least a 'Close' column
            data_path = os.path.join(MARKET_DATA_DIR, f"{ticker}_data.csv")
            df = pd.read_csv(data_path)
            # Ensure DataFrame index as dates. Adjust if CSV contains a Date column.
            df.index = pd.to_datetime(df.index)
            df.rename(columns={"Close": "close"}, inplace=True)

            variance_series = compute_variance(df, length=20, src="close")

            # Ensure output directory exists
            variance_dir = os.path.join(FIGURES_DIR, "variance")
            if not os.path.exists(variance_dir):
                os.makedirs(variance_dir)
            plot_variance(
                df,
                variance_series,
                filename=os.path.join(variance_dir, f"{ticker}_variance.png"),
            )
            print(variance_series.tail(5))
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
