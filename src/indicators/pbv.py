"""
This module implements the Price Volume Trend (PVT) indicator.
The PVT is calculated as the cumulative sum of:
    (change in close / previous close) * volume

If no volume data is provided (i.e. cumulative volume = 0),
the module raises an error.

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


def compute_pvt(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Price Volume Trend (PVT) indicator.

    Parameters:
      df : pd.DataFrame
         DataFrame containing at least the following columns:
           - 'close': closing prices.
           - 'volume': volume data.

    Returns:
      pd.Series representing the PVT.

    Raises:
      ValueError: If cumulative volume is 0 at the end of data.
    """
    # Check that necessary columns exist
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'volume' columns.")

    # Check cumulative volume across the DataFrame.
    total_volume = df["volume"].sum()
    if total_volume == 0:
        raise ValueError("No volume is provided by the data vendor.")

    # Compute the daily change ratio: (close - previous close) / previous close.
    change_ratio = df["close"].diff() / df["close"].shift(1)
    # Multiply by volume to obtain daily contribution to PVT.
    daily_pvt = change_ratio * df["volume"]
    # First value should be 0 (or you can set it to the first available PVT)
    daily_pvt.iloc[0] = 0

    # Cumulative sum (starting at zero)
    pvt = daily_pvt.cumsum()

    return pvt


def plot_pvt(df: pd.DataFrame, pvt: pd.Series, filename: str = "pvt.png"):
    """
    Plot the Price Volume Trend (PVT) indicator.

    Parameters:
      df : pd.DataFrame
         DataFrame that should include a datetime index or similar for the x-axis.
      pvt : pd.Series
         The Price Volume Trend computed from compute_pvt.
      filename: str, optional
         Filename for saving the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, pvt, label="PVT", color="#2962FF", linewidth=2)
    plt.title("Price Volume Trend (PVT)")
    plt.xlabel("Date")
    plt.ylabel("PVT")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"PVT plot saved as {filename}")


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Generate sample data
    # np.random.seed(42)
    # dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    try:
        for ticker in TICKERS:
            prices = pd.read_csv(f"{RAW_DATA_DIR}/{ticker}_data.csv")["Close"]
            dates = prices.index
            close = prices
            volume = pd.read_csv(f"{RAW_DATA_DIR}/{ticker}_data.csv")["Volume"]

            # Build DataFrame
            df_sample = pd.DataFrame({"close": close, "volume": volume}, index=dates)

            # Compute the PVT indicator.
            pvt = compute_pvt(df_sample)

            if not os.path.exists(f"{FIGURES_DIR}/pbv"):
                os.makedirs(f"{FIGURES_DIR}/pbv")

            # Plot and save the PVT chart.
            plot_pvt(df_sample, pvt, filename=f"{FIGURES_DIR}/pbv/{ticker}_pvt.png")

            # Display the last few values.
            print("Last 5 PVT values:")
            print(pvt.tail(5))
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
