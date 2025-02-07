# File: src/indicators/kelter_channels.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjust sys.path to load configuration
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Average True Range (ATR), which is used for market volatility.

    Parameters:
      df: pd.DataFrame
         Must contain 'high', 'low', and 'close' columns.
      period: int, default=14
         lookback period for ATR calculation.

    Returns:
      pd.Series representing the ATR.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range is the maximum of these three values
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def compute_kelter_channels(
    df: pd.DataFrame, period: int = 14, multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Compute Kelter Channels based on an Exponential Moving Average (EMA) and ATR.

    Parameters:
      df: pd.DataFrame
         Must contain 'close', 'high', and 'low' to compute ATR.
      period: int, default=14
         Period to compute the EMA and ATR.
      multiplier: float, default=1.5
         Multiplier for ATR to set the channel width.

    Returns:
      pd.DataFrame with columns:
         - ema: Exponential Moving Average (midline).
         - atr: ATR value.
         - upper_channel: ema + (multiplier * atr).
         - lower_channel: ema - (multiplier * atr).
    """
    # Compute EMA using span=period
    ema = df["close"].ewm(span=period, adjust=False).mean()
    atr = compute_atr(df, period=period)

    channels = pd.DataFrame(index=df.index)
    channels["ema"] = ema
    channels["atr"] = atr
    channels["upper_channel"] = ema + multiplier * atr
    channels["lower_channel"] = ema - multiplier * atr
    return channels


def plot_kelter_channels(
    df: pd.DataFrame, channels: pd.DataFrame, filename: str = "kelter_channels.png"
):
    """
    Plot the Kelter Channels along with the close price.

    Parameters:
      df: pd.DataFrame
         Must contain a 'close' column.
      channels: pd.DataFrame
         DataFrame returned from compute_kelter_channels.
      filename: str, default='kelter_channels.png'
         Output filename for the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.style.use("dark_background")
    # Price series (close)
    plt.plot(df.index, df["close"], label="Close Price", color="white", linewidth=1.5)
    # EMA midline
    plt.plot(
        channels.index,
        channels["ema"],
        label="EMA (Midline)",
        color="fuchsia",
        linewidth=2,
    )
    # Upper and Lower Channels
    plt.plot(
        channels.index,
        channels["upper_channel"],
        label="Upper Channel",
        color="green",
        linestyle="--",
        linewidth=1.5,
    )
    plt.plot(
        channels.index,
        channels["lower_channel"],
        label="Lower Channel",
        color="red",
        linestyle="--",
        linewidth=1.5,
    )

    plt.title("Kelter Channels")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best", fontsize="small", framealpha=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Kelter Channels plot saved as {filename}")


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    try:
        for ticker in TICKERS:
            data_path = os.path.join(MARKET_DATA_DIR, f"{ticker}_data.csv")
            df = pd.read_csv(data_path)
            # Ensure the DataFrame index is datetime
            df.index = pd.to_datetime(df.index)
            # Standardize column names to lowercase for consistency
            df.rename(
                columns={"Close": "close", "High": "high", "Low": "low"}, inplace=True
            )

            channels = compute_kelter_channels(df, period=14, multiplier=1.5)

            # Output directory for Kelter Channels plots
            channels_dir = os.path.join(FIGURES_DIR, "kelter_channels")
            if not os.path.exists(channels_dir):
                os.makedirs(channels_dir)
            plot_kelter_channels(
                df,
                channels,
                filename=os.path.join(channels_dir, f"{ticker}_kelter_channels.png"),
            )
            print(channels.tail(5))
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
