# macd and signal line
"""
macd.py

This module implements a MACD indicator with histogram and cross dots,
inspired by the provided Pine Script by ChrisMoody. The indicator
computes the MACD line (difference between fast and slow EMA), a signal
line (simple moving average of MACD), and the histogram (MACD minus Signal).

Features:
 - Option to display the MACD and Signal lines.
 - Change MACD line color: if MACD >= Signal then "lime", otherwise "red",
   when macd_color_change is True. Otherwise, MACD line is drawn in red
   and Signal in lime or yellow, as defined.
 - Histogram can be colored with 4 colors based on directional changes:
    • If histogram is increasing and > 0  -> "aqua"
    • If histogram is decreasing and > 0  -> "blue"
    • If histogram is decreasing and <= 0 -> "red"
    • If histogram is increasing and <= 0 -> "maroon"
   If hist_color_change is False, histogram is drawn in gray.
 - Option to plot dots whenever MACD and Signal cross.

Usage:
  Run the module as a script for a demonstration on synthetic data.

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


class MACD:
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Initialize MACD indicator.

        Parameters:
        • fast_period: int
            The period for the fast EMA
        • slow_period: int
            The period for the slow EMA
        • signal_period: int
            The period for the signal line EMA
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def compute(self, src: pd.Series) -> dict:
        """
        Compute MACD indicator.

        Parameters:
        • src: pd.Series
            Price series

        Returns:
        • dict with keys:
            - macd: MACD line
            - signal: Signal line
            - hist: MACD histogram
        """
        # Calculate EMAs
        fast_ema = src.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = src.ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        macd = fast_ema - slow_ema

        # Calculate signal line
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()

        # Calculate histogram
        hist = macd - signal

        return {"macd": macd, "signal": signal, "hist": hist}


def compute_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """Legacy function for backward compatibility."""
    macd = MACD(fast_period, slow_period, signal_period)
    return macd.compute(df["close"] if "close" in df else df)


def _get_line_segments(x, y):
    """
    Create line segments for multi-color line plotting.
    Returns array of segments: [[(x0,y0), (x1,y1)], [(x1,y1), (x2,y2)], ...]
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_macd(
    df: pd.DataFrame,
    macd_df: pd.DataFrame,
    show_macd: bool = True,
    show_dots: bool = True,
    show_hist: bool = True,
    macd_color_change: bool = True,
    hist_color_change: bool = True,
    filename: str = "macd.png",
):
    """
    Plot the MACD indicator along with the histogram and cross dots.

    Parameters:
      df: pd.DataFrame
         DataFrame containing a datetime index for the x-axis.
      macd_df: pd.DataFrame
         DataFrame with "macd", "signal", and "hist" columns.
      show_macd: bool
         If True, plot MACD and Signal lines.
      show_dots: bool
         If True, plot dots where MACD crosses Signal.
      show_hist: bool
         If True, display the histogram.
      macd_color_change: bool
         If True, MACD is colored based on crossing relative to the Signal.
      hist_color_change: bool
         If True, histogram uses 4 colors based on directional changes.
      filename: str
         Name of the file to save the plot.
    """
    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df.index))

    # Unpack series
    macd_series = macd_df["macd"].to_numpy()
    signal_series = macd_df["signal"].to_numpy()
    hist_series = macd_df["hist"].to_numpy()

    # Determine MACD line color for each segment
    # Use condition: if MACD >= Signal then "lime", else "red"
    if macd_color_change:
        seg_colors = [
            "lime" if m >= s else "red" for m, s in zip(macd_series, signal_series)
        ]
    else:
        seg_colors = ["red"] * len(macd_series)

    # Plot MACD line using segments with individual colors (if show_macd and available data)
    if show_macd:
        # Remove NaN values (can be at beginning due to rolling signal)
        valid = ~np.isnan(macd_series) & ~np.isnan(signal_series)
        xv = x[valid]
        yv = macd_series[valid]
        # Create segments from valid points
        segments = _get_line_segments(xv, yv)
        # Color for each segment: use color of the starting point of the segment
        seg_color_list = [seg_colors[i] for i, val in enumerate(valid) if val][:-1]
        lc = LineCollection(segments, colors=seg_color_list, linewidth=4)
        ax.add_collection(lc)

        # Plot Signal line.
        # For signal line, if macd_color_change is True use constant "yellow", else "lime"
        sig_color = "yellow" if macd_color_change else "lime"
        ax.plot(x, signal_series, label="Signal Line", color=sig_color, linewidth=2)

    # Plot histogram bars
    if show_hist:
        bars_color = []
        # Determine color for each histogram bar based on its move compared with previous value
        for i in range(len(hist_series)):
            # For the first bar, default to yellow if color change enabled else gray.
            if i == 0 or np.isnan(hist_series[i]) or np.isnan(hist_series[i - 1]):
                bars_color.append("yellow" if hist_color_change else "gray")
            else:
                if hist_color_change:
                    if (hist_series[i] > hist_series[i - 1]) and (hist_series[i] > 0):
                        bars_color.append("aqua")
                    elif (hist_series[i] < hist_series[i - 1]) and (hist_series[i] > 0):
                        bars_color.append("blue")
                    elif (hist_series[i] < hist_series[i - 1]) and (
                        hist_series[i] <= 0
                    ):
                        bars_color.append("red")
                    elif (hist_series[i] > hist_series[i - 1]) and (
                        hist_series[i] <= 0
                    ):
                        bars_color.append("maroon")
                    else:
                        bars_color.append("yellow")
                else:
                    bars_color.append("gray")
        ax.bar(x, hist_series, color=bars_color, width=0.8, label="Histogram")

    # Plot dots at MACD/Signal crosses if enabled.
    if show_dots and show_macd:
        # Identify cross points: cross where the sign of (MACD - Signal) changes.
        macd_minus_signal = macd_series - signal_series
        crosses = np.sign(macd_minus_signal[:-1]) != np.sign(macd_minus_signal[1:])
        cross_indices = (
            np.where(crosses)[0] + 1
        )  # +1 gives the index of the crossing bar

        # For each cross, we plot a dot on the Signal line
        dot_color = []
        for idx in cross_indices:
            # For dot, use same color as MACD line at that point
            if macd_color_change:
                dot_color.append(
                    "lime" if macd_series[idx] >= signal_series[idx] else "red"
                )
            else:
                dot_color.append("red")
        ax.scatter(
            x[cross_indices],
            signal_series[cross_indices],
            color=dot_color,
            s=100,
            zorder=5,
            label="Cross Dot",
        )

    # Draw horizontal zero line
    ax.axhline(0, color="white", linewidth=2, linestyle="solid", label="Zero Line")

    # Adjust ticks to show dates
    ax.set_xticks(x[:: max(1, len(x) // 10)])
    # Handle both datetime index and string index cases
    if hasattr(df.index, "strftime"):
        ax.set_xticklabels(
            df.index.strftime("%Y-%m-%d")[:: max(1, len(x) // 10)],
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticklabels(df.index[:: max(1, len(x) // 10)], rotation=45, ha="right")

    ax.set_title("CM_MacD_Ult_MTF")
    ax.set_xlabel("Date")
    ax.set_ylabel("Indicator Value")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"MACD plot saved as {filename}")


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    # np.random.seed(42)
    # dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    # # Create a random walk for close prices
    # close = np.cumsum(np.random.randn(len(dates))) + 100

    for ticker in TICKERS:
        prices = pd.read_csv(f"{RAW_DATA_DIR}/{ticker}_data.csv")["Close"]
        dates = prices.index
        close = prices

        df_sample = pd.DataFrame({"close": close}, index=dates)

        # Compute MACD components
        macd_data = compute_macd(
            df_sample, fast_period=12, slow_period=26, signal_period=9
        )

        if not os.path.exists(f"{FIGURES_DIR}/macd"):
            os.makedirs(f"{FIGURES_DIR}/macd")

        # Plot the indicator:
        plot_macd(
            df_sample,
            macd_data,
            show_macd=True,
            show_dots=True,
            show_hist=True,
            macd_color_change=True,
            hist_color_change=True,
            filename=f"{FIGURES_DIR}/macd/{ticker}_macd.png",
        )

        # Display last few computed MACD values.
        print("Last 5 MACD data points:")
        print(macd_data)
