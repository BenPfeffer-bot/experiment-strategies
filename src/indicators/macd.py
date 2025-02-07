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
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
import logging

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


@dataclass
class MACDParameters:
    """Parameters for MACD calculation"""

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    price_column: str = "close"


class MACD:
    """
    Moving Average Convergence Divergence (MACD) indicator implementation.

    The MACD indicator is a trend-following momentum indicator that shows the relationship
    between two moving averages of an asset's price. It consists of:
    - MACD Line: Difference between fast and slow EMAs
    - Signal Line: EMA of the MACD line
    - Histogram: Difference between MACD and Signal lines

    Attributes:
        params (MACDParameters): Configuration parameters for MACD calculation
        logger (logging.Logger): Logger instance for error and info messages
    """

    def __init__(self, params: Optional[MACDParameters] = None):
        """
        Initialize MACD indicator with given parameters.

        Args:
            params: Optional[MACDParameters]
                Parameters for MACD calculation. If None, uses default values.
        """
        self.params = params if params else MACDParameters()
        self._validate_parameters()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize containers for computed values
        self.macd: Optional[pd.Series] = None
        self.signal: Optional[pd.Series] = None
        self.histogram: Optional[pd.Series] = None

    def _validate_parameters(self) -> None:
        """Validate MACD parameters."""
        if self.params.fast_period >= self.params.slow_period:
            raise ValueError("Fast period must be less than slow period")
        if (
            self.params.fast_period <= 0
            or self.params.slow_period <= 0
            or self.params.signal_period <= 0
        ):
            raise ValueError("All periods must be positive")

    def compute(self, data: Union[pd.Series, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Compute MACD indicator components.

        Args:
            data: Union[pd.Series, pd.DataFrame]
                Price data. If DataFrame, uses the column specified in params.price_column

        Returns:
            Dict with keys 'macd', 'signal', and 'hist' containing the computed Series

        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        try:
            # Extract price series
            if isinstance(data, pd.DataFrame):
                if self.params.price_column not in data.columns:
                    raise ValueError(
                        f"Price column '{self.params.price_column}' not found in data"
                    )
                price_series = data[self.params.price_column]
            else:
                price_series = data

            # Validate data
            if len(price_series) < self.params.slow_period:
                raise ValueError(
                    f"Insufficient data points. Need at least {self.params.slow_period}"
                )

            # Calculate EMAs
            fast_ema = price_series.ewm(
                span=self.params.fast_period, adjust=False
            ).mean()
            slow_ema = price_series.ewm(
                span=self.params.slow_period, adjust=False
            ).mean()

            # Calculate MACD line
            self.macd = fast_ema - slow_ema

            # Calculate signal line
            self.signal = self.macd.ewm(
                span=self.params.signal_period, adjust=False
            ).mean()

            # Calculate histogram
            self.histogram = self.macd - self.signal

            return {"macd": self.macd, "signal": self.signal, "hist": self.histogram}

        except Exception as e:
            self.logger.error(f"Error computing MACD: {str(e)}")
            raise

    def generate_signals(self, threshold: float = 0) -> pd.Series:
        """
        Generate trading signals based on MACD crossovers.

        Args:
            threshold: float
                Minimum difference for signal generation

        Returns:
            pd.Series with values:
                1 for buy signals (MACD crosses above Signal)
                -1 for sell signals (MACD crosses below Signal)
                0 for no signal
        """
        if self.macd is None or self.signal is None:
            raise ValueError("MACD not computed. Call compute() first.")

        signals = pd.Series(0, index=self.macd.index)

        # Calculate crossovers
        macd_above_signal = self.macd > self.signal
        signal_crossover = macd_above_signal.astype(int).diff()

        # Generate signals with threshold
        signals[signal_crossover > 0] = 1  # Buy signal
        signals[signal_crossover < 0] = -1  # Sell signal

        # Apply threshold filter
        if threshold > 0:
            signals[abs(self.macd - self.signal) < threshold] = 0

        return signals

    def get_current_signal(self) -> Tuple[int, float]:
        """
        Get the current trading signal and strength.

        Returns:
            Tuple[int, float]: (signal, strength)
                signal: 1 for buy, -1 for sell, 0 for neutral
                strength: absolute difference between MACD and Signal
        """
        if self.macd is None or self.signal is None:
            raise ValueError("MACD not computed. Call compute() first.")

        current_macd = self.macd.iloc[-1]
        current_signal = self.signal.iloc[-1]
        current_hist = self.histogram.iloc[-1]

        # Determine signal
        if current_macd > current_signal:
            signal = 1
        elif current_macd < current_signal:
            signal = -1
        else:
            signal = 0

        strength = abs(current_hist)
        return signal, strength


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
    macd_data: Dict[str, pd.Series],
    show_macd: bool = True,
    show_dots: bool = True,
    show_hist: bool = True,
    macd_color_change: bool = True,
    hist_color_change: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot the MACD indicator with customizable visualization options.

    Args:
        df: pd.DataFrame
            DataFrame containing price data with datetime index
        macd_data: Dict[str, pd.Series]
            Dictionary containing MACD components ('macd', 'signal', 'hist')
        show_macd: bool
            If True, plot MACD and Signal lines
        show_dots: bool
            If True, plot dots at MACD/Signal crossovers
        show_hist: bool
            If True, display the histogram
        macd_color_change: bool
            If True, MACD line color changes based on Signal line crossings
        hist_color_change: bool
            If True, histogram uses multiple colors based on direction
        save_path: Optional[str]
            If provided, saves the plot to this path
        title: Optional[str]
            Custom title for the plot

    Raises:
        ValueError: If required data is missing or invalid
    """
    try:
        # Validate input data
        required_keys = ["macd", "signal", "hist"]
        if not all(key in macd_data for key in required_keys):
            raise ValueError("macd_data missing required components")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(df.index))

        # Extract data
        macd_series = macd_data["macd"].to_numpy()
        signal_series = macd_data["signal"].to_numpy()
        hist_series = macd_data["hist"].to_numpy()

        # Plot histogram if enabled
        if show_hist:
            colors = []
            prev_hist = None
            for hist in hist_series:
                if hist_color_change:
                    if prev_hist is None:
                        color = "gray"
                    else:
                        if hist > 0:
                            color = "aqua" if hist > prev_hist else "blue"
                        else:
                            color = "maroon" if hist > prev_hist else "red"
                    prev_hist = hist
                else:
                    color = "gray"
                colors.append(color)
            ax.bar(
                x, hist_series, color=colors, width=0.8, label="Histogram", alpha=0.5
            )

        # Plot MACD and Signal lines if enabled
        if show_macd:
            # Plot MACD line with color changes
            valid_mask = ~np.isnan(macd_series) & ~np.isnan(signal_series)
            x_valid = x[valid_mask]
            macd_valid = macd_series[valid_mask]

            if macd_color_change:
                segments = np.column_stack(
                    (x_valid[:-1], macd_valid[:-1], x_valid[1:], macd_valid[1:])
                )
                segments = segments.reshape(-1, 2, 2)

                # Determine colors based on MACD vs Signal relationship
                colors = [
                    "lime" if m >= s else "red"
                    for m, s in zip(macd_valid[:-1], signal_series[valid_mask][:-1])
                ]

                line_segments = LineCollection(segments, colors=colors, linewidth=2)
                ax.add_collection(line_segments)
            else:
                ax.plot(x_valid, macd_valid, "red", label="MACD", linewidth=2)

            # Plot Signal line
            ax.plot(
                x_valid,
                signal_series[valid_mask],
                "yellow" if macd_color_change else "lime",
                label="Signal",
                linewidth=1.5,
            )

            # Plot crossover dots if enabled
            if show_dots:
                crossovers = np.diff(macd_series > signal_series).astype(bool)
                crossover_idx = np.where(crossovers)[0] + 1

                dot_colors = [
                    "lime" if macd_series[i] >= signal_series[i] else "red"
                    for i in crossover_idx
                ]

                ax.scatter(
                    x[crossover_idx],
                    signal_series[crossover_idx],
                    c=dot_colors,
                    s=100,
                    zorder=5,
                    label="Crossovers",
                )

        # Add zero line
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.5)

        # Customize appearance
        ax.set_title(title or "MACD Indicator")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

        # Set x-axis ticks to show dates
        tick_locations = np.linspace(0, len(x) - 1, min(10, len(x)))
        tick_locations = tick_locations.astype(int)
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(
            [df.index[i].strftime("%Y-%m-%d") for i in tick_locations],
            rotation=45,
            ha="right",
        )

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        plt.close()
        raise ValueError(f"Error plotting MACD: {str(e)}")


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    """Example usage of the MACD indicator"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create output directory if it doesn't exist
        macd_output_dir = os.path.join(FIGURES_DIR, "macd")
        os.makedirs(macd_output_dir, exist_ok=True)

        # Process each ticker
        for ticker in TICKERS:
            try:
                logger.info(f"Processing MACD for {ticker}")

                # Load data
                data_path = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
                df = pd.read_csv(data_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

                # Initialize MACD with custom parameters
                params = MACDParameters(
                    fast_period=12,
                    slow_period=26,
                    signal_period=9,
                    price_column="Close",
                )
                macd_indicator = MACD(params)

                # Compute MACD
                macd_data = macd_indicator.compute(df)

                # Generate signals
                signals = macd_indicator.generate_signals(threshold=0.1)

                # Get current signal
                current_signal, signal_strength = macd_indicator.get_current_signal()
                logger.info(
                    f"{ticker} current signal: {current_signal} (strength: {signal_strength:.4f})"
                )

                # Plot MACD
                save_path = os.path.join(macd_output_dir, f"{ticker}_macd.png")
                plot_macd(
                    df,
                    macd_data,
                    show_macd=True,
                    show_dots=True,
                    show_hist=True,
                    save_path=save_path,
                    title=f"MACD Indicator - {ticker}",
                )

                logger.info(f"MACD plot saved for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
