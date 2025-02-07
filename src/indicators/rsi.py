"""
Multi-Length RSI (Relative Strength Index) Indicator

This module implements an enhanced Multi-Length RSI indicator inspired by LuxAlgo.
It computes an average RSI over a range of lookback lengths while tracking overbought/oversold
conditions and generating adaptive channels.

Features:
- Multiple lookback period support
- Adaptive upper and lower channels
- Overbought/Oversold tracking
- Signal generation
- Customizable smoothing
- Advanced visualization

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
import logging
from datetime import datetime

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


@dataclass
class RSIParameters:
    """Parameters for RSI calculation"""

    max_length: int = 20
    min_length: int = 10
    overbought: float = 80
    oversold: float = 20
    smoothing_factor: float = 0.1  # For channel smoothing
    price_column: str = "close"
    signal_threshold: float = 5.0  # Minimum difference for signal generation


@dataclass
class RSIResult:
    """Container for RSI calculation results"""

    avg_rsi: pd.Series
    buy_rsi: pd.Series
    sell_rsi: pd.Series
    overbought_pct: pd.Series
    oversold_pct: pd.Series


class RSIMultiLength:
    """
    Enhanced Multi-Length RSI Indicator Implementation.

    This indicator computes RSI values over multiple lookback periods and combines them
    to create a more robust signal. It includes:
    - Average RSI across multiple timeframes
    - Adaptive upper and lower channels
    - Overbought/Oversold percentage tracking
    - Signal generation based on channel crossovers

    Attributes:
        params (RSIParameters): Configuration parameters
        logger (logging.Logger): Logger instance
        results (Optional[RSIResult]): Computed results
    """

    def __init__(self, params: Optional[RSIParameters] = None):
        """
        Initialize RSI indicator with given parameters.

        Args:
            params: Optional[RSIParameters]
                Parameters for RSI calculation. If None, uses default values.
        """
        self.params = params if params else RSIParameters()
        self._validate_parameters()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize results container
        self.results: Optional[RSIResult] = None

        # Number of RSI lengths used
        self.N = self.params.max_length - self.params.min_length + 1

    def _validate_parameters(self) -> None:
        """Validate RSI parameters."""
        if self.params.max_length <= self.params.min_length:
            raise ValueError("max_length must be greater than min_length")
        if self.params.min_length < 2:
            raise ValueError("min_length must be at least 2")
        if not 0 <= self.params.overbought <= 100:
            raise ValueError("overbought must be between 0 and 100")
        if not 0 <= self.params.oversold <= 100:
            raise ValueError("oversold must be between 0 and 100")
        if self.params.oversold >= self.params.overbought:
            raise ValueError("oversold must be less than overbought")
        if not 0 < self.params.smoothing_factor <= 1:
            raise ValueError("smoothing_factor must be between 0 and 1")

    def compute(self, data: Union[pd.Series, pd.DataFrame]) -> RSIResult:
        """
        Compute multi-length RSI and channels.

        Args:
            data: Union[pd.Series, pd.DataFrame]
                Price data. If DataFrame, uses the column specified in params.price_column

        Returns:
            RSIResult containing computed values

        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Extract price series
            if isinstance(data, pd.DataFrame):
                if self.params.price_column not in data.columns:
                    raise ValueError(
                        f"Price column '{self.params.price_column}' not found"
                    )
                price_series = data[self.params.price_column]
            else:
                price_series = data

            # Validate data
            if len(price_series) < self.params.max_length:
                raise ValueError(
                    f"Insufficient data points. Need at least {self.params.max_length}"
                )

            n_bars = len(price_series)

            # Initialize arrays for RMA calculations
            num = np.zeros(self.N)  # Numerator for each RSI length
            den = np.zeros(self.N)  # Denominator for each RSI length

            # Output arrays
            avg_rsi_list = []
            buy_rsi_list = []
            sell_rsi_list = []
            overbought_pct_list = []
            oversold_pct_list = []

            # Channel smoothing state
            prev_buy_rsi = None
            prev_sell_rsi = None

            # Process each bar
            for t in range(n_bars):
                diff = (
                    0.0 if t == 0 else price_series.iloc[t] - price_series.iloc[t - 1]
                )

                overbuy = oversell = 0
                sum_rsi = 0.0

                # Calculate RSI for each length
                for k in range(self.N):
                    period = self.params.min_length + k
                    alpha = 1.0 / period

                    # Update RMA values
                    num[k] = alpha * diff + (1 - alpha) * num[k]
                    den[k] = alpha * abs(diff) + (1 - alpha) * den[k]

                    # Calculate RSI
                    rsi_val = 50.0 if den[k] == 0 else 50 * (num[k] / den[k]) + 50
                    sum_rsi += rsi_val

                    # Track overbought/oversold conditions
                    if rsi_val > self.params.overbought:
                        overbuy += 1
                    if rsi_val < self.params.oversold:
                        oversell += 1

                # Calculate average RSI
                avg_rsi = sum_rsi / self.N

                # Update channels with smoothing
                if prev_buy_rsi is None:
                    buy_rsi = sell_rsi = avg_rsi
                else:
                    # Apply adaptive smoothing
                    ob_weight = (overbuy / self.N) * self.params.smoothing_factor
                    os_weight = (oversell / self.N) * self.params.smoothing_factor

                    buy_rsi = prev_buy_rsi + ob_weight * (avg_rsi - prev_buy_rsi)
                    sell_rsi = prev_sell_rsi + os_weight * (avg_rsi - prev_sell_rsi)

                # Store results
                avg_rsi_list.append(avg_rsi)
                buy_rsi_list.append(buy_rsi)
                sell_rsi_list.append(sell_rsi)
                overbought_pct_list.append((overbuy / self.N) * 100)
                oversold_pct_list.append((oversell / self.N) * 100)

                # Update previous values
                prev_buy_rsi = buy_rsi
                prev_sell_rsi = sell_rsi

            # Create result series
            index = price_series.index
            self.results = RSIResult(
                avg_rsi=pd.Series(avg_rsi_list, index=index, name="Average RSI"),
                buy_rsi=pd.Series(buy_rsi_list, index=index, name="Upper Channel"),
                sell_rsi=pd.Series(sell_rsi_list, index=index, name="Lower Channel"),
                overbought_pct=pd.Series(
                    overbought_pct_list, index=index, name="Overbought %"
                ),
                oversold_pct=pd.Series(
                    oversold_pct_list, index=index, name="Oversold %"
                ),
            )

            return self.results

        except Exception as e:
            self.logger.error(f"Error computing RSI: {str(e)}")
            raise

    def generate_signals(self, threshold: Optional[float] = None) -> pd.Series:
        """
        Generate trading signals based on RSI values and channels.

        Args:
            threshold: Optional[float]
                Minimum difference for signal generation. If None, uses params.signal_threshold

        Returns:
            pd.Series with values:
                1 for buy signals
                -1 for sell signals
                0 for no signal
        """
        if self.results is None:
            raise ValueError("RSI not computed. Call compute() first.")

        threshold = threshold if threshold is not None else self.params.signal_threshold
        signals = pd.Series(0, index=self.results.avg_rsi.index)

        # Generate signals based on channel crossovers and RSI levels
        avg_rsi = self.results.avg_rsi
        buy_rsi = self.results.buy_rsi
        sell_rsi = self.results.sell_rsi

        # Buy conditions
        buy_conditions = (
            (avg_rsi < self.params.oversold)  # RSI oversold
            & (avg_rsi > sell_rsi)  # RSI above lower channel
            & (self.results.oversold_pct > threshold)  # Significant oversold percentage
        )
        signals[buy_conditions] = 1

        # Sell conditions
        sell_conditions = (
            (avg_rsi > self.params.overbought)  # RSI overbought
            & (avg_rsi < buy_rsi)  # RSI below upper channel
            & (
                self.results.overbought_pct > threshold
            )  # Significant overbought percentage
        )
        signals[sell_conditions] = -1

        return signals

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current indicator state.

        Returns:
            Dict containing current values for RSI, channels, and percentages
        """
        if self.results is None:
            raise ValueError("RSI not computed. Call compute() first.")

        return {
            "rsi": float(self.results.avg_rsi.iloc[-1]),
            "upper_channel": float(self.results.buy_rsi.iloc[-1]),
            "lower_channel": float(self.results.sell_rsi.iloc[-1]),
            "overbought_pct": float(self.results.overbought_pct.iloc[-1]),
            "oversold_pct": float(self.results.oversold_pct.iloc[-1]),
        }

    def plot(
        self,
        price_data: Optional[pd.Series] = None,
        show_price: bool = True,
        show_signals: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot RSI indicator with optional price data and signals.

        Args:
            price_data: Optional[pd.Series]
                Price data to plot alongside RSI
            show_price: bool
                If True and price_data provided, shows price subplot
            show_signals: bool
                If True, shows signal markers
            save_path: Optional[str]
                If provided, saves plot to this path
            title: Optional[str]
                Custom plot title
        """
        if self.results is None:
            raise ValueError("RSI not computed. Call compute() first.")

        try:
            # Create figure with subplots
            n_plots = 1 + (1 if show_price and price_data is not None else 0)
            fig, axes = plt.subplots(
                n_plots, 1, figsize=(12, 8 * n_plots), height_ratios=[2] * n_plots
            )

            if n_plots == 1:
                axes = [axes]

            # Plot price if available
            if show_price and price_data is not None:
                ax_price = axes[0]
                ax_price.plot(price_data.index, price_data, label="Price", color="blue")
                ax_price.set_title("Price" if title is None else title)
                ax_price.grid(True)
                ax_price.legend()

                # Plot signals on price chart if requested
                if show_signals:
                    signals = self.generate_signals()
                    buy_points = price_data[signals == 1]
                    sell_points = price_data[signals == -1]

                    ax_price.scatter(
                        buy_points.index,
                        buy_points,
                        color="green",
                        marker="^",
                        s=100,
                        label="Buy",
                    )
                    ax_price.scatter(
                        sell_points.index,
                        sell_points,
                        color="red",
                        marker="v",
                        s=100,
                        label="Sell",
                    )

            # Plot RSI
            ax_rsi = axes[-1]

            # Plot channels and average RSI
            ax_rsi.plot(
                self.results.avg_rsi.index,
                self.results.avg_rsi,
                label="RSI",
                color="purple",
                linewidth=1.5,
            )
            ax_rsi.plot(
                self.results.buy_rsi.index,
                self.results.buy_rsi,
                label="Upper Channel",
                color="green",
                linestyle="--",
            )
            ax_rsi.plot(
                self.results.sell_rsi.index,
                self.results.sell_rsi,
                label="Lower Channel",
                color="red",
                linestyle="--",
            )

            # Fill between channels
            ax_rsi.fill_between(
                self.results.avg_rsi.index,
                self.results.buy_rsi,
                self.results.sell_rsi,
                color="lavender",
                alpha=0.3,
            )

            # Add overbought/oversold levels
            ax_rsi.axhline(
                self.params.overbought, color="red", linestyle=":", alpha=0.5
            )
            ax_rsi.axhline(
                self.params.oversold, color="green", linestyle=":", alpha=0.5
            )
            ax_rsi.axhline(50, color="gray", linestyle=":", alpha=0.5)

            # Add percentage annotations
            last_idx = self.results.avg_rsi.index[-1]
            ax_rsi.text(
                last_idx,
                90,
                f"Overbought: {self.results.overbought_pct.iloc[-1]:.1f}%",
                color="red",
                va="center",
            )
            ax_rsi.text(
                last_idx,
                10,
                f"Oversold: {self.results.oversold_pct.iloc[-1]:.1f}%",
                color="green",
                va="center",
            )

            ax_rsi.set_title("Multi-Length RSI" if title is None else title)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True)
            ax_rsi.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            plt.close()
            raise ValueError(f"Error plotting RSI: {str(e)}")


if __name__ == "__main__":
    """Example usage of the RSI indicator"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create output directory
        rsi_output_dir = os.path.join(FIGURES_DIR, "rsi")
        os.makedirs(rsi_output_dir, exist_ok=True)

        # Process each ticker
        for ticker in TICKERS:
            try:
                logger.info(f"Processing RSI for {ticker}")

                # Load data
                data_path = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
                df = pd.read_csv(data_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

                # Initialize RSI with custom parameters
                params = RSIParameters(
                    max_length=20,
                    min_length=10,
                    overbought=80,
                    oversold=20,
                    smoothing_factor=0.1,
                    price_column="Close",
                    signal_threshold=5.0,
                )
                rsi_indicator = RSIMultiLength(params)

                # Compute RSI
                results = rsi_indicator.compute(df)

                # Generate signals
                signals = rsi_indicator.generate_signals()

                # Get current state
                current_state = rsi_indicator.get_current_state()
                logger.info(f"{ticker} current RSI: {current_state['rsi']:.2f}")

                # Plot RSI with price
                save_path = os.path.join(rsi_output_dir, f"{ticker}_rsi.png")
                rsi_indicator.plot(
                    price_data=df["Close"],
                    show_price=True,
                    show_signals=True,
                    save_path=save_path,
                    title=f"Multi-Length RSI - {ticker}",
                )

                logger.info(f"RSI plot saved for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
