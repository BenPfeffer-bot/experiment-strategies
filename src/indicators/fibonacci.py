"""
Enhanced Fibonacci Trailing Stop Indicator

This module implements an advanced Fibonacci Trailing Stop indicator with dynamic pivot detection
and adaptive trailing stops. It uses Fibonacci ratios to calculate support/resistance levels
and generates trading signals based on price action and pivot points.

Features:
- Dynamic pivot detection
- Multiple Fibonacci ratio support
- Adaptive trailing stops
- Price trigger options
- Signal generation
- Advanced visualization
- Memory-based optimization

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import logging
from datetime import datetime
from enum import Enum

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


class TriggerType(Enum):
    """Types of price triggers for stop adjustments"""

    CLOSE = "close"  # Use close price for trigger
    WICK = "wick"  # Use high/low prices based on trend


class FibLevel(Enum):
    """Common Fibonacci retracement/extension levels"""

    EXT_2618 = 2.618
    EXT_1618 = 1.618
    EXT_1000 = 1.000
    RET_0618 = 0.618
    RET_0500 = 0.500
    RET_0382 = 0.382
    RET_0236 = 0.236


@dataclass
class FibParameters:
    """Parameters for Fibonacci calculations"""

    left_window: int = 20  # Left window for pivot detection
    right_window: int = 1  # Right window for pivot detection
    use_custom_fib: bool = False  # Whether to use custom Fibonacci level
    custom_fib: float = 1.618  # Custom Fibonacci multiplier
    default_fib: float = -0.382  # Default Fibonacci level
    trigger_type: TriggerType = TriggerType.CLOSE
    price_column: str = "close"
    high_column: str = "high"
    low_column: str = "low"
    signal_threshold: float = 0.02  # Minimum distance for signal generation


@dataclass
class Pivot:
    """Container for pivot point information"""

    bar_index: int
    price: float
    direction: int  # 1 for high pivot, -1 for low pivot
    strength: float = 1.0  # Pivot strength based on surrounding price action


@dataclass
class FibResult:
    """Container for Fibonacci calculation results"""

    trailing_stop: pd.Series
    other_side: pd.Series
    direction: pd.Series
    pivots: pd.Series
    signals: pd.Series


class FibonacciTrailingStop:
    """
    Enhanced Fibonacci Trailing Stop Indicator Implementation.

    This indicator detects pivot points and uses Fibonacci ratios to generate
    adaptive trailing stops and trading signals. Features include:
    - Dynamic pivot detection with strength scoring
    - Multiple Fibonacci ratio support
    - Adaptive trailing stops with price triggers
    - Signal generation based on price action
    - Advanced visualization options

    Attributes:
        params (FibParameters): Configuration parameters
        logger (logging.Logger): Logger instance
        results (Optional[FibResult]): Computed results
    """

    def __init__(self, params: Optional[FibParameters] = None):
        """
        Initialize Fibonacci indicator with given parameters.

        Args:
            params: Optional[FibParameters]
                Parameters for calculations. If None, uses default values.
        """
        self.params = params if params else FibParameters()
        self._validate_parameters()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize results container
        self.results: Optional[FibResult] = None

        # Initialize pivot storage
        self.pivots: List[Pivot] = []

        # Calculate secondary window sizes
        self.secondary_left = max(1, math.ceil(self.params.left_window / 2))
        self.secondary_right = max(1, math.ceil(self.params.right_window / 2))

    def _validate_parameters(self) -> None:
        """Validate Fibonacci parameters."""
        if self.params.left_window < 2:
            raise ValueError("left_window must be at least 2")
        if self.params.right_window < 1:
            raise ValueError("right_window must be at least 1")
        if self.params.custom_fib <= 0:
            raise ValueError("custom_fib must be positive")

    def _is_pivot_high(self, idx: int, L: int, R: int, highs: np.ndarray) -> bool:
        """
        Check if price at index is a pivot high.

        Args:
            idx: int
                Index to check
            L: int
                Left window size
            R: int
                Right window size
            highs: np.ndarray
                Array of high prices

        Returns:
            bool indicating if index is a pivot high
        """
        if idx < L or idx > len(highs) - R - 1:
            return False
        window = highs[idx - L : idx + R + 1]
        return highs[idx] == max(window)

    def _is_pivot_low(self, idx: int, L: int, R: int, lows: np.ndarray) -> bool:
        """
        Check if price at index is a pivot low.

        Args:
            idx: int
                Index to check
            L: int
                Left window size
            R: int
                Right window size
            lows: np.ndarray
                Array of low prices

        Returns:
            bool indicating if index is a pivot low
        """
        if idx < L or idx > len(lows) - R - 1:
            return False
        window = lows[idx - L : idx + R + 1]
        return lows[idx] == min(window)

    def _calculate_pivot_strength(
        self, idx: int, price: float, direction: int, prices: np.ndarray
    ) -> float:
        """
        Calculate pivot strength based on surrounding price action.

        Args:
            idx: int
                Pivot index
            price: float
                Pivot price
            direction: int
                Pivot direction (1 for high, -1 for low)
            prices: np.ndarray
                Array of prices

        Returns:
            float between 0 and 1 indicating pivot strength
        """
        try:
            # Get surrounding prices
            left_prices = prices[max(0, idx - self.params.left_window) : idx]
            right_prices = prices[
                idx + 1 : min(len(prices), idx + self.params.right_window + 1)
            ]

            if len(left_prices) == 0 or len(right_prices) == 0:
                return 0.5

            # Calculate price differences
            left_diff = abs(price - left_prices).mean()
            right_diff = abs(price - right_prices).mean()

            # Calculate strength based on price differences
            total_diff = left_diff + right_diff
            if total_diff == 0:
                return 0.5

            # Higher difference means stronger pivot
            strength = min(1.0, (total_diff / price) * 10)

            return strength

        except Exception as e:
            self.logger.warning(f"Error calculating pivot strength: {str(e)}")
            return 0.5

    def compute(self, data: Union[pd.Series, pd.DataFrame]) -> FibResult:
        """
        Compute Fibonacci levels and trailing stops.

        Args:
            data: Union[pd.Series, pd.DataFrame]
                Price data. If DataFrame, uses columns specified in params

        Returns:
            FibResult containing computed values

        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Extract price series
            if isinstance(data, pd.DataFrame):
                required_columns = [
                    self.params.price_column,
                    self.params.high_column,
                    self.params.low_column,
                ]
                missing_columns = [
                    col for col in required_columns if col not in data.columns
                ]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")

                closes = data[self.params.price_column].values
                highs = data[self.params.high_column].values
                lows = data[self.params.low_column].values
            else:
                # If Series, use it for all prices
                closes = highs = lows = data.values

            n_bars = len(closes)

            # Initialize result arrays
            trailing_stop = np.full(n_bars, np.nan)
            other_side = np.full(n_bars, np.nan)
            direction = np.zeros(n_bars, dtype=int)
            pivot_points = np.full(n_bars, np.nan)
            signals = np.zeros(n_bars, dtype=int)

            # Initialize variables
            current_direction = 0
            stop_level = closes[0]
            swing_max = highs[0]
            swing_min = lows[0]
            other_level = stop_level

            # Precompute secondary pivots
            secondary_highs = np.full(n_bars, np.nan)
            secondary_lows = np.full(n_bars, np.nan)

            for i in range(self.secondary_left, n_bars - self.secondary_right):
                if self._is_pivot_high(
                    i, self.secondary_left, self.secondary_right, highs
                ):
                    secondary_highs[i] = highs[i]
                if self._is_pivot_low(
                    i, self.secondary_left, self.secondary_right, lows
                ):
                    secondary_lows[i] = lows[i]

            # Process each bar
            for t in range(n_bars):
                # Check for primary pivots
                if self._is_pivot_high(
                    t, self.params.left_window, self.params.right_window, highs
                ):
                    pivot_bar = t - self.params.right_window
                    pivot_price = highs[pivot_bar]
                    strength = self._calculate_pivot_strength(
                        pivot_bar, pivot_price, 1, highs
                    )

                    if self.pivots:
                        first = self.pivots[0]
                        if first.direction == 1 and pivot_price > first.price:
                            first.bar_index = pivot_bar
                            first.price = pivot_price
                            first.strength = strength
                        elif first.direction == -1 and pivot_price > first.price:
                            self.pivots.insert(
                                0, Pivot(pivot_bar, pivot_price, 1, strength)
                            )
                    else:
                        self.pivots.insert(
                            0, Pivot(pivot_bar, pivot_price, 1, strength)
                        )

                    pivot_points[pivot_bar] = pivot_price

                if self._is_pivot_low(
                    t, self.params.left_window, self.params.right_window, lows
                ):
                    pivot_bar = t - self.params.right_window
                    pivot_price = lows[pivot_bar]
                    strength = self._calculate_pivot_strength(
                        pivot_bar, pivot_price, -1, lows
                    )

                    if self.pivots:
                        first = self.pivots[0]
                        if first.direction == -1 and pivot_price < first.price:
                            first.bar_index = pivot_bar
                            first.price = pivot_price
                            first.strength = strength
                        elif first.direction == 1 and pivot_price < first.price:
                            self.pivots.insert(
                                0, Pivot(pivot_bar, pivot_price, -1, strength)
                            )
                    else:
                        self.pivots.insert(
                            0, Pivot(pivot_bar, pivot_price, -1, strength)
                        )

                    pivot_points[pivot_bar] = pivot_price

                # Process pivots if we have at least two
                if len(self.pivots) >= 2:
                    p0, p1 = self.pivots[:2]
                    current_max = max(p0.price, p1.price)
                    current_min = min(p0.price, p1.price)

                    # Initialize levels for first two pivots
                    if len(self.pivots) == 2:
                        stop_level = (current_max + current_min) / 2.0
                        other_level = stop_level

                    # Calculate Fibonacci adjustments
                    price_range = current_max - current_min
                    fib_level = (
                        self.params.custom_fib
                        if self.params.use_custom_fib
                        else self.params.default_fib
                    )

                    max_adjusted = current_max + price_range * fib_level
                    min_adjusted = current_min - price_range * fib_level

                    # Determine swing direction
                    swing_direction = (
                        -price_range if p0.price < p1.price else price_range
                    )

                    # Get trigger price
                    if self.params.trigger_type == TriggerType.CLOSE:
                        trigger_price = closes[t]
                    else:
                        if current_direction < 1:
                            trigger_price = highs[t]
                        elif current_direction > -1:
                            trigger_price = lows[t]
                        else:
                            trigger_price = closes[t]

                    # Update trailing stop
                    if current_direction < 1:
                        if trigger_price > stop_level:
                            stop_level = min_adjusted
                            current_direction = 1
                            signals[t] = 1  # Buy signal
                        else:
                            stop_level = min(stop_level, max_adjusted)

                    if current_direction > -1:
                        if trigger_price < stop_level:
                            stop_level = max_adjusted
                            current_direction = -1
                            signals[t] = -1  # Sell signal
                        else:
                            stop_level = max(stop_level, min_adjusted)

                    # Update other side using secondary pivots
                    if current_direction < 1:
                        if not np.isnan(secondary_highs[t]):
                            other_level = min(
                                other_level, secondary_highs[t], stop_level
                            )
                        else:
                            other_level = min(other_level, stop_level)
                    elif current_direction > -1:
                        if not np.isnan(secondary_lows[t]):
                            other_level = max(
                                other_level, secondary_lows[t], stop_level
                            )
                        else:
                            other_level = max(other_level, stop_level)

                # Store results
                trailing_stop[t] = stop_level
                other_side[t] = other_level
                direction[t] = current_direction

            # Create result series
            index = data.index if isinstance(data, (pd.Series, pd.DataFrame)) else None
            self.results = FibResult(
                trailing_stop=pd.Series(
                    trailing_stop, index=index, name="Trailing Stop"
                ),
                other_side=pd.Series(other_side, index=index, name="Other Side"),
                direction=pd.Series(direction, index=index, name="Direction"),
                pivots=pd.Series(pivot_points, index=index, name="Pivots"),
                signals=pd.Series(signals, index=index, name="Signals"),
            )

            return self.results

        except Exception as e:
            self.logger.error(f"Error computing Fibonacci levels: {str(e)}")
            raise

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current indicator state.

        Returns:
            Dict containing current values for stops and direction
        """
        if self.results is None:
            raise ValueError("Fibonacci levels not computed. Call compute() first.")

        return {
            "trailing_stop": float(self.results.trailing_stop.iloc[-1]),
            "other_side": float(self.results.other_side.iloc[-1]),
            "direction": int(self.results.direction.iloc[-1]),
            "last_pivot": float(self.pivots[0].price) if self.pivots else np.nan,
            "pivot_strength": float(self.pivots[0].strength) if self.pivots else np.nan,
        }

    def plot(
        self,
        price_data: Optional[pd.Series] = None,
        show_signals: bool = True,
        show_pivots: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot Fibonacci indicator with price data.

        Args:
            price_data: Optional[pd.Series]
                Price data to plot
            show_signals: bool
                If True, shows trading signals
            show_pivots: bool
                If True, shows pivot points
            save_path: Optional[str]
                If provided, saves plot to this path
            title: Optional[str]
                Custom plot title
        """
        if self.results is None:
            raise ValueError("Fibonacci levels not computed. Call compute() first.")

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot price data
            if price_data is not None:
                ax.plot(
                    price_data.index,
                    price_data,
                    label="Price",
                    color="black",
                    linewidth=1,
                )

            # Plot trailing stop
            ax.plot(
                self.results.trailing_stop.index,
                self.results.trailing_stop,
                label="Trailing Stop",
                color="blue",
                linewidth=1.5,
            )

            # Plot other side
            ax.plot(
                self.results.other_side.index,
                self.results.other_side,
                label="Other Side",
                color="gray",
                linestyle="--",
                alpha=0.5,
            )

            # Plot pivots if requested
            if show_pivots and price_data is not None:
                pivot_mask = ~np.isnan(self.results.pivots)
                pivot_points = price_data[pivot_mask]

                ax.scatter(
                    pivot_points.index,
                    pivot_points,
                    color="purple",
                    marker="o",
                    s=50,
                    label="Pivot Points",
                )

            # Plot signals if requested
            if show_signals and price_data is not None:
                signals = self.results.signals
                buy_points = price_data[signals == 1]
                sell_points = price_data[signals == -1]

                ax.scatter(
                    buy_points.index,
                    buy_points,
                    color="green",
                    marker="^",
                    s=100,
                    label="Buy Signal",
                )
                ax.scatter(
                    sell_points.index,
                    sell_points,
                    color="red",
                    marker="v",
                    s=100,
                    label="Sell Signal",
                )

            # Customize plot
            ax.set_title(title or "Fibonacci Trailing Stop")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            plt.close()
            raise ValueError(f"Error plotting Fibonacci indicator: {str(e)}")


if __name__ == "__main__":
    """Example usage of the Fibonacci indicator"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create output directory
        fib_output_dir = os.path.join(FIGURES_DIR, "fibonacci")
        os.makedirs(fib_output_dir, exist_ok=True)

        # Process each ticker
        for ticker in TICKERS:
            try:
                logger.info(f"Processing Fibonacci indicator for {ticker}")

                # Load data
                data_path = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
                df = pd.read_csv(data_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

                # Initialize indicator with custom parameters
                params = FibParameters(
                    left_window=20,
                    right_window=1,
                    use_custom_fib=True,
                    custom_fib=1.618,
                    trigger_type=TriggerType.CLOSE,
                    price_column="Close",
                    high_column="High",
                    low_column="Low",
                    signal_threshold=0.02,
                )
                fib_indicator = FibonacciTrailingStop(params)

                # Compute indicator
                results = fib_indicator.compute(df)

                # Get current state
                current_state = fib_indicator.get_current_state()
                logger.info(
                    f"{ticker} current stop: {current_state['trailing_stop']:.2f} "
                    f"(Direction: {current_state['direction']}, "
                    f"Pivot Strength: {current_state['pivot_strength']:.2%})"
                )

                # Plot indicator
                save_path = os.path.join(fib_output_dir, f"{ticker}_fibonacci.png")
                fib_indicator.plot(
                    price_data=df["Close"],
                    show_signals=True,
                    show_pivots=True,
                    save_path=save_path,
                    title=f"Fibonacci Trailing Stop - {ticker}",
                )

                logger.info(f"Fibonacci plot saved for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
