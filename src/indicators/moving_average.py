"""
Enhanced Moving Average Indicator with Forecasting

This module implements an advanced Moving Average indicator with forecasting capabilities.
It supports multiple types of moving averages (SMA, EMA, WMA) and provides price level
forecasting using percentile-based interpolation of historical deviations.

Features:
- Multiple MA types (Simple, Exponential, Weighted)
- Price level forecasting
- Trend detection and analysis
- Adaptive memory management
- Signal generation
- Advanced visualization

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import FIGURES_DIR, TICKERS, RAW_DATA_DIR


class MAType(Enum):
    """Types of Moving Averages"""

    SIMPLE = "simple"
    EXPONENTIAL = "exponential"
    WEIGHTED = "weighted"


@dataclass
class MAParameters:
    """Parameters for Moving Average calculation"""

    window: int = 20
    ma_type: MAType = MAType.SIMPLE
    price_column: str = "close"
    max_memory: int = 50
    forecast_length: int = 100
    up_percentile: float = 80
    mid_percentile: float = 50
    down_percentile: float = 20
    signal_threshold: float = 0.02  # 2% threshold for signal generation


@dataclass
class ForecastLevels:
    """Container for forecast price levels"""

    upper: float
    mid: float
    lower: float
    confidence: float  # Confidence score based on memory size and consistency


@dataclass
class MAResult:
    """Container for Moving Average calculation results"""

    ma: pd.Series
    trend: pd.Series
    forecasts: pd.DataFrame
    signals: pd.Series


class MovingAverageForecast:
    """
    Enhanced Moving Average Indicator Implementation.

    This indicator computes various types of moving averages and provides
    price forecasting based on trend analysis and historical deviations.

    Features:
    - Multiple MA types (SMA, EMA, WMA)
    - Trend detection
    - Price level forecasting
    - Signal generation
    - Memory-based learning

    Attributes:
        params (MAParameters): Configuration parameters
        logger (logging.Logger): Logger instance
        results (Optional[MAResult]): Computed results
    """

    def __init__(self, params: Optional[MAParameters] = None):
        """
        Initialize Moving Average indicator with given parameters.

        Args:
            params: Optional[MAParameters]
                Parameters for MA calculation. If None, uses default values.
        """
        self.params = params if params else MAParameters()
        self._validate_parameters()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize results container
        self.results: Optional[MAResult] = None

        # Initialize memory containers
        self.memory = {"up": [], "down": []}
        self.trend_prices = {"up": None, "down": None}

    def _validate_parameters(self) -> None:
        """Validate Moving Average parameters."""
        if self.params.window < 2:
            raise ValueError("window must be at least 2")
        if self.params.max_memory < self.params.window:
            raise ValueError("max_memory must be at least as large as window")
        if not 0 <= self.params.up_percentile <= 100:
            raise ValueError("up_percentile must be between 0 and 100")
        if not 0 <= self.params.down_percentile <= 100:
            raise ValueError("down_percentile must be between 0 and 100")
        if not 0 <= self.params.mid_percentile <= 100:
            raise ValueError("mid_percentile must be between 0 and 100")
        if self.params.down_percentile >= self.params.mid_percentile:
            raise ValueError("down_percentile must be less than mid_percentile")
        if self.params.mid_percentile >= self.params.up_percentile:
            raise ValueError("mid_percentile must be less than up_percentile")

    def _compute_ma(self, data: pd.Series) -> pd.Series:
        """
        Compute moving average based on specified type.

        Args:
            data: pd.Series
                Price data

        Returns:
            pd.Series containing the moving average values
        """
        if self.params.ma_type == MAType.SIMPLE:
            return data.rolling(window=self.params.window, min_periods=1).mean()
        elif self.params.ma_type == MAType.EXPONENTIAL:
            return data.ewm(span=self.params.window, adjust=False).mean()
        else:  # Weighted
            weights = np.arange(1, self.params.window + 1)
            return data.rolling(window=self.params.window, min_periods=1).apply(
                lambda x: np.sum(weights * x) / np.sum(weights)
            )

    def _update_memory(self, trend: str, price: float, ref_price: float) -> None:
        """
        Update trend memory with new price deviation.

        Args:
            trend: str
                Current trend ('up' or 'down')
            price: float
                Current price
            ref_price: float
                Reference price for deviation calculation
        """
        delta = price - ref_price
        self.memory[trend].append(delta)

        # Maintain memory size
        if len(self.memory[trend]) > self.params.max_memory:
            self.memory[trend].pop(0)

    def _compute_forecast(
        self, trend: str, ref_price: float
    ) -> Optional[ForecastLevels]:
        """
        Compute forecast levels for current trend.

        Args:
            trend: str
                Current trend
            ref_price: float
                Reference price

        Returns:
            ForecastLevels if enough data available, None otherwise
        """
        if len(self.memory[trend]) < 3:  # Need minimum data points
            return None

        deltas = np.array(self.memory[trend])

        # Calculate forecast levels
        upper = ref_price + np.percentile(deltas, self.params.up_percentile)
        mid = ref_price + np.percentile(deltas, self.params.mid_percentile)
        lower = ref_price + np.percentile(deltas, self.params.down_percentile)

        # Calculate confidence based on memory size and consistency
        memory_score = min(len(deltas) / self.params.max_memory, 1.0)
        consistency = 1 - (np.std(deltas) / (upper - lower) if upper != lower else 0)
        confidence = (memory_score + consistency) / 2

        return ForecastLevels(upper=upper, mid=mid, lower=lower, confidence=confidence)

    def compute(self, data: Union[pd.Series, pd.DataFrame]) -> MAResult:
        """
        Compute moving average and generate forecasts.

        Args:
            data: Union[pd.Series, pd.DataFrame]
                Price data. If DataFrame, uses column specified in params.price_column

        Returns:
            MAResult containing computed values

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
            if len(price_series) < self.params.window:
                raise ValueError(
                    f"Insufficient data points. Need at least {self.params.window}"
                )

            # Compute moving average
            ma_series = self._compute_ma(price_series)

            # Initialize result containers
            trends = pd.Series(index=price_series.index, dtype=str)
            forecasts = pd.DataFrame(
                index=price_series.index,
                columns=["upper", "mid", "lower", "confidence"],
            )
            signals = pd.Series(0, index=price_series.index)

            # Process each point
            for i in range(len(price_series)):
                price = price_series.iloc[i]
                ma = ma_series.iloc[i]

                # Determine trend
                current_trend = "up" if price > ma else "down"
                trends.iloc[i] = current_trend

                # Check for trend change
                trend_changed = i > 0 and trends.iloc[i] != trends.iloc[i - 1]

                if trend_changed:
                    self.trend_prices[current_trend] = price
                    self.memory[current_trend] = []

                # Update memory
                if not np.isnan(ma):
                    ref_price = self.trend_prices[current_trend]
                    if ref_price is not None:
                        self._update_memory(current_trend, price, ref_price)

                        # Generate forecast
                        forecast = self._compute_forecast(current_trend, ref_price)
                        if forecast:
                            forecasts.iloc[i] = [
                                forecast.upper,
                                forecast.mid,
                                forecast.lower,
                                forecast.confidence,
                            ]

                            # Generate signals based on price position relative to forecast
                            if current_trend == "up" and price >= forecast.upper:
                                signals.iloc[i] = -1  # Sell signal
                            elif current_trend == "down" and price <= forecast.lower:
                                signals.iloc[i] = 1  # Buy signal

            # Store results
            self.results = MAResult(
                ma=ma_series, trend=trends, forecasts=forecasts, signals=signals
            )

            return self.results

        except Exception as e:
            self.logger.error(f"Error computing Moving Average: {str(e)}")
            raise

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current indicator state.

        Returns:
            Dict containing current MA value, trend, and forecast levels
        """
        if self.results is None:
            raise ValueError("Moving Average not computed. Call compute() first.")

        return {
            "ma": float(self.results.ma.iloc[-1]),
            "trend": self.results.trend.iloc[-1],
            "forecast_upper": float(self.results.forecasts.iloc[-1]["upper"]),
            "forecast_mid": float(self.results.forecasts.iloc[-1]["mid"]),
            "forecast_lower": float(self.results.forecasts.iloc[-1]["lower"]),
            "confidence": float(self.results.forecasts.iloc[-1]["confidence"]),
        }

    def plot(
        self,
        price_data: Optional[pd.Series] = None,
        show_signals: bool = True,
        show_forecasts: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot Moving Average indicator with forecasts.

        Args:
            price_data: Optional[pd.Series]
                Price data to plot
            show_signals: bool
                If True, shows trading signals
            show_forecasts: bool
                If True, shows forecast levels
            save_path: Optional[str]
                If provided, saves plot to this path
            title: Optional[str]
                Custom plot title
        """
        if self.results is None:
            raise ValueError("Moving Average not computed. Call compute() first.")

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot price and MA
            if price_data is not None:
                ax.plot(
                    price_data.index,
                    price_data,
                    label="Price",
                    color="black",
                    linewidth=1,
                )
            ax.plot(
                self.results.ma.index,
                self.results.ma,
                label=f"{self.params.ma_type.value.title()} MA ({self.params.window})",
                color="blue",
                linewidth=1.5,
            )

            # Plot forecasts if requested
            if show_forecasts:
                forecasts = self.results.forecasts

                # Plot forecast levels with confidence-based alpha
                for i in range(len(forecasts)):
                    if not np.isnan(forecasts.iloc[i]["upper"]):
                        confidence = forecasts.iloc[i]["confidence"]
                        alpha = max(0.2, confidence)

                        # Plot forecast ranges
                        ax.fill_between(
                            [forecasts.index[i]],
                            [forecasts.iloc[i]["lower"]],
                            [forecasts.iloc[i]["upper"]],
                            color="gray",
                            alpha=alpha * 0.3,
                        )
                        ax.plot(
                            [forecasts.index[i]],
                            [forecasts.iloc[i]["mid"]],
                            color="orange",
                            alpha=alpha,
                            linestyle="--",
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
            ax.set_title(
                title
                or f"Moving Average Forecast ({self.params.ma_type.value.title()})"
            )
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
            raise ValueError(f"Error plotting Moving Average: {str(e)}")


if __name__ == "__main__":
    """Example usage of the Moving Average indicator"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create output directory
        ma_output_dir = os.path.join(FIGURES_DIR, "moving_average")
        os.makedirs(ma_output_dir, exist_ok=True)

        # Process each ticker
        for ticker in TICKERS:
            try:
                logger.info(f"Processing Moving Average for {ticker}")

                # Load data
                data_path = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
                df = pd.read_csv(data_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

                # Initialize indicator with custom parameters
                params = MAParameters(
                    window=20,
                    ma_type=MAType.EXPONENTIAL,
                    price_column="Close",
                    max_memory=50,
                    forecast_length=100,
                    up_percentile=80,
                    mid_percentile=50,
                    down_percentile=20,
                    signal_threshold=0.02,
                )
                ma_indicator = MovingAverageForecast(params)

                # Compute indicator
                results = ma_indicator.compute(df)

                # Get current state
                current_state = ma_indicator.get_current_state()
                logger.info(
                    f"{ticker} current MA: {current_state['ma']:.2f} "
                    f"(Trend: {current_state['trend']}, "
                    f"Confidence: {current_state['confidence']:.2%})"
                )

                # Plot indicator
                save_path = os.path.join(ma_output_dir, f"{ticker}_ma.png")
                ma_indicator.plot(
                    price_data=df["Close"],
                    show_signals=True,
                    show_forecasts=True,
                    save_path=save_path,
                    title=f"Moving Average Forecast - {ticker}",
                )

                logger.info(f"Moving Average plot saved for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
