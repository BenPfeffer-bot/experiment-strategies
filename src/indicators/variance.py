# File: src/indicators/variance.py
"""
Enhanced Variance Indicator

This module implements an advanced Variance indicator with multiple calculation methods
and adaptive window sizes. It provides volatility analysis and risk metrics for
trading signals based on variance patterns.

Features:
- Multiple variance calculation methods
- Adaptive window sizing
- Volatility regime detection
- Risk-adjusted signals
- Advanced visualization
- Performance optimization

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import logging
from datetime import datetime
from enum import Enum

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


class VarianceType(Enum):
    """Types of variance calculations"""

    STANDARD = "standard"  # Standard rolling variance
    EWMA = "ewma"  # Exponentially weighted moving average
    PARKINSON = "parkinson"  # Uses high-low range
    GARMAN_KLASS = "garman_klass"  # Uses OHLC prices


class VolatilityRegime(Enum):
    """Volatility regime classifications"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class VarianceParameters:
    """Parameters for variance calculations"""

    period: int = 20  # Main calculation period
    ewm_alpha: float = 0.1  # EWMA decay factor
    use_log_returns: bool = True  # Use log returns for calculations
    variance_type: VarianceType = VarianceType.STANDARD
    price_column: str = "close"
    high_column: str = "high"
    low_column: str = "low"
    open_column: str = "open"
    signal_threshold: float = 0.02  # Minimum change for signal generation
    regime_percentiles: Tuple[float, float] = (33.0, 67.0)  # Regime boundaries


@dataclass
class VarianceResult:
    """Container for variance calculation results"""

    variance: pd.Series
    volatility: pd.Series  # Square root of variance
    regime: pd.Series
    signals: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    zscore: pd.Series = None  # Standardized variance


class VarianceIndicator:
    """
    Enhanced Variance Indicator Implementation.

    This indicator calculates price variance using multiple methods and provides
    volatility regime detection and signal generation. Features include:
    - Multiple variance calculation methods
    - Volatility regime classification
    - Signal generation based on variance patterns
    - Risk-adjusted metrics
    - Advanced visualization options

    Attributes:
        params (VarianceParameters): Configuration parameters
        logger (logging.Logger): Logger instance
        results (Optional[VarianceResult]): Computed results
    """

    def __init__(self, params: Optional[VarianceParameters] = None):
        """
        Initialize Variance indicator with given parameters.

        Args:
            params: Optional[VarianceParameters]
                Parameters for calculations. If None, uses default values.
        """
        self.params = params if params else VarianceParameters()
        self._validate_parameters()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize results container
        self.results: Optional[VarianceResult] = None

    def _validate_parameters(self) -> None:
        """Validate Variance parameters."""
        if self.params.period < 2:
            raise ValueError("period must be at least 2")
        if not 0 < self.params.ewm_alpha <= 1:
            raise ValueError("ewm_alpha must be between 0 and 1")
        if (
            not 0
            <= self.params.regime_percentiles[0]
            < self.params.regime_percentiles[1]
            <= 100
        ):
            raise ValueError(
                "regime_percentiles must be in ascending order and between 0 and 100"
            )

    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate price returns.

        Args:
            prices: np.ndarray
                Price series

        Returns:
            np.ndarray of returns
        """
        if self.params.use_log_returns:
            returns = np.log(prices[1:] / prices[:-1])
        else:
            returns = np.diff(prices) / prices[:-1]

        # Pad first value
        returns = np.insert(returns, 0, 0)

        return returns

    def _calculate_parkinson_variance(
        self, high: np.ndarray, low: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Parkinson's variance estimator.

        Args:
            high: np.ndarray
                High prices
            low: np.ndarray
                Low prices

        Returns:
            np.ndarray of variance values
        """
        # Calculate normalized range
        hl_range = np.log(high / low)
        squared_range = hl_range * hl_range

        # Calculate Parkinson's variance (with bias correction)
        variance = squared_range / (4 * np.log(2))

        # Calculate rolling mean
        variance = pd.Series(variance).rolling(window=self.params.period).mean().values

        return variance

    def _calculate_garman_klass_variance(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Garman-Klass variance estimator.

        Args:
            open_: np.ndarray
                Open prices
            high: np.ndarray
                High prices
            low: np.ndarray
                Low prices
            close: np.ndarray
                Close prices

        Returns:
            np.ndarray of variance values
        """
        # Calculate components
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)

        # Calculate Garman-Klass variance
        variance = 0.5 * log_hl * log_hl - (2 * np.log(2) - 1) * log_co * log_co

        # Calculate rolling mean
        variance = pd.Series(variance).rolling(window=self.params.period).mean().values

        return variance

    def compute(self, data: Union[pd.Series, pd.DataFrame]) -> VarianceResult:
        """
        Compute variance and related metrics.

        Args:
            data: Union[pd.Series, pd.DataFrame]
                Price data. If DataFrame, uses columns specified in params

        Returns:
            VarianceResult containing computed values

        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Extract price series
            if isinstance(data, pd.DataFrame):
                if self.params.variance_type in [
                    VarianceType.PARKINSON,
                    VarianceType.GARMAN_KLASS,
                ]:
                    required_columns = [
                        self.params.price_column,
                        self.params.high_column,
                        self.params.low_column,
                    ]
                    if self.params.variance_type == VarianceType.GARMAN_KLASS:
                        required_columns.append(self.params.open_column)
                else:
                    required_columns = [self.params.price_column]

                missing_columns = [
                    col for col in required_columns if col not in data.columns
                ]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")

                closes = data[self.params.price_column].values
                if self.params.variance_type in [
                    VarianceType.PARKINSON,
                    VarianceType.GARMAN_KLASS,
                ]:
                    highs = data[self.params.high_column].values
                    lows = data[self.params.low_column].values
                    if self.params.variance_type == VarianceType.GARMAN_KLASS:
                        opens = data[self.params.open_column].values
            else:
                # If Series, use it for all prices
                closes = data.values
                highs = lows = opens = closes

            n_bars = len(closes)

            # Calculate variance based on type
            if self.params.variance_type == VarianceType.STANDARD:
                returns = self._calculate_returns(closes)
                variance = (
                    pd.Series(returns * returns)
                    .rolling(window=self.params.period)
                    .mean()
                    .values
                )

            elif self.params.variance_type == VarianceType.EWMA:
                returns = self._calculate_returns(closes)
                variance = (
                    pd.Series(returns * returns)
                    .ewm(alpha=self.params.ewm_alpha)
                    .mean()
                    .values
                )

            elif self.params.variance_type == VarianceType.PARKINSON:
                variance = self._calculate_parkinson_variance(highs, lows)

            else:  # GARMAN_KLASS
                variance = self._calculate_garman_klass_variance(
                    opens, highs, lows, closes
                )

            # Calculate volatility (annualized)
            volatility = np.sqrt(variance) * np.sqrt(252)

            # Calculate rolling statistics for z-scores
            rolling_mean = (
                pd.Series(variance).rolling(window=self.params.period).mean().values
            )
            rolling_std = (
                pd.Series(variance).rolling(window=self.params.period).std().values
            )

            # Calculate z-scores
            zscore = np.zeros_like(variance)
            mask = rolling_std != 0
            zscore[mask] = (variance[mask] - rolling_mean[mask]) / rolling_std[mask]

            # Determine volatility regimes
            regimes = np.full(n_bars, VolatilityRegime.MEDIUM.value)
            percentiles = np.nanpercentile(variance, self.params.regime_percentiles)
            regimes[variance <= percentiles[0]] = VolatilityRegime.LOW.value
            regimes[variance > percentiles[1]] = VolatilityRegime.HIGH.value

            # Generate signals based on z-scores
            signals = np.zeros(n_bars)
            signals[zscore > self.params.signal_threshold] = 1  # High variance signal
            signals[zscore < -self.params.signal_threshold] = -1  # Low variance signal

            # Create result series
            index = data.index if isinstance(data, (pd.Series, pd.DataFrame)) else None
            self.results = VarianceResult(
                variance=pd.Series(variance, index=index, name="Variance"),
                volatility=pd.Series(volatility, index=index, name="Volatility"),
                regime=pd.Series(regimes, index=index, name="Regime"),
                signals=pd.Series(signals, index=index, name="Signals"),
                rolling_mean=pd.Series(rolling_mean, index=index, name="Rolling Mean"),
                rolling_std=pd.Series(rolling_std, index=index, name="Rolling Std"),
                zscore=pd.Series(zscore, index=index, name="Z-Score"),
            )

            return self.results

        except Exception as e:
            self.logger.error(f"Error computing Variance: {str(e)}")
            raise

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current indicator state.

        Returns:
            Dict containing current values for variance metrics
        """
        if self.results is None:
            raise ValueError("Variance not computed. Call compute() first.")

        return {
            "variance": float(self.results.variance.iloc[-1]),
            "volatility": float(self.results.volatility.iloc[-1]),
            "regime": self.results.regime.iloc[-1],
            "signal": float(self.results.signals.iloc[-1]),
            "zscore": float(self.results.zscore.iloc[-1]),
        }

    def plot(
        self,
        price_data: Optional[pd.Series] = None,
        show_signals: bool = True,
        show_regimes: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot Variance indicator with price data.

        Args:
            price_data: Optional[pd.Series]
                Price data to plot
            show_signals: bool
                If True, shows trading signals
            show_regimes: bool
                If True, shows regime changes
            save_path: Optional[str]
                If provided, saves plot to this path
            title: Optional[str]
                Custom plot title
        """
        if self.results is None:
            raise ValueError("Variance not computed. Call compute() first.")

        try:
            # Create figure with subplots
            n_plots = 2 + (1 if show_regimes else 0)
            fig, axes = plt.subplots(
                n_plots,
                1,
                figsize=(12, 4 * n_plots),
                height_ratios=[2, 1] + ([1] if show_regimes else []),
            )

            # Plot price and volatility
            ax_price = axes[0]
            if price_data is not None:
                ax_price.plot(
                    price_data.index,
                    price_data,
                    label="Price",
                    color="black",
                    linewidth=1,
                )
                ax_price.set_title("Price and Volatility")

                # Plot volatility overlay
                ax_vol = ax_price.twinx()
                ax_vol.plot(
                    self.results.volatility.index,
                    self.results.volatility,
                    label="Volatility",
                    color="blue",
                    alpha=0.5,
                )
                ax_vol.set_ylabel("Volatility")

            # Plot variance and signals
            ax_var = axes[1]
            ax_var.plot(
                self.results.variance.index,
                self.results.variance,
                label="Variance",
                color="purple",
            )
            ax_var.plot(
                self.results.rolling_mean.index,
                self.results.rolling_mean,
                label="Mean",
                color="green",
                linestyle="--",
            )

            # Plot signals if requested
            if show_signals:
                signals = self.results.signals
                high_var = self.results.variance[signals == 1]
                low_var = self.results.variance[signals == -1]

                ax_var.scatter(
                    high_var.index,
                    high_var,
                    color="red",
                    marker="^",
                    s=100,
                    label="High Variance",
                )
                ax_var.scatter(
                    low_var.index,
                    low_var,
                    color="green",
                    marker="v",
                    s=100,
                    label="Low Variance",
                )

            ax_var.set_title("Variance")
            ax_var.grid(True, alpha=0.3)
            ax_var.legend()

            # Plot regimes if requested
            if show_regimes:
                ax_regime = axes[2]
                regimes = pd.Series(
                    np.where(
                        self.results.regime == VolatilityRegime.HIGH.value,
                        1,
                        np.where(
                            self.results.regime == VolatilityRegime.LOW.value, -1, 0
                        ),
                    ),
                    index=self.results.regime.index,
                )
                ax_regime.plot(regimes.index, regimes, color="blue", label="Regime")
                ax_regime.set_yticks([-1, 0, 1])
                ax_regime.set_yticklabels(["Low", "Medium", "High"])
                ax_regime.set_title("Volatility Regime")
                ax_regime.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            plt.close()
            raise ValueError(f"Error plotting Variance: {str(e)}")


if __name__ == "__main__":
    """Example usage of the Variance indicator"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create output directory
        variance_output_dir = os.path.join(FIGURES_DIR, "variance")
        os.makedirs(variance_output_dir, exist_ok=True)

        # Process each ticker
        for ticker in TICKERS:
            try:
                logger.info(f"Processing Variance for {ticker}")

                # Load data
                data_path = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
                df = pd.read_csv(data_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

                # Initialize indicator with custom parameters
                params = VarianceParameters(
                    period=20,
                    ewm_alpha=0.1,
                    use_log_returns=True,
                    variance_type=VarianceType.GARMAN_KLASS,
                    price_column="Close",
                    high_column="High",
                    low_column="Low",
                    open_column="Open",
                    signal_threshold=2.0,
                    regime_percentiles=(33.0, 67.0),
                )
                variance_indicator = VarianceIndicator(params)

                # Compute indicator
                results = variance_indicator.compute(df)

                # Get current state
                current_state = variance_indicator.get_current_state()
                logger.info(
                    f"{ticker} current volatility: {current_state['volatility']:.2%} "
                    f"(Regime: {current_state['regime']}, "
                    f"Z-Score: {current_state['zscore']:.2f})"
                )

                # Plot indicator
                save_path = os.path.join(variance_output_dir, f"{ticker}_variance.png")
                variance_indicator.plot(
                    price_data=df["Close"],
                    show_signals=True,
                    show_regimes=True,
                    save_path=save_path,
                    title=f"Variance Analysis - {ticker}",
                )

                logger.info(f"Variance plot saved for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
