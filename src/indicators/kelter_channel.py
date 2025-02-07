# File: src/indicators/kelter_channels.py
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

# Adjust sys.path to load configuration
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


class ChannelType(Enum):
    """Types of Kelter Channel calculations"""

    STANDARD = "standard"  # Standard deviation based
    ATR = "atr"  # Average True Range based
    HYBRID = "hybrid"  # Combination of both


@dataclass
class KelterParameters:
    """Parameters for Kelter Channel calculations"""

    period: int = 20  # Main calculation period
    atr_period: int = 14  # ATR calculation period
    std_multiplier: float = 2.0  # Standard deviation multiplier
    atr_multiplier: float = 1.5  # ATR multiplier
    channel_type: ChannelType = ChannelType.HYBRID
    price_column: str = "close"
    high_column: str = "high"
    low_column: str = "low"
    signal_threshold: float = 0.02  # Minimum distance for signal generation
    use_log_returns: bool = True  # Use log returns for volatility calculation


@dataclass
class KelterResult:
    """Container for Kelter Channel calculation results"""

    middle: pd.Series
    upper: pd.Series
    lower: pd.Series
    width: pd.Series
    signals: pd.Series
    atr: pd.Series
    volatility: pd.Series


class KelterChannel:
    """
    Enhanced Kelter Channel Indicator Implementation.

    This indicator calculates dynamic price channels based on volatility and ATR,
    providing support for multiple calculation methods and signal generation.

    Features:
    - Multiple channel calculation methods
    - Dynamic volatility assessment
    - Adaptive ATR calculations
    - Signal generation based on breakouts
    - Channel width analysis
    - Advanced visualization options

    Attributes:
        params (KelterParameters): Configuration parameters
        logger (logging.Logger): Logger instance
        results (Optional[KelterResult]): Computed results
    """

    def __init__(self, params: Optional[KelterParameters] = None):
        """
        Initialize Kelter Channel indicator with given parameters.

        Args:
            params: Optional[KelterParameters]
                Parameters for calculations. If None, uses default values.
        """
        self.params = params if params else KelterParameters()
        self._validate_parameters()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize results container
        self.results: Optional[KelterResult] = None

    def _validate_parameters(self) -> None:
        """Validate Kelter Channel parameters."""
        if self.params.period < 2:
            raise ValueError("period must be at least 2")
        if self.params.atr_period < 1:
            raise ValueError("atr_period must be at least 1")
        if self.params.std_multiplier <= 0:
            raise ValueError("std_multiplier must be positive")
        if self.params.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Average True Range.

        Args:
            high: np.ndarray
                High prices
            low: np.ndarray
                Low prices
            close: np.ndarray
                Close prices

        Returns:
            np.ndarray of ATR values
        """
        # Calculate True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        tr = np.maximum(tr1, tr2)
        tr = np.maximum(tr, tr3)

        # Calculate ATR
        atr = pd.Series(tr).rolling(window=self.params.atr_period).mean().values

        return atr

    def _calculate_volatility(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate price volatility.

        Args:
            prices: np.ndarray
                Price series

        Returns:
            np.ndarray of volatility values
        """
        if self.params.use_log_returns:
            returns = np.log(prices[1:] / prices[:-1])
        else:
            returns = np.diff(prices) / prices[:-1]

        # Pad first value
        returns = np.insert(returns, 0, 0)

        # Calculate rolling standard deviation
        volatility = pd.Series(returns).rolling(window=self.params.period).std().values

        return volatility

    def compute(self, data: Union[pd.Series, pd.DataFrame]) -> KelterResult:
        """
        Compute Kelter Channel values.

        Args:
            data: Union[pd.Series, pd.DataFrame]
                Price data. If DataFrame, uses columns specified in params

        Returns:
            KelterResult containing computed values

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

            # Calculate middle line (SMA)
            middle = pd.Series(closes).rolling(window=self.params.period).mean().values

            # Calculate ATR
            atr = self._calculate_atr(highs, lows, closes)

            # Calculate volatility
            volatility = self._calculate_volatility(closes)

            # Calculate channel width based on type
            if self.params.channel_type == ChannelType.STANDARD:
                width = volatility * self.params.std_multiplier * closes
            elif self.params.channel_type == ChannelType.ATR:
                width = atr * self.params.atr_multiplier
            else:  # HYBRID
                # Combine both methods with dynamic weighting
                vol_width = volatility * self.params.std_multiplier * closes
                atr_width = atr * self.params.atr_multiplier
                # Use more volatile measure
                width = np.maximum(vol_width, atr_width)

            # Calculate channels
            upper = middle + width
            lower = middle - width

            # Generate signals
            signals = np.zeros(n_bars)

            # Buy signal when price crosses above upper channel
            signals[closes > upper] = 1

            # Sell signal when price crosses below lower channel
            signals[closes < lower] = -1

            # Create result series
            index = data.index if isinstance(data, (pd.Series, pd.DataFrame)) else None
            self.results = KelterResult(
                middle=pd.Series(middle, index=index, name="Middle"),
                upper=pd.Series(upper, index=index, name="Upper"),
                lower=pd.Series(lower, index=index, name="Lower"),
                width=pd.Series(width, index=index, name="Width"),
                signals=pd.Series(signals, index=index, name="Signals"),
                atr=pd.Series(atr, index=index, name="ATR"),
                volatility=pd.Series(volatility, index=index, name="Volatility"),
            )

            return self.results

        except Exception as e:
            self.logger.error(f"Error computing Kelter Channel: {str(e)}")
            raise

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current indicator state.

        Returns:
            Dict containing current values for channels and signals
        """
        if self.results is None:
            raise ValueError("Kelter Channel not computed. Call compute() first.")

        return {
            "middle": float(self.results.middle.iloc[-1]),
            "upper": float(self.results.upper.iloc[-1]),
            "lower": float(self.results.lower.iloc[-1]),
            "width": float(self.results.width.iloc[-1]),
            "signal": float(self.results.signals.iloc[-1]),
            "atr": float(self.results.atr.iloc[-1]),
            "volatility": float(self.results.volatility.iloc[-1]),
        }

    def plot(
        self,
        price_data: Optional[pd.Series] = None,
        show_signals: bool = True,
        show_width: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot Kelter Channel indicator with price data.

        Args:
            price_data: Optional[pd.Series]
                Price data to plot
            show_signals: bool
                If True, shows trading signals
            show_width: bool
                If True, shows channel width subplot
            save_path: Optional[str]
                If provided, saves plot to this path
            title: Optional[str]
                Custom plot title
        """
        if self.results is None:
            raise ValueError("Kelter Channel not computed. Call compute() first.")

        try:
            # Create figure with subplots
            n_plots = 1 + (1 if show_width else 0)
            fig, axes = plt.subplots(
                n_plots,
                1,
                figsize=(12, 8 * n_plots),
                height_ratios=[3, 1] if show_width else [1],
            )

            if n_plots == 1:
                axes = [axes]

            # Plot main chart
            ax = axes[0]

            # Plot price data
            if price_data is not None:
                ax.plot(
                    price_data.index,
                    price_data,
                    label="Price",
                    color="black",
                    linewidth=1,
                )

            # Plot channels
            ax.plot(
                self.results.middle.index,
                self.results.middle,
                label="Middle",
                color="blue",
                linewidth=1.5,
            )
            ax.plot(
                self.results.upper.index,
                self.results.upper,
                label="Upper",
                color="green",
                linestyle="--",
            )
            ax.plot(
                self.results.lower.index,
                self.results.lower,
                label="Lower",
                color="red",
                linestyle="--",
            )

            # Fill between channels
            ax.fill_between(
                self.results.upper.index,
                self.results.upper,
                self.results.lower,
                color="gray",
                alpha=0.1,
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

            ax.set_title(
                title or f"Kelter Channel ({self.params.channel_type.value.title()})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Plot width if requested
            if show_width:
                ax_width = axes[1]
                ax_width.plot(
                    self.results.width.index,
                    self.results.width,
                    label="Channel Width",
                    color="purple",
                )
                ax_width.set_title("Channel Width")
                ax_width.grid(True, alpha=0.3)
                ax_width.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            plt.close()
            raise ValueError(f"Error plotting Kelter Channel: {str(e)}")


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create output directory
        kelter_output_dir = os.path.join(FIGURES_DIR, "kelter")
        os.makedirs(kelter_output_dir, exist_ok=True)

        # Process each ticker
        for ticker in TICKERS:
            try:
                logger.info(f"Processing Kelter Channel for {ticker}")

                # Load data
                data_path = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
                df = pd.read_csv(data_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

                # Initialize indicator with custom parameters
                params = KelterParameters(
                    period=20,
                    atr_period=14,
                    std_multiplier=2.0,
                    atr_multiplier=1.5,
                    channel_type=ChannelType.HYBRID,
                    price_column="Close",
                    high_column="High",
                    low_column="Low",
                    signal_threshold=0.02,
                    use_log_returns=True,
                )
                kelter_indicator = KelterChannel(params)

                # Compute indicator
                results = kelter_indicator.compute(df)

                # Get current state
                current_state = kelter_indicator.get_current_state()
                logger.info(
                    f"{ticker} current width: {current_state['width']:.2f} "
                    f"(ATR: {current_state['atr']:.2f}, "
                    f"Volatility: {current_state['volatility']:.2%})"
                )

                # Plot indicator
                save_path = os.path.join(kelter_output_dir, f"{ticker}_kelter.png")
                kelter_indicator.plot(
                    price_data=df["Close"],
                    show_signals=True,
                    show_width=True,
                    save_path=save_path,
                    title=f"Kelter Channel - {ticker}",
                )

                logger.info(f"Kelter Channel plot saved for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
