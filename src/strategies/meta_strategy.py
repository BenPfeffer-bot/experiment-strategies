# File: src/strategies/meta_strategy.py

import os
import sys
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import indicator modules
from src.indicators.rsi import RSIMultiLength
from src.indicators.fibonacci import FibonacciTrailingStop
from src.indicators.macd import compute_macd, MACD
from src.indicators.moving_average import MovingAverageForecast
from src.indicators.variance import compute_variance
from src.indicators.kelter_channel import compute_kelter_channels

# Import strategy components
from src.analysis.market_regime import MarketRegimeDetector
from src.risks.risk_mgmt import RiskManager
from src.optimization.param_optimizer import BayesianOptimizer

# Import configuration variables
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


@dataclass
class SignalParameters:
    """Parameters for signal generation"""

    rsi_max_length: int = 20
    rsi_min_length: int = 10
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    fib_length: int = 20
    fib_retracement: float = 0.382
    ma_window: int = 20
    ma_memory: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    variance_length: int = 20
    kelter_period: int = 14
    kelter_multiplier: float = 1.5
    signal_threshold: int = 3


class MetaStrategy:
    """
    Enhanced Meta Strategy that combines multiple technical indicators with
    regime detection and dynamic risk management.
    """

    def __init__(
        self,
        params: Optional[SignalParameters] = None,
        output_dir: str = "experiments/meta_strategy",
    ):
        self.params = params if params else SignalParameters()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = RiskManager()

        # Setup logging
        self._setup_logging()

        # Initialize indicators
        self._init_indicators()

        # Store results
        self.results = {}

    def _setup_logging(self):
        """Configure logging"""
        log_file = (
            self.output_dir
            / f"meta_strategy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("MetaStrategy")

    def _init_indicators(self):
        """Initialize technical indicators with current parameters"""
        self.rsi_indicator = RSIMultiLength(
            max_length=self.params.rsi_max_length,
            min_length=self.params.rsi_min_length,
            overbought=self.params.rsi_overbought,
            oversold=self.params.rsi_oversold,
        )

        self.fib_stop = FibonacciTrailingStop(
            L=self.params.fib_length,
            R=1,
            use_labels=False,
            F=self.params.fib_retracement,
            use_f=False,
            trigger="close",
        )

        self.ma_forecaster = MovingAverageForecast(
            window=self.params.ma_window,
            max_memory=self.params.ma_memory,
            forecast_length=100,
            up_per=80,
            mid_per=50,
            dn_per=20,
        )

    def load_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Load and preprocess price data for a ticker.
        """
        try:
            filepath = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
            df = pd.read_csv(filepath)

            # Process datetime index
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], utc=True)
                df.set_index("Date", inplace=True)
            else:
                df.index = pd.to_datetime(df.index, utc=True)

            # Normalize dates by setting time to midnight
            df.index = df.index.floor("D")
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            # Ensure we have required columns
            if "Close" in df.columns:
                df["price"] = df["Close"]
            elif "close" in df.columns:
                df["price"] = df["close"]
            else:
                self.logger.error(f"No price data found for {ticker}")
                return None

            # Add required columns with proper handling
            df["returns"] = df["price"].pct_change()
            df["volume"] = (
                df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
            )
            df["high"] = df["High"] if "High" in df.columns else df["price"]
            df["low"] = df["Low"] if "Low" in df.columns else df["price"]

            # Forward fill any missing values
            df = df.fillna(method="ffill").fillna(method="bfill")

            return df

        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {str(e)}")
            return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on multiple indicators."""
        try:
            # Initialize RSI
            rsi = RSIMultiLength(
                max_length=20, min_length=10, overbought=80, oversold=20
            )
            # Calculate RSI using the price series
            rsi_signals = rsi.compute(df["price"])

            # Initialize other indicators
            macd = MACD()
            macd_signals = macd.compute(df["price"])

            # Combine signals
            signals = pd.DataFrame(index=df.index)
            signals["rsi"] = rsi_signals["avg_rsi"]
            signals["macd"] = macd_signals["macd"]
            signals["macd_signal"] = macd_signals["signal"]
            signals["macd_hist"] = macd_signals["hist"]

            # Generate trading signals
            signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell

            # RSI signals
            signals.loc[signals["rsi"] < 30, "signal"] = 1
            signals.loc[signals["rsi"] > 70, "signal"] = -1

            # MACD signals
            signals.loc[signals["macd_hist"] > 0, "signal"] = signals["signal"] + 1
            signals.loc[signals["macd_hist"] < 0, "signal"] = signals["signal"] - 1

            # Normalize signals
            signals["signal"] = signals["signal"].clip(-1, 1)

            # Add regime information
            signals["regime"] = df["regime"]

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()

    def _objective_function(self, params: Dict, data: pd.DataFrame) -> float:
        """
        Objective function for parameter optimization.
        Returns negative score (for minimization) combining Sharpe ratio with penalties.
        """
        try:
            # Convert parameters to appropriate types
            params = {
                k: int(v) if k != "kelter_multiplier" else float(v)
                for k, v in params.items()
            }

            # Update strategy parameters
            self.params = SignalParameters(**params)
            self._init_indicators()

            # Generate signals
            signals = self.generate_signals(data)
            if signals is None:
                return float("inf")

            # Calculate performance metrics
            returns = data["returns"].fillna(0)
            strategy_returns = returns * signals["signal"].shift(1).fillna(0)

            # Handle edge cases
            if (
                len(strategy_returns) < 100
            ):  # Need enough data for meaningful statistics
                return float("inf")

            # Calculate Sharpe ratio with error handling
            returns_std = strategy_returns.std()
            if returns_std == 0 or np.isnan(returns_std):
                return float("inf")

            returns_mean = strategy_returns.mean()
            if np.isnan(returns_mean):
                return float("inf")

            sharpe = np.sqrt(252) * returns_mean / returns_std

            # Add penalties for excessive trading and drawdown
            trades = np.abs(signals["signal"].diff()).sum()
            trade_penalty = 0.1 * trades / len(signals)

            cumulative_returns = strategy_returns.cumsum()
            max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
            dd_penalty = abs(max_drawdown)

            # Return final score with bounds
            score = -(sharpe - trade_penalty - dd_penalty)
            if np.isnan(score) or np.isinf(score):
                return float("inf")
            return min(max(score, -100), 100)  # Bound the score

        except Exception as e:
            self.logger.error(f"Error in objective function: {str(e)}")
            return float("inf")

    def optimize_parameters(
        self, data: pd.DataFrame, regime_label: int
    ) -> Optional[SignalParameters]:
        """
        Optimize strategy parameters for a specific market regime.
        """
        try:
            if len(data) < 100:
                self.logger.warning(f"Insufficient data for regime {regime_label}")
                return None

            # Define parameter space
            param_space = {
                "rsi_max_length": (10, 30),
                "rsi_min_length": (5, 15),
                "rsi_overbought": (65, 80),
                "rsi_oversold": (20, 35),
                "ma_window": (10, 50),
                "variance_length": (10, 30),
                "kelter_period": (10, 20),
                "kelter_multiplier": (1.0, 2.0),
                "signal_threshold": (2, 4),
            }

            # Ensure data is properly prepared
            data = data.copy()
            data["returns"] = data["returns"].fillna(0)
            data = data.dropna(subset=["price"])

            if len(data) < 100:
                self.logger.warning("Not enough clean data for optimization")
                return None

            # Create optimizer with partial function to include data
            from functools import partial

            objective = partial(self._objective_function, data=data)

            # Create optimizer with more conservative settings
            optimizer = BayesianOptimizer(
                param_space=param_space,
                objective_function=objective,
                n_initial_points=3,
                n_iterations=5,
            )

            # Run optimization
            best_params, best_score = optimizer.optimize()

            if best_score == float("inf") or np.isnan(best_score):
                self.logger.warning(f"Optimization failed for regime {regime_label}")
                return None

            # Convert optimized parameters to SignalParameters
            return SignalParameters(**best_params)

        except Exception as e:
            self.logger.error(
                f"Error optimizing parameters for regime {regime_label}: {str(e)}"
            )
            return None

    def process_ticker(self, ticker: str) -> Dict:
        """
        Process a single ticker with regime detection and parameter optimization.
        """
        try:
            self.logger.info(f"Processing {ticker}...")

            # Load data
            df = self.load_price_data(ticker)
            if df is None:
                return None

            # Detect market regimes
            regimes, regime_data = self.regime_detector.detect_regimes(df)

            # Process each regime
            regime_signals = {}
            for regime in regimes:
                # Get data for this regime
                regime_mask = regime_data["regime"] == regime.label
                regime_data_subset = df[regime_mask].copy()

                if len(regime_data_subset) < 100:  # Skip regimes with too little data
                    continue

                # Optimize parameters for this regime
                optimized_params = self.optimize_parameters(
                    regime_data_subset, regime.label
                )
                if optimized_params:
                    self.params = optimized_params
                    self._init_indicators()

                    # Generate signals for this regime
                    signals = self.generate_signals(regime_data_subset)
                    if signals is not None:
                        regime_signals[regime.label] = signals

            # Combine regime-specific signals
            combined_signals = pd.DataFrame(index=df.index)
            for label, signals in regime_signals.items():
                regime_mask = regime_data["regime"] == label
                combined_signals.loc[regime_mask] = signals.loc[regime_mask]

            # Add regime information to signals
            combined_signals["regime"] = regime_data["regime"]
            combined_signals["regime_probability"] = regime_data["probability"]

            # Save results
            output_dir = os.path.join(MARKET_DATA_DIR, "processed")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{ticker}_integrated_signals.csv")
            combined_signals.to_csv(output_file)

            self.logger.info(f"Completed processing {ticker}")
            return {
                "ticker": ticker,
                "n_regimes": len(regimes),
                "signals": combined_signals,
            }

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            return None


def main():
    """
    Run the meta strategy for all tickers.
    """
    strategy = MetaStrategy()
    results = {}

    for ticker in TICKERS:
        result = strategy.process_ticker(ticker)
        if result:
            results[ticker] = result

    return results


if __name__ == "__main__":
    main()
