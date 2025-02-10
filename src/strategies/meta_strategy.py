# File: src/strategies/meta_strategy.py

import os
import sys
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import indicator modules
from src.indicators.rsi import RSIMultiLength, RSIParameters
from src.indicators.fibonacci import FibonacciTrailingStop, FibParameters, TriggerType
from src.indicators.macd import MACD
from src.indicators.moving_average import MovingAverageForecast, MAParameters
from src.indicators.variance import VarianceIndicator, VarianceParameters
from src.indicators.kelter_channel import KelterChannel, KelterParameters
from src.indicators.pbv import PBVIndicator

# Import strategy components
from src.analysis.market_regime import MarketRegimeDetector
from src.risks.risk_mgmt import RiskManager
from src.optimization.param_optimizer import BayesianOptimizer

# Import configuration variables
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR


@dataclass
class SignalParameters:
    """Parameters for signal generation"""

    # RSI parameters
    rsi_max_length: int = 20
    rsi_min_length: int = 10
    rsi_overbought: float = 70
    rsi_oversold: float = 30

    # Moving Average parameters
    ma_window: int = 20
    ma_memory: int = 50
    ma_forecast_length: int = 100
    ma_up_percentile: float = 80
    ma_mid_percentile: float = 50
    ma_down_percentile: float = 20

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Fibonacci parameters
    fib_length: int = 20
    fib_retracement: float = 0.382
    fib_trigger: str = "close"

    # Kelter Channel parameters
    kelter_period: int = 14
    kelter_multiplier: float = 1.5

    # Variance parameters
    variance_length: int = 20
    variance_ewm_alpha: float = 0.1
    variance_use_log: bool = True

    # Signal generation parameters
    signal_threshold: float = 0.2  # Threshold for signal generation
    regime_weight: float = 0.5  # Weight given to regime-specific signals
    volatility_scale: bool = True  # Whether to scale signals by volatility
    confidence_threshold: float = 0.6  # Minimum confidence for signal generation

    # Risk management parameters
    max_position_size: float = 0.2  # Maximum position size as fraction of capital
    stop_loss: float = 0.02  # Stop loss as fraction of position value
    take_profit: float = 0.04  # Take profit as fraction of position value
    trailing_stop: bool = True  # Whether to use trailing stops
    risk_per_trade: float = 0.01  # Maximum risk per trade as fraction of capital


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

        # Get logger
        self.logger = logging.getLogger("MetaStrategy")

        # Only add handlers if they don't exist
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # File handler
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # Prevent propagation to root logger
            self.logger.propagate = False

    def _init_indicators(self):
        """Initialize technical indicators with current parameters"""
        self.rsi_indicator = RSIMultiLength(
            params=RSIParameters(
                max_length=self.params.rsi_max_length,
                min_length=self.params.rsi_min_length,
                overbought=self.params.rsi_overbought,
                oversold=self.params.rsi_oversold,
            )
        )

        self.fib_stop = FibonacciTrailingStop(
            params=FibParameters(
                left_window=self.params.fib_length,
                right_window=1,
                use_custom_fib=False,
                custom_fib=self.params.fib_retracement,
                trigger_type=TriggerType.CLOSE,
            )
        )

        self.ma_forecaster = MovingAverageForecast(
            params=MAParameters(
                window=self.params.ma_window,
                max_memory=self.params.ma_memory,
                forecast_length=self.params.ma_forecast_length,
                up_percentile=self.params.ma_up_percentile,
                mid_percentile=self.params.ma_mid_percentile,
                down_percentile=self.params.ma_down_percentile,
            )
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

            # Clean and preprocess data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()  # Use more modern fillna methods

            # Add log returns for regime detection
            df["log_returns"] = np.log1p(df["returns"])

            # Winsorize extreme values
            for col in ["returns", "log_returns"]:
                df[col] = np.clip(
                    df[col], df[col].quantile(0.001), df[col].quantile(0.999)
                )

            return df

        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {str(e)}")
            return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on multiple indicators."""
        try:
            # Ensure required columns exist and are numeric
            required_columns = ["price", "high", "low", "returns"]
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Initialize signals DataFrame with improved error handling
            try:
                signals = pd.DataFrame(
                    index=df.index, dtype=float
                )  # Explicitly set dtype
            except Exception as e:
                self.logger.error(f"Error initializing signals DataFrame: {str(e)}")
                return pd.DataFrame()

            # Initialize signal column with zeros
            signals["signal"] = 0.0  # Explicitly create as float

            # Generate individual indicator signals
            try:
                # RSI Signals
                rsi_signals = self.rsi_indicator.compute(df["price"])
                signals["rsi"] = pd.to_numeric(rsi_signals.avg_rsi, errors="coerce")
                signals["rsi_upper"] = pd.to_numeric(
                    rsi_signals.buy_rsi, errors="coerce"
                )
                signals["rsi_lower"] = pd.to_numeric(
                    rsi_signals.sell_rsi, errors="coerce"
                )

                # MACD Signals
                macd = MACD()
                macd_signals = macd.compute(df["price"])
                signals["macd"] = pd.to_numeric(macd_signals["macd"], errors="coerce")
                signals["macd_signal"] = pd.to_numeric(
                    macd_signals["signal"], errors="coerce"
                )
                signals["macd_hist"] = pd.to_numeric(
                    macd_signals["hist"], errors="coerce"
                )

                # Moving Average Signals
                ma_signals = self.ma_forecaster.compute(df["price"])
                signals["ma"] = pd.to_numeric(ma_signals.ma, errors="coerce")
                signals["ma_trend"] = pd.to_numeric(ma_signals.trend, errors="coerce")
                signals["ma_forecast"] = pd.to_numeric(
                    ma_signals.forecasts["mid"], errors="coerce"
                )

                # Generate combined signals with improved robustness
                # RSI-based signals
                signals.loc[
                    signals["rsi"] < self.params.rsi_oversold * 0.9, "signal"
                ] = (
                    signals.loc[
                        signals["rsi"] < self.params.rsi_oversold * 0.9, "signal"
                    ]
                    + 1.0
                )
                signals.loc[
                    signals["rsi"] > self.params.rsi_overbought * 1.1, "signal"
                ] = (
                    signals.loc[
                        signals["rsi"] > self.params.rsi_overbought * 1.1, "signal"
                    ]
                    - 1.0
                )

                # MACD signals with confirmation
                macd_cross = (signals["macd"] - signals["macd_signal"]).diff()
                signals.loc[(macd_cross > 0) & (signals["macd_hist"] > 0), "signal"] = (
                    signals.loc[(macd_cross > 0) & (signals["macd_hist"] > 0), "signal"]
                    + 0.5
                )
                signals.loc[(macd_cross < 0) & (signals["macd_hist"] < 0), "signal"] = (
                    signals.loc[(macd_cross < 0) & (signals["macd_hist"] < 0), "signal"]
                    - 0.5
                )

                # Moving Average trend signals
                ma_conditions = (
                    (signals["ma_trend"] > 0)
                    & (df["price"] > signals["ma"])
                    & (signals["ma_forecast"] > signals["ma"])
                )
                signals.loc[ma_conditions, "signal"] = (
                    signals.loc[ma_conditions, "signal"] + 0.5
                )

                ma_conditions_down = (
                    (signals["ma_trend"] < 0)
                    & (df["price"] < signals["ma"])
                    & (signals["ma_forecast"] < signals["ma"])
                )
                signals.loc[ma_conditions_down, "signal"] = (
                    signals.loc[ma_conditions_down, "signal"] - 0.5
                )

                # Normalize final signals to [-1, 1] range
                signals["signal"] = signals["signal"].clip(-1, 1)

                # Add confidence scores
                indicator_signs = np.sign(
                    [
                        signals["rsi"] - 50,
                        signals["macd_hist"],
                        signals["ma_trend"],
                    ]
                )
                signals["confidence"] = np.abs(np.nanmean(indicator_signs, axis=0))

                # Handle NaN values using modern pandas methods
                signals = signals.ffill().fillna(0)

                return signals

            except Exception as e:
                self.logger.error(f"Error generating indicator signals: {str(e)}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()

    @classmethod
    def _objective_function_class(
        cls, params: Dict, data: pd.DataFrame, strategy: "MetaStrategy"
    ) -> float:
        """
        Class method version of the objective function for parallel optimization.
        """
        try:
            # Convert parameters to appropriate types
            params = {
                k: int(v)
                if k
                not in [
                    "kelter_multiplier",
                    "variance_ewm_alpha",
                    "signal_threshold",
                    "regime_weight",
                    "confidence_threshold",
                ]
                else float(v)
                for k, v in params.items()
            }

            # Update strategy parameters
            strategy.params = SignalParameters(**params)
            strategy._init_indicators()

            # Generate signals
            signals = strategy.generate_signals(data)
            if signals is None or signals.empty:
                return float("inf")

            # Calculate performance metrics
            returns = pd.to_numeric(data["returns"], errors="coerce").fillna(0)
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

            # Annualize Sharpe ratio
            sharpe = np.sqrt(252) * returns_mean / returns_std

            # Calculate trading metrics
            trades = np.abs(signals["signal"].diff()).fillna(0).sum()
            trade_penalty = 0.1 * trades / len(signals)

            # Calculate drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())

            # Calculate final score with bounds
            score = -(sharpe - trade_penalty - max_drawdown)

            # Handle invalid scores
            if np.isnan(score) or np.isinf(score):
                return float("inf")

            # Bound the score
            return min(max(score, -100), 100)

        except Exception as e:
            # Don't log in parallel processes
            return float("inf")

    def optimize_parameters(
        self, data: pd.DataFrame, regime_label: int
    ) -> Optional[SignalParameters]:
        """
        Optimize strategy parameters for a specific market regime with improved robustness.
        """
        try:
            # Increase minimum data requirement for better statistical significance
            if len(data) < 150:  # Increased from 100
                self.logger.warning(
                    f"Insufficient data for regime {regime_label} ({len(data)} samples). "
                    "Using default parameters with adaptive scaling."
                )
                # Return scaled default parameters instead of None
                return self._get_scaled_default_parameters(data)

            # Define parameter space with improved bounds and scale
            param_space = {
                # RSI parameters with tighter bounds
                "rsi_min_length": (5, 15),
                "rsi_max_length": (15, 30),
                "rsi_overbought": (65, 80),
                "rsi_oversold": (20, 35),
                # Technical indicator parameters
                "kelter_period": (10, 20),
                "kelter_multiplier": (1.2, 2.0),
                "variance_length": (10, 30),
                "variance_ewm_alpha": (0.1, 0.5),
                # Forecast parameters
                "ma_forecast_length": (10, 30),
                "fib_length": (10, 30),
                # Signal generation parameters
                "signal_threshold": (0.1, 0.5),
                "regime_weight": (0.3, 0.7),
                "confidence_threshold": (0.4, 0.8),
            }

            # Create optimizer with improved GP configuration
            optimizer = BayesianOptimizer(
                param_space=param_space,
                objective_function=self._objective_function_class,
                n_initial_points=15,  # Increased from 10
                n_iterations=8,  # Increased from 5
                exploration_weight=0.4,  # Increased exploration
                random_state=42,
                data=data,
                strategy=self,
                gp_params={
                    "kernel": None,  # Let sklearn choose optimal kernel
                    "n_restarts_optimizer": 5,  # Increased from default
                    "normalize_y": True,  # Add normalization
                    "alpha": 1e-6,  # Reduced noise assumption
                },
            )

            # Run optimization with improved error handling and validation
            try:
                result = optimizer.optimize()
                if result is None:
                    self.logger.error(
                        f"Failed to find optimal parameters for regime {regime_label}"
                    )
                    return self._get_scaled_default_parameters(data)

                # Validate and adjust parameters
                best_params = self._validate_and_adjust_parameters(result.best_params)

                # Log optimization results with more detail
                self._log_optimization_results(regime_label, result, best_params)

                return SignalParameters(**best_params)

            except Exception as e:
                self.logger.error(f"Optimization failed: {str(e)}")
                return self._get_scaled_default_parameters(data)

        except Exception as e:
            self.logger.error(
                f"Error optimizing parameters for regime {regime_label}: {str(e)}"
            )
            return self._get_scaled_default_parameters(data)

    def _get_scaled_default_parameters(self, data: pd.DataFrame) -> SignalParameters:
        """Get default parameters scaled to the data characteristics."""
        volatility = data["returns"].std()
        trend_strength = abs(data["returns"].mean() / data["returns"].std())

        # Scale parameters based on market characteristics
        params = SignalParameters()
        params.signal_threshold *= 1 + volatility
        params.confidence_threshold *= 1 + trend_strength
        params.regime_weight *= 1 + trend_strength

        return params

    def _validate_and_adjust_parameters(self, params: Dict) -> Dict:
        """Validate and adjust parameters to ensure consistency."""
        validated = {}
        for k, v in params.items():
            if k in [
                "kelter_multiplier",
                "variance_ewm_alpha",
                "signal_threshold",
                "regime_weight",
                "confidence_threshold",
            ]:
                validated[k] = max(0.0, min(float(v), 1.0))
            else:
                validated[k] = max(1, min(int(v), 100))

        # Ensure RSI length relationship
        if validated["rsi_min_length"] >= validated["rsi_max_length"]:
            avg_length = (validated["rsi_min_length"] + validated["rsi_max_length"]) / 2
            validated["rsi_min_length"] = max(5, int(avg_length - 5))
            validated["rsi_max_length"] = min(30, int(avg_length + 5))

        # Ensure RSI threshold relationship
        if validated["rsi_oversold"] >= validated["rsi_overbought"]:
            mid_point = (validated["rsi_oversold"] + validated["rsi_overbought"]) / 2
            validated["rsi_oversold"] = max(20, int(mid_point - 15))
            validated["rsi_overbought"] = min(80, int(mid_point + 15))

        return validated

    def _log_optimization_results(
        self, regime_label: int, result: Any, best_params: Dict
    ) -> None:
        """Log detailed optimization results."""
        self.logger.info(f"Optimization results for regime {regime_label}:")
        self.logger.info(f"Best score: {result.best_score:.4f}")
        self.logger.info("Best parameters:")
        for param, value in best_params.items():
            self.logger.info(f"{param}: {value}")

        if hasattr(result, "all_scores_"):
            self.logger.info(f"Score distribution:")
            scores = pd.Series(result.all_scores_)
            self.logger.info(f"Mean: {scores.mean():.4f}")
            self.logger.info(f"Std: {scores.std():.4f}")
            self.logger.info(f"Min: {scores.min():.4f}")
            self.logger.info(f"Max: {scores.max():.4f}")

    def process_ticker(self, ticker: str) -> Dict:
        """
        Process a single ticker with improved regime detection and parameter optimization.
        """
        try:
            self.logger.info(f"Processing {ticker}...")

            # Load data
            df = self.load_price_data(ticker)
            if df is None:
                return None

            # Detect market regimes with error handling
            try:
                regimes, regime_data = self.regime_detector.detect_regimes(df)
            except Exception as e:
                self.logger.warning(
                    f"HMM detection failed: {str(e)}. Using fallback method."
                )
                # Improved fallback regime detection using multiple features
                vol = df["returns"].rolling(20).std()
                trend = df["returns"].rolling(20).mean()

                # Create regime features
                features = pd.DataFrame(
                    {
                        "volatility": vol,
                        "trend": trend,
                    }
                )

                # Normalize features
                features = (features - features.mean()) / features.std()

                # Assign regimes using K-means
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=4, random_state=42)
                regime_data = pd.DataFrame(index=df.index)
                regime_data["regime"] = kmeans.fit_predict(features.fillna(0))
                regime_data["probability"] = 1.0
                regimes = [type("Regime", (), {"label": i}) for i in range(4)]

            # Process each regime
            regime_signals = {}
            valid_regimes = 0

            for regime in regimes:
                # Get data for this regime
                regime_mask = regime_data["regime"] == regime.label
                regime_data_subset = df[regime_mask].copy()

                if len(regime_data_subset) < 100:
                    self.logger.warning(
                        f"Insufficient data for regime {regime.label} ({len(regime_data_subset)} samples)"
                    )
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
                    if signals is not None and not signals.empty:
                        regime_signals[regime.label] = signals
                        valid_regimes += 1

            # Combine regime-specific signals
            combined_signals = pd.DataFrame(index=df.index)

            if valid_regimes > 0:
                for label, signals in regime_signals.items():
                    regime_mask = regime_data["regime"] == label
                    combined_signals.loc[regime_mask] = signals.loc[regime_mask]

                # Fill any gaps with neutral signals
                combined_signals = combined_signals.fillna(0)

                # Add regime information to signals
                combined_signals["regime"] = regime_data["regime"]
                combined_signals["regime_probability"] = regime_data["probability"]

                # Save results
                output_dir = os.path.join(MARKET_DATA_DIR, "processed")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(
                    output_dir, f"{ticker}_integrated_signals.csv"
                )
                combined_signals.to_csv(output_file)

                self.logger.info(
                    f"Completed processing {ticker} with {valid_regimes} valid regimes"
                )
                return {
                    "ticker": ticker,
                    "n_regimes": valid_regimes,
                    "signals": combined_signals,
                }
            else:
                self.logger.warning(f"No valid signals generated for {ticker}")
                return None

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
