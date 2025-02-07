import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.models.ml_model import MLModel
from src.models.reinforcement_learning import QLearningAgent
from src.backtesting.backtest_engine import BacktestEngine
from src.optimization.param_optimizer import BayesianOptimizer
from src.analysis.market_regime import MarketRegimeDetector
from src.risks.risk_mgmt import RiskManager
from src.utils.config import TICKERS, MARKET_DATA_DIR


class EnhancedStrategy:
    def __init__(self, output_dir: str = "experiments/enhanced_strategy"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.risk_manager = RiskManager()
        self.regime_detector = MarketRegimeDetector()

        # Store results
        self.results = {}

    def _setup_logging(self):
        """Configure logging"""
        log_file = (
            self.output_dir
            / f"enhanced_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("EnhancedStrategy")

    def _load_and_preprocess_data(self, ticker: str) -> pd.DataFrame:
        """Load and preprocess data for a ticker"""
        try:
            # Load data
            data = pd.read_csv(
                os.path.join(
                    MARKET_DATA_DIR, "processed", f"{ticker}_integrated_signals.csv"
                ),
                parse_dates=["Date"],
                index_col="Date",
            )

            # Ensure all required columns exist
            required_columns = ["price", "volume", "high", "low", "open", "close"]
            for col in required_columns:
                if col not in data.columns:
                    if col == "price":
                        data["price"] = data["close"]
                    else:
                        data[col] = 0

            # Calculate basic features if not present
            if "returns" not in data.columns:
                data["returns"] = data["price"].pct_change()

            if "volatility" not in data.columns:
                data["volatility"] = data["returns"].rolling(20).std() * np.sqrt(252)

            if "variance" not in data.columns:
                data["variance"] = data["volatility"] ** 2

            # Forward fill any missing values
            data = data.ffill().bfill()

            return data

        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {str(e)}")
            return None

    def _objective_function(self, params):
        """Objective function for parameter optimization"""
        try:
            # Convert numeric parameters
            params = {
                k: int(v)
                if k in ["lookback_period", "volatility_window", "momentum_window"]
                else float(v)
                for k, v in params.items()
            }

            # Initialize models with the parameters
            ml_model = MLModel()
            ml_model.train_ml_model(self._optimization_ticker)

            # Run backtest with risk management
            backtest = BacktestEngine(
                strategy=ml_model,
                data=self._optimization_data,
                initial_capital=100000,
                position_size=params["position_size"],
                transaction_costs=0.001,
            )

            metrics = backtest.run_backtest()
            if metrics is None:
                return np.inf

            # Extract metrics
            sharpe = metrics.sharpe_ratio
            sortino = metrics.sortino_ratio
            calmar = metrics.calmar_ratio
            max_dd = abs(metrics.max_drawdown)

            # Return infinity if any metric is NaN
            if (
                np.isnan(sharpe)
                or np.isnan(sortino)
                or np.isnan(calmar)
                or np.isnan(max_dd)
            ):
                return np.inf

            # Cap metrics to prevent infinite values
            sharpe = np.clip(sharpe, -10, 10)
            sortino = np.clip(sortino, -10, 10)
            calmar = np.clip(calmar, -10, 10)
            max_dd = np.clip(max_dd, 0, 1)

            # Calculate score with bounded metrics
            score = 0.4 * sharpe + 0.3 * sortino + 0.2 * calmar + 0.1 * (1 - max_dd)

            # Apply drawdown penalty
            dd_penalty = 1 + max_dd if max_dd <= 0.2 else np.exp(max_dd * 2)
            score = score / dd_penalty

            # Ensure score is finite
            if np.isnan(score) or np.isinf(score):
                return np.inf

            return -score  # Negative because we're minimizing

        except Exception as e:
            self.logger.error(f"Error in objective function: {str(e)}")
            return np.inf

    def optimize_parameters(self, data: pd.DataFrame, ticker: str):
        """Optimize strategy parameters using Bayesian optimization"""
        self.logger.info(f"Optimizing parameters for {ticker}")

        # Define parameter space with reasonable bounds
        param_space = {
            "lookback_period": (10, 100),
            "volatility_window": (5, 60),
            "momentum_window": (5, 60),
            "position_size": (0.1, 0.3),
            "stop_loss": (0.01, 0.05),
            "take_profit": (0.02, 0.10),
        }

        try:
            # Store data and ticker for objective function
            self._optimization_data = data
            self._optimization_ticker = ticker

            # Create optimizer with reduced iterations for faster convergence
            optimizer = BayesianOptimizer(
                param_space=param_space,
                objective_function=self._objective_function,
                n_initial_points=5,  # Reduced from 10
                n_iterations=10,  # Reduced from 50
            )

            best_params, best_value = optimizer.optimize()

            # Clean up stored data
            del self._optimization_data
            del self._optimization_ticker

            # Log optimization results
            self.logger.info(f"Optimization completed for {ticker}")
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(
                f"Best score: {-best_value}"
            )  # Convert back to positive score

            return best_params

        except Exception as e:
            self.logger.error(f"Optimization failed for {ticker}: {str(e)}")
            # Clean up stored data in case of error
            if hasattr(self, "_optimization_data"):
                del self._optimization_data
            if hasattr(self, "_optimization_ticker"):
                del self._optimization_ticker
            return None

    def train_and_evaluate(self, ticker: str):
        """Train and evaluate the enhanced strategy for a single ticker"""
        self.logger.info(f"Processing {ticker}")

        # Load and preprocess data
        data = self._load_and_preprocess_data(ticker)
        if data is None:
            return None

        try:
            # Detect market regimes
            regimes, regime_data = self.regime_detector.detect_regimes(data)

            # Log regime information
            self.logger.info(f"Detected {len(regimes)} market regimes for {ticker}")
            for regime in regimes:
                self.logger.info(f"Regime {regime.label}: {regime.description}")

            # Optimize parameters for each regime
            regime_params = {}
            for regime in regimes:
                regime_mask = regime_data["regime"] == regime.label
                if not any(regime_mask):
                    continue

                regime_data_subset = data[regime_mask]
                if len(regime_data_subset) < 100:  # Skip regimes with too little data
                    continue

                self.logger.info(f"Optimizing parameters for regime {regime.label}")
                params = self.optimize_parameters(regime_data_subset, ticker)
                if params:
                    regime_params[regime.label] = params

            if not regime_params:
                self.logger.warning(f"No valid regime parameters found for {ticker}")
                return None

            # Train ML model with regime-specific parameters
            ml_model = MLModel()
            ml_model.train_ml_model(ticker)

            # Add regime information to data for backtesting
            data = data.copy()
            data["current_regime"] = regime_data["regime"]
            data["regime_probability"] = regime_data["probability"]

            # Run backtest with regime-aware parameters
            backtest = BacktestEngine(
                strategy=ml_model,
                data=data,
                initial_capital=100000,
                position_size=0.1,  # Default position size
                transaction_costs=0.001,
            )

            metrics = backtest.run_backtest()
            backtest.plot_results(
                save_path=self.output_dir / f"{ticker}_performance.png"
            )

            # Store results
            self.results[ticker] = {
                "regime_params": regime_params,
                "metrics": metrics.__dict__ if metrics else None,
                "regimes": [
                    {
                        "label": r.label,
                        "trend": r.trend,
                        "volatility": r.volatility,
                        "liquidity": r.liquidity,
                        "avg_return": r.avg_return,
                        "avg_volume": r.avg_volume,
                    }
                    for r in regimes
                ],
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error in train_and_evaluate for {ticker}: {str(e)}")
            return None

    def run_strategy(self):
        """Run the enhanced strategy for all tickers"""
        summary_metrics = []

        for ticker in TICKERS:
            try:
                metrics = self.train_and_evaluate(ticker)
                if metrics is not None:
                    summary_metrics.append(
                        {
                            "ticker": ticker,
                            "sharpe_ratio": metrics.sharpe_ratio,
                            "sortino_ratio": metrics.sortino_ratio,
                            "max_drawdown": metrics.max_drawdown,
                            "total_return": self.results[ticker]["metrics"].get(
                                "total_return", 0
                            ),
                            "win_rate": metrics.win_rate,
                        }
                    )
            except Exception as e:
                self.logger.error(f"Error processing {ticker}: {str(e)}")
                continue

        # Create summary report only if we have results
        if summary_metrics:
            summary_df = pd.DataFrame(summary_metrics)
            self._generate_summary_report(summary_df)
        else:
            self.logger.warning("No successful results to generate summary report")
            summary_df = pd.DataFrame()

        # Save all results
        self.save_results()

        return summary_df

    def _generate_summary_report(self, summary_df: pd.DataFrame):
        """Generate summary report with visualizations"""
        if summary_df.empty:
            self.logger.warning("Cannot generate summary report: No data available")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Plot distributions
            sns.histplot(data=summary_df, x="sharpe_ratio", ax=axes[0, 0])
            axes[0, 0].set_title("Distribution of Sharpe Ratios")

            sns.histplot(data=summary_df, x="max_drawdown", ax=axes[0, 1])
            axes[0, 1].set_title("Distribution of Maximum Drawdowns")

            # Plot correlations
            metrics_corr = summary_df.drop("ticker", axis=1).corr()
            sns.heatmap(metrics_corr, annot=True, fmt=".2f", ax=axes[1, 0])
            axes[1, 0].set_title("Correlation of Metrics")

            # Plot performance comparison
            summary_df.plot(
                x="ticker",
                y=["sharpe_ratio", "sortino_ratio"],
                kind="bar",
                ax=axes[1, 1],
            )
            axes[1, 1].set_title("Performance Comparison")
            axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(self.output_dir / "strategy_summary.png")

            # Save summary statistics
            summary_stats = summary_df.describe()
            summary_stats.to_csv(self.output_dir / "summary_statistics.csv")

            # Log summary statistics
            self.logger.info("\nStrategy Summary Statistics:")
            self.logger.info(f"\n{summary_stats}")

        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")

    def save_results(self):
        """Save all results and configurations"""
        try:
            import json

            # Save results
            results_file = self.output_dir / "strategy_results.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=4)

            self.logger.info(f"Results saved to {results_file}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")


def main():
    # Initialize and run enhanced strategy
    strategy = EnhancedStrategy()
    summary_df = strategy.run_strategy()

    if not summary_df.empty:
        print("\nStrategy execution completed. Summary of results:")
        print(summary_df.describe())
    else:
        print("\nStrategy execution completed with no successful results.")


if __name__ == "__main__":
    main()
