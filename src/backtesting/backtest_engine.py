import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from datetime import datetime
import joblib
from src.utils.optimization import BayesianOptimizer
import logging

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

logger = logging.getLogger(__name__)


@dataclass
class TradeStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    var_95: float
    expected_shortfall: float
    kelly_fraction: float


@dataclass
class BacktestMetrics:
    """Class to hold backtest performance metrics"""

    returns: np.ndarray
    portfolio_values: np.ndarray
    trades: List[Dict]

    @property
    def sharpe_ratio(self) -> float:
        if len(self.returns) < 2:
            return np.nan
        return np.mean(self.returns) / np.std(self.returns) * np.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        if len(self.returns) < 2:
            return np.nan
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        return np.mean(self.returns) / np.std(downside_returns) * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        if len(self.portfolio_values) < 2:
            return np.nan
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (self.portfolio_values - peak) / peak
        return np.min(drawdown)

    @property
    def calmar_ratio(self) -> float:
        if len(self.returns) < 2:
            return np.nan
        max_dd = abs(self.max_drawdown)
        if max_dd == 0:
            return np.inf
        return np.mean(self.returns) * 252 / max_dd

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return np.nan
        winning_trades = sum(1 for trade in self.trades if trade["type"] == "buy")
        return winning_trades / len(self.trades)


class BacktestEngine:
    def __init__(
        self,
        strategy,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        position_size: float = 0.1,
        transaction_costs: float = 0.001,
        benchmark_data: Optional[pd.DataFrame] = None,
    ):
        self.strategy = strategy
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_costs = transaction_costs
        self.benchmark_data = benchmark_data

        # Initialize performance tracking
        self.portfolio_value = []
        self.positions = []
        self.trades = []
        self.returns = []
        self.benchmark_returns = []

        # Risk metrics
        self.var_window = 20
        self.confidence_level = 0.95
        self.risk_free_rate = 0.02  # Annual risk-free rate

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) < 2:
            return 0.0

        cum_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / rolling_max - 1
        return np.min(drawdowns)

    def calculate_alpha_beta(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> tuple:
        """Calculate alpha and beta relative to benchmark"""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0

        # Calculate beta
        covar = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_var = np.var(benchmark_returns)
        beta = covar / benchmark_var if benchmark_var != 0 else 0

        # Calculate alpha
        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
        return alpha, beta

    def run_backtest(self):
        """Execute backtest with enhanced risk management"""
        capital = self.initial_capital
        position = 0
        self.trades = []
        returns = []
        portfolio_values = [capital]

        for i in range(len(self.data)):
            # Get strategy signal
            signal = self.strategy.generate_signal(self.data.iloc[: i + 1])

            # Convert signal to scalar if it's an array
            if isinstance(signal, (np.ndarray, pd.Series)):
                signal = signal[-1]

            # Apply position sizing and risk management
            new_position = self._calculate_position_size(signal, capital, i)

            # Execute trade if position changes
            if new_position != position:
                trade_cost = (
                    abs(new_position - position)
                    * self.data["price"].iloc[i]
                    * self.transaction_costs
                )
                capital -= trade_cost

                if new_position != 0:  # Opening or modifying position
                    self.trades.append(
                        {
                            "date": self.data.index[i],
                            "type": "buy" if new_position > position else "sell",
                            "price": self.data["price"].iloc[i],
                            "size": abs(new_position - position),
                            "cost": trade_cost,
                        }
                    )

            position = new_position

            # Calculate returns and portfolio value
            position_value = position * self.data["price"].iloc[i]
            portfolio_value = capital + position_value

            if len(portfolio_values) > 0:
                daily_return = (
                    portfolio_value - portfolio_values[-1]
                ) / portfolio_values[-1]
                returns.append(daily_return)

            portfolio_values.append(portfolio_value)

        # Calculate metrics
        if len(returns) > 0:
            metrics = BacktestMetrics(
                returns=np.array(returns),
                portfolio_values=np.array(portfolio_values),
                trades=self.trades,
            )
            return metrics

        return None

    def _calculate_position_size(
        self, signal: float, capital: float, index: int
    ) -> float:
        """Calculate position size with dynamic risk adjustment"""
        if signal == 0:
            return 0

        # Calculate volatility-adjusted position size
        volatility = (
            self.data["variance"].iloc[index]
            if "variance" in self.data.columns
            else 0.02
        )
        vol_scalar = np.sqrt(252) * np.sqrt(volatility)
        target_risk = 0.02  # 2% target daily VaR

        # Kelly criterion position sizing
        win_rate = (
            sum(r > 0 for r in self.returns[-100:]) / 100
            if len(self.returns) > 100
            else 0.5
        )
        avg_win = (
            np.mean([r for r in self.returns[-100:] if r > 0])
            if len(self.returns) > 100
            else 0.01
        )
        avg_loss = (
            abs(np.mean([r for r in self.returns[-100:] if r < 0]))
            if len(self.returns) > 100
            else 0.01
        )

        kelly_fraction = (
            win_rate - ((1 - win_rate) / (avg_win / avg_loss)) if avg_loss > 0 else 0
        )
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Limit to half-Kelly

        # Combine position sizing approaches
        base_position = capital * self.position_size
        vol_adjusted = (
            base_position * (target_risk / vol_scalar)
            if vol_scalar > 0
            else base_position
        )
        final_position = vol_adjusted * kelly_fraction * np.sign(signal)

        return final_position

    def _calculate_performance_metrics(self) -> TradeStats:
        """Calculate comprehensive performance metrics"""
        returns_array = np.array(self.returns)
        benchmark_array = (
            np.array(self.benchmark_returns) if self.benchmark_returns else None
        )

        # Basic trade statistics
        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        trade_returns = (
            pd.Series([t["price"] for t in self.trades[1:]])
            / pd.Series([t["price"] for t in self.trades[:-1]])
            - 1
        )
        winning_trades = sum(trade_returns > 0)

        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(returns_array)
        sharpe = self.calculate_sharpe_ratio(returns_array)
        sortino = self.calculate_sortino_ratio(returns_array)

        # Alpha and Beta
        if benchmark_array is not None:
            alpha, beta = self.calculate_alpha_beta(returns_array, benchmark_array)
            information_ratio = (
                np.mean(returns_array - benchmark_array)
                / np.std(returns_array - benchmark_array)
                if np.std(returns_array - benchmark_array) != 0
                else 0
            )
        else:
            alpha = beta = information_ratio = 0

        # Value at Risk and Expected Shortfall
        var_95 = np.percentile(returns_array, 5)
        expected_shortfall = np.mean(returns_array[returns_array <= var_95])

        # Kelly Criterion
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        kelly_fraction = (
            win_rate - ((1 - win_rate) / (avg_win / avg_loss)) if avg_loss > 0 else 0
        )

        # Calculate Calmar ratio
        total_return = self.portfolio_value[-1] / self.initial_capital - 1
        calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        return TradeStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=abs(wins.sum() / losses.sum())
            if len(losses) > 0
            else float("inf"),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            kelly_fraction=kelly_fraction,
        )

    def plot_results(self, save_path: Optional[str] = None):
        """Generate comprehensive performance visualizations"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2)

        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        equity_curve = pd.Series(self.portfolio_value, index=self.data.index)
        equity_curve.plot(ax=ax1, label="Portfolio Value")
        if self.benchmark_data is not None:
            benchmark_curve = (
                1 + pd.Series(self.benchmark_returns, index=self.data.index)
            ).cumprod()
            benchmark_curve.plot(ax=ax1, label="Benchmark", alpha=0.7)
        ax1.set_title("Equity Curve")
        ax1.legend()

        # Returns distribution
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(self.returns, ax=ax2, kde=True)
        ax2.axvline(x=0, color="r", linestyle="--", alpha=0.5)
        ax2.set_title("Returns Distribution")

        # Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        drawdown = (
            pd.Series(self.returns).cumsum() - pd.Series(self.returns).cumsum().cummax()
        )
        drawdown.plot(ax=ax3)
        ax3.set_title("Drawdown")

        # Position over time
        ax4 = fig.add_subplot(gs[2, 0])
        pd.Series(self.positions, index=self.data.index).plot(ax=ax4)
        ax4.set_title("Position Size")

        # Rolling Sharpe ratio
        ax5 = fig.add_subplot(gs[2, 1])
        returns_series = pd.Series(self.returns)
        rolling_sharpe = returns_series.rolling(252).apply(
            lambda x: self.calculate_sharpe_ratio(x)
        )
        rolling_sharpe.plot(ax=ax5)
        ax5.set_title("Rolling Sharpe Ratio")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_results(self, filepath: str):
        """Save backtest results and metrics"""
        results = {
            "portfolio_value": self.portfolio_value,
            "positions": self.positions,
            "trades": self.trades,
            "returns": self.returns,
            "benchmark_returns": self.benchmark_returns,
            "metrics": self._calculate_performance_metrics().__dict__,
        }
        joblib.dump(results, filepath)

    def load_results(self, filepath: str):
        """Load saved backtest results"""
        results = joblib.load(filepath)
        self.portfolio_value = results["portfolio_value"]
        self.positions = results["positions"]
        self.trades = results["trades"]
        self.returns = results["returns"]
        self.benchmark_returns = results["benchmark_returns"]
        return TradeStats(**results["metrics"])

    def optimize_parameters(self, data, ticker):
        """Optimize strategy parameters using Bayesian optimization."""
        try:
            # Define parameter space
            param_space = {
                "lookback_period": (10, 100),
                "volatility_window": (5, 60),
                "momentum_window": (5, 60),
                "position_size": (0.1, 0.3),
            }

            def objective_function(params):
                try:
                    # Convert integer parameters
                    params["lookback_period"] = int(params["lookback_period"])
                    params["volatility_window"] = int(params["volatility_window"])
                    params["momentum_window"] = int(params["momentum_window"])
                    params["position_size"] = float(params["position_size"])

                    # Run backtest with these parameters
                    results = self._run_backtest(data, params)
                    if results is None or "returns" not in results:
                        return np.inf

                    returns = np.array(results["returns"])
                    if len(returns) < 2:
                        return np.inf

                    # Calculate Sharpe ratio
                    sharpe = self.calculate_sharpe_ratio(returns)
                    if np.isnan(sharpe):
                        return np.inf

                    return -sharpe  # Negative because we're minimizing

                except Exception as e:
                    logger.error(f"Error in objective function: {str(e)}")
                    return np.inf

            # Create optimizer instance
            optimizer = BayesianOptimizer(
                param_space=param_space,
                objective_function=objective_function,
                n_initial_points=5,
                n_iterations=10,
            )

            # Run optimization
            best_params, best_value = optimizer.optimize()

            if best_params is not None:
                # Convert parameters to appropriate types
                return {
                    "lookback_period": int(best_params["lookback_period"]),
                    "volatility_window": int(best_params["volatility_window"]),
                    "momentum_window": int(best_params["momentum_window"]),
                    "position_size": float(best_params["position_size"]),
                }

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")

        return None
