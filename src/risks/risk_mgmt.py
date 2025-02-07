# File: src/risk/risk_management.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    volatility: float
    var: float  # Value at Risk
    cvar: float  # Conditional Value at Risk
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float


class RiskManager:
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        target_volatility: float = 0.15,
        max_position_size: float = 0.25,
        max_leverage: float = 1.5,
        confidence_level: float = 0.95,
    ):
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.confidence_level = confidence_level

    def calculate_position_size(
        self,
        account_size: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: float,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate optimal position size using risk-based sizing
        """
        # Basic Kelly Criterion calculation
        win_rate = 0.5  # Default assumption
        risk_reward = abs((entry_price - stop_loss) / stop_loss)
        kelly_fraction = win_rate - ((1 - win_rate) / risk_reward)

        # Apply half-Kelly for more conservative sizing
        kelly_fraction *= 0.5

        # Calculate position size based on account risk
        risk_amount = account_size * risk_per_trade
        max_loss_per_share = abs(entry_price - stop_loss)

        # Base position size on risk amount
        position_size = risk_amount / max_loss_per_share

        # Adjust for volatility if provided
        if volatility is not None:
            vol_adjustment = self.target_volatility / volatility
            position_size *= vol_adjustment

        # Apply Kelly fraction
        position_size *= kelly_fraction

        # Ensure position size doesn't exceed max allowed
        max_allowed = account_size * self.max_position_size
        position_size = min(position_size, max_allowed / entry_price)

        return position_size

    def calculate_portfolio_weights(
        self, returns: pd.DataFrame, method: str = "equal_risk_contribution"
    ) -> np.ndarray:
        """
        Calculate optimal portfolio weights using various methods
        """
        if method == "equal_risk_contribution":
            return self._equal_risk_contribution(returns)
        elif method == "minimum_variance":
            return self._minimum_variance(returns)
        elif method == "maximum_sharpe":
            return self._maximum_sharpe(returns)
        else:
            raise ValueError(f"Unknown portfolio optimization method: {method}")

    def _equal_risk_contribution(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate Equal Risk Contribution portfolio weights
        """
        n_assets = returns.shape[1]
        cov_matrix = returns.cov().values

        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            asset_contribs = np.dot(cov_matrix, weights) * weights / portfolio_vol
            return np.sum((asset_contribs - portfolio_vol / n_assets) ** 2)

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # weights sum to 1
            {"type": "ineq", "fun": lambda x: x},  # long only
        ]

        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _minimum_variance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate Minimum Variance portfolio weights
        """
        n_assets = returns.shape[1]
        cov_matrix = returns.cov().values

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _maximum_sharpe(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate Maximum Sharpe Ratio portfolio weights
        """
        n_assets = returns.shape[1]
        mean_returns = returns.mean() * 252  # Annualized returns
        cov_matrix = returns.cov() * 252  # Annualized covariance

        def neg_sharpe_ratio(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a return series
        """
        # Annualize factors
        sqrt_252 = np.sqrt(252)

        # Basic metrics
        volatility = returns.std() * sqrt_252
        mean_return = returns.mean() * 252

        # VaR and CVaR
        var = -np.percentile(returns, (1 - self.confidence_level) * 100)
        cvar = -returns[returns <= -var].mean()

        # Downside returns for Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = (
            downside_returns.std() * sqrt_252 if len(downside_returns) > 0 else np.inf
        )

        # Calculate ratios
        excess_return = mean_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility != 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std != 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar_ratio = excess_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return RiskMetrics(
            volatility=volatility,
            var=var,
            cvar=cvar,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
        )

    def calculate_rolling_volatility(
        self, prices: pd.Series, window: int = 20, min_periods: int = 5
    ) -> pd.Series:
        """
        Compute rolling annualized volatility from daily returns
        """
        returns = prices.pct_change().dropna()
        daily_vol = returns.rolling(window=window, min_periods=min_periods).std()
        annual_vol = daily_vol * np.sqrt(252)
        return annual_vol

    def adjust_for_correlation(
        self, position_sizes: Dict[str, float], correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Adjust position sizes based on correlation between assets
        """
        assets = list(position_sizes.keys())
        weights = np.array([position_sizes[asset] for asset in assets])

        # Create correlation-based adjustment factors
        adjustments = np.ones(len(assets))
        for i, asset in enumerate(assets):
            # Calculate average correlation with other assets
            correlations = correlation_matrix.loc[asset, assets].drop(asset)
            avg_correlation = correlations.mean()

            # Reduce position size for highly correlated assets
            if avg_correlation > 0.5:
                adjustments[i] = 1 - (avg_correlation - 0.5)

        # Apply adjustments
        adjusted_weights = weights * adjustments

        # Normalize to maintain total exposure
        total_exposure = np.sum(np.abs(weights))
        if total_exposure > 0:
            adjusted_weights = adjusted_weights * (
                total_exposure / np.sum(np.abs(adjusted_weights))
            )

        return dict(zip(assets, adjusted_weights))


if __name__ == "__main__":
    # Example usage
    risk_manager = RiskManager()

    # Test position sizing
    position_size = risk_manager.calculate_position_size(
        account_size=100000,
        risk_per_trade=0.02,
        entry_price=100,
        stop_loss=95,
        volatility=0.2,
    )
    print(f"Calculated position size: {position_size:.2f} shares")

    # Test portfolio optimization with sample data
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(252, 4) * 0.02, columns=["Asset1", "Asset2", "Asset3", "Asset4"]
    )

    weights = risk_manager.calculate_portfolio_weights(
        returns, method="equal_risk_contribution"
    )
    print("\nOptimal portfolio weights:")
    for asset, weight in zip(returns.columns, weights):
        print(f"{asset}: {weight:.4f}")

    # Test risk metrics
    metrics = risk_manager.calculate_risk_metrics(returns["Asset1"])
    print("\nRisk Metrics for Asset1:")
    print(f"Volatility: {metrics.volatility:.4f}")
    print(f"VaR: {metrics.var:.4f}")
    print(f"CVaR: {metrics.cvar:.4f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.4f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.4f}")
