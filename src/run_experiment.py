# File: run_experiment.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import numpy as np
import pandas as pd
from dvclive import Live

from src.backtesting.backtester import backtest_strategy


def simulate_backtest(epoch, n_periods=250):
    """
    Simulate a backtest experiment.
    For reproducibility, set the random seed based on the epoch.
    """
    np.random.seed(epoch + 1)
    dates = pd.date_range(start="2023-01-01", periods=n_periods, freq="D")
    # Simulated price series: random walk around 100
    prices = pd.Series(100 + np.cumsum(np.random.randn(n_periods)), index=dates)
    # Generate a random signal series: -1, 0, or 1 with probabilities
    signals = np.random.choice([-1, 0, 1], size=n_periods, p=[0.3, 0.4, 0.3])
    signals_df = pd.DataFrame(
        {"price": prices, "combined_signal": signals}, index=dates
    )

    trade_log, final_balance, performance = backtest_strategy(
        signals_df, initial_account=100000, risk_percent=0.01, ticker="DEMO"
    )
    return final_balance, performance


def main():
    # Get number of epochs from command-line argument (default to 10)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # Initialize DVCLive with save_dvc_exp=True
    with Live(save_dvc_exp=True) as live:
        live.log_param("epochs", epochs)

        for epoch in range(epochs):
            final_balance, performance = simulate_backtest(epoch)

            # Log experiment metrics to DVCLive
            live.log_metric("backtest/final_balance", final_balance)
            live.log_metric(
                "backtest/cumulative_return", performance["cumulative_return"]
            )
            live.log_metric("backtest/max_drawdown", performance["max_drawdown"])
            live.log_metric("backtest/sharpe_ratio", performance["sharpe_ratio"])

            # Move to the next step (epoch)
            live.next_step()

            print(
                f"Epoch {epoch + 1}/{epochs} completed: Final Balance = {final_balance}, Performance = {performance}"
            )


if __name__ == "__main__":
    main()
