# File: src/backtesting/backtester.py
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Imports from our modules
from src.risk.risk_management import calculate_position_size, calculate_rolling_volatility
from src.execution.executor import SimulatedBroker

def compute_performance(trade_log, initial_account):
    """
    Compute performance metrics based on trade log.
    Returns cumulative returns, maximum drawdown, and Sharpe ratio.
    """
    # Build equity curve from trade log based on balance updates.
    df = pd.DataFrame(trade_log)
    df['balance'] = df['balance'].astype(float)
    df.sort_values(by="time", inplace=True)
    df.set_index("time", inplace=True)
    df = df.resample('D').ffill().dropna()
    equity_curve = df['balance']
    returns = equity_curve.pct_change().dropna()
    cumulative_return = (equity_curve.iloc[-1] / initial_account) - 1
    # Max drawdown:
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    # Sharpe ratio (assume risk-free rate = 0)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else np.nan
    metrics = {
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe
    }
    return metrics

def backtest_strategy(signals_df, initial_account=100000, risk_percent=0.01, ticker="TICKER"):
    """
    Backtester that cycles through integrated signals and simulates trade execution.
    Uses:
        - signals_df: DataFrame with at least "price" and "combined_signal".
        - A fixed stop loss rule (e.g., 2% adverse price move).
        - Dynamic position sizing based on risk management.
    
    Returns:
        trade_log: List of trade entries/exits.
        final_balance: Final account value.
        performance: Dictionary containing performance metrics.
    """
    broker = SimulatedBroker()
    balance = initial_account
    position = 0         # positive for long, negative for short
    entry_price = None
    trade_log = []
    stop_loss_long = 0.98
    stop_loss_short = 1.02

    # Calculate rolling volatility for enhanced risk management
    volatility = calculate_rolling_volatility(signals_df['price'], window=20)

    for time, row in signals_df.iterrows():
        price = row['price']
        signal = row['combined_signal']
        vol = volatility.loc[time] if time in volatility.index else np.nan
        
        # Exit conditions: Check if an adverse move has occurred.
        if position > 0:
            if signal == -1 or price < entry_price * stop_loss_long:
                broker.send_order(ticker, "sell", position, price, time)
                profit = (price - entry_price) * position
                balance += profit
                trade_log.append({
                    "time": time,
                    "side": "sell",
                    "price": price,
                    "position": position,
                    "pnl": profit,
                    "balance": balance
                })
                position = 0
                entry_price = None
        elif position < 0:
            if signal == 1 or price > entry_price * stop_loss_short:
                broker.send_order(ticker, "buy", abs(position), price, time)
                profit = (entry_price - price) * abs(position)
                balance += profit
                trade_log.append({
                    "time": time,
                    "side": "buy",
                    "price": price,
                    "position": position,
                    "pnl": profit,
                    "balance": balance
                })
                position = 0
                entry_price = None

        # Entry condition: if no open position and a non-zero signal.
        if position == 0 and signal != 0:
            if signal == 1:  # long signal
                # Adjust stop loss using volatility if available:
                adjusted_stop = price * stop_loss_long if pd.isna(vol) else price - 1.5 * vol * price
                pos_size = calculate_position_size(balance, risk_percent, price, adjusted_stop)
                pos_size = int(max(pos_size, 1))
                broker.send_order(ticker, "buy", pos_size, price, time)
                entry_price = price
                position = pos_size
                trade_log.append({
                    "time": time,
                    "side": "buy",
                    "price": price,
                    "position": pos_size,
                    "pnl": 0,
                    "balance": balance
                })
            elif signal == -1:  # short signal
                adjusted_stop = price * stop_loss_short if pd.isna(vol) else price + 1.5 * vol * price
                pos_size = calculate_position_size(balance, risk_percent, price, adjusted_stop)
                pos_size = int(max(pos_size, 1))
                broker.send_order(ticker, "sell", pos_size, price, time)
                entry_price = price
                position = -pos_size
                trade_log.append({
                    "time": time,
                    "side": "sell",
                    "price": price,
                    "position": -pos_size,
                    "pnl": 0,
                    "balance": balance
                })
    
    performance = compute_performance(trade_log, initial_account)
    return trade_log, balance, performance

# Example usage:
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    # np.random.seed(42)
    # dates = pd.date_range(start="2023-01-01", periods=250, freq="D")
    # Simulated price series: Random walk around 100
    prices = pd.Series(100 + np.cumsum(np.random.randn(250)), index=dates)
    # Build a simple signal series for demonstration (1, -1, 0 with some persistence)
    signals = np.random.choice([-1, 0, 1], size=250, p=[0.3, 0.4, 0.3])
    signals_df = pd.DataFrame({"price": prices, "combined_signal": signals}, index=dates)
    
    trade_log, final_balance, performance = backtest_strategy(signals_df, initial_account=100000, risk_percent=0.01, ticker="DEMO")
    print("Final account balance:", final_balance)
    print("Performance metrics:", performance)
    print("Trade log:")
    print(pd.DataFrame(trade_log))
