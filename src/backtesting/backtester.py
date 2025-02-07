# File: src/backtesting/backtester.py
import pandas as pd
import numpy as np
import os
import logging
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Imports from our modules
from src.utils.config import TICKERS, MARKET_DATA_DIR
from src.risks.risk_mgmt import calculate_position_size, calculate_rolling_volatility
from src.backtesting.broker import SimulatedBroker


def compute_performance(trade_log_df, initial_balance):
    """
    Compute performance metrics based on trade log DataFrame.
    Returns cumulative returns, maximum drawdown, and Sharpe ratio.
    """
    # If no trades were executed, return default metrics
    if trade_log_df.empty:
        return {"cumulative_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": np.nan}

    # Ensure all datetime entries are converted to pandas datetime
    if not pd.api.types.is_datetime64_any_dtype(trade_log_df["entry_date"]):
        trade_log_df["entry_date"] = pd.to_datetime(
            trade_log_df["entry_date"], errors="coerce"
        )
    if not pd.api.types.is_datetime64_any_dtype(trade_log_df["exit_date"]):
        trade_log_df["exit_date"] = pd.to_datetime(
            trade_log_df["exit_date"], errors="coerce"
        )
    trade_log_df = trade_log_df.dropna(subset=["entry_date", "exit_date"])

    # Create a daily balance series
    dates = pd.date_range(
        start=trade_log_df["entry_date"].min(),
        end=trade_log_df["exit_date"].max(),
        freq="D",
    )
    daily_balance = pd.Series(float(initial_balance), index=dates)

    # Update balance based on trades
    for _, trade in trade_log_df.iterrows():
        daily_balance[trade["exit_date"] :] = trade["balance"]

    # Calculate returns and metrics
    returns = daily_balance.pct_change().dropna()
    cumulative_return = (daily_balance.iloc[-1] / initial_balance) - 1

    # Max drawdown
    roll_max = daily_balance.cummax()
    drawdown = (daily_balance - roll_max) / roll_max
    max_drawdown = drawdown.min()

    # Sharpe ratio (assume risk-free rate = 0)
    sharpe = (
        np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else np.nan
    )

    metrics = {
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
    }
    return metrics


def backtest_strategy(
    ticker: str,
    initial_balance: float = 10000,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.02,
    return_trade_log: bool = False,
) -> dict:
    """
    Backtest a trading strategy for a given ticker.

    Args:
        ticker (str): The ticker symbol to backtest
        initial_balance (float): Initial account balance
        risk_per_trade (float): Percentage of account balance to risk per trade
        stop_loss_pct (float): Stop loss percentage
        return_trade_log (bool): Whether to return the trade log along with performance

    Returns:
        If return_trade_log is False:
            dict: Dictionary containing backtest results
        If return_trade_log is True:
            tuple: (performance_dict, trade_log_list)
    """
    try:
        # Read signals data
        signals_df = pd.read_csv(
            f"{MARKET_DATA_DIR}/processed/{ticker}_integrated_signals.csv"
        )

        if signals_df.empty:
            print(f"No signals data for {ticker}")
            return (None, None) if return_trade_log else None

        # Convert date to datetime and set as index
        signals_df["Date"] = pd.to_datetime(signals_df["Date"])
        signals_df.set_index("Date", inplace=True)

        # Normalize index to UTC datetime at 23:00
        signals_df.index = signals_df.index.tz_localize(None)
        signals_df.index = signals_df.index.normalize() + pd.Timedelta(hours=23)
        signals_df.index = signals_df.index.tz_localize("UTC")

        print(f"Initial shape of signals data for {ticker}: {signals_df.shape}")
        print(f"Sample normalized dates for {ticker}:")
        print(signals_df.index[0:5])

        # Check for duplicates in signals data
        if signals_df.index.duplicated().any():
            print(
                f"Found {sum(signals_df.index.duplicated())} duplicate dates in signals data for {ticker}"
            )
            print("Duplicate dates:")
            print(signals_df.index[signals_df.index.duplicated()].unique())
            signals_df = signals_df[~signals_df.index.duplicated(keep="first")]

        # Sort index
        signals_df.sort_index(inplace=True)

        print(f"Final shape of signals data for {ticker}: {signals_df.shape}")

        # Check for missing data after alignment
        if signals_df["price"].isna().any():
            print(f"Missing data found after alignment for {ticker}")
            return (None, None) if return_trade_log else None

        # Run backtest
        broker = SimulatedBroker(initial_balance=initial_balance)
        trade_log = []
        position = None

        for date, row in signals_df.iterrows():
            price = row["price"]
            signal = row["combined_signal"]

            # Entry logic
            if position is None and signal == 1:  # Enter long
                # Calculate position size based on risk and price
                stop_loss = price * (1 - stop_loss_pct)
                risk_amount = broker.balance * risk_per_trade
                # Limit position size to 25% of available balance
                max_position_cost = broker.balance * 0.25
                position_size = min(
                    risk_amount / (price - stop_loss),  # Risk-based size
                    max_position_cost / price,  # Balance-based size
                )

                # Open position
                position = broker.open_position(
                    ticker=ticker,
                    size=position_size,
                    entry_price=price,
                    stop_loss=stop_loss,
                    date=date,
                )
                if position:
                    trade_log.append(position)

            # Exit logic
            elif position is not None and signal == -1:  # Exit long
                # Close position
                trade = broker.close_position(
                    position=position, exit_price=price, date=date
                )
                if trade:
                    trade_log.append(trade)
                    position = None

            # Update stop loss
            elif position is not None:
                if price <= position["stop_loss"]:
                    # Stop loss hit
                    trade = broker.close_position(
                        position=position, exit_price=position["stop_loss"], date=date
                    )
                    if trade:
                        trade_log.append(trade)
                        position = None

        # Close any remaining positions at the last price
        if position is not None:
            trade = broker.close_position(
                position=position,
                exit_price=signals_df.iloc[-1]["price"],
                date=signals_df.index[-1],
            )
            if trade:
                trade_log.append(trade)

        # Convert trade log to DataFrame
        trade_log_df = pd.DataFrame(trade_log)

        if trade_log_df.empty:
            print(f"No trades executed for {ticker}")
            return (None, None) if return_trade_log else None

        if "balance" not in trade_log_df.columns:
            print(f"Invalid trade log for {ticker}")
            return (None, None) if return_trade_log else None

        # Compute performance metrics
        performance = compute_performance(trade_log_df, initial_balance)
        performance["final_balance"] = trade_log_df["balance"].iloc[-1]

        if return_trade_log:
            return performance, trade_log
        return performance

    except Exception as e:
        print(f"Error backtesting {ticker}: {str(e)}")
        return (None, None) if return_trade_log else None


# Example usage:
if __name__ == "__main__":
    results = {}
    for ticker in TICKERS:
        print(f"\nBacktesting {ticker}...")

        try:
            # Read the signals CSV which contains both signals and price data
            signals_df = pd.read_csv(
                f"{MARKET_DATA_DIR}/processed/{ticker}_integrated_signals.csv",
                dtype={"Date": str},
                index_col=0,
                on_bad_lines="skip",  # Handle malformed CSV lines
            )

            print(f"\nProcessing {ticker} data:")
            print(f"Initial shape: {signals_df.index.shape}")

            # Convert index to UTC datetime and normalize to 23:00 UTC
            signals_df.index = pd.to_datetime(signals_df.index, utc=True)
            signals_df.index = signals_df.index.map(
                lambda x: x.normalize() + pd.Timedelta(hours=23)
            )

            # Check for duplicates
            duplicates = signals_df.index.duplicated(keep="last")
            if duplicates.any():
                print(f"Found {duplicates.sum()} duplicate dates")
                print("Duplicate dates:")
                duplicate_dates = signals_df.index[duplicates]
                print(duplicate_dates)
                # Remove duplicates keeping the last value
                signals_df = signals_df[~duplicates]
                print(f"Shape after removing duplicates: {signals_df.index.shape}")

            print("Sample of normalized dates:")
            print(signals_df.index[:5])
            signals_df = signals_df.sort_index()
            print(f"Final data shape: {signals_df.index.shape}")

            # Verify no missing data
            if signals_df.isnull().any().any():
                print(f"Found missing data for {ticker}")
                continue

            # Run backtest
            performance, trade_log = backtest_strategy(
                ticker,
                initial_balance=10000,
                risk_per_trade=0.02,
                stop_loss_pct=0.02,
                return_trade_log=True,
            )

            # Skip output if signals_df was empty and backtest_strategy returned no trades.
            if not performance:
                print(f"Skipped {ticker} because signals data was empty.")
                continue

            # Store results
            results[ticker] = performance

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

    # Print overall results summary
    print("\nOverall Results Summary:")
    print("-" * 50)
    if results:
        avg_return = np.mean([r["cumulative_return"] for r in results.values()])
        avg_drawdown = np.mean([r["max_drawdown"] for r in results.values()])
        avg_sharpe = np.mean(
            [
                r["sharpe_ratio"]
                for r in results.values()
                if not np.isnan(r["sharpe_ratio"])
            ]
        )

        print(f"Number of stocks successfully backtested: {len(results)}")
        print(f"Average Cumulative Return: {avg_return:.2%}")
        print(f"Average Maximum Drawdown: {avg_drawdown:.2%}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    else:
        print("No successful backtests to report.")
