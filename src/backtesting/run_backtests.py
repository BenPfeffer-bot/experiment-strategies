import os
import sys
import json
from datetime import datetime

# Add project root to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pandas as pd
from src.utils.config import MARKET_DATA_DIR
from src.backtesting.backtester import backtest_strategy


def save_backtest_results(results_data, base_dir="db/backtesting"):
    """
    Save backtest results to a JSON file with timestamp.
    """
    # Create timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure the directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Create the filename with timestamp
    filename = os.path.join(base_dir, f"report_{timestamp}.json")

    # Save the results
    with open(filename, "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"\nBacktest results saved to: {filename}")


def main():
    # Get list of signal files
    signal_files = [
        f
        for f in os.listdir(os.path.join(MARKET_DATA_DIR, "processed"))
        if f.endswith("_integrated_signals.csv")
    ]

    # Extract tickers from filenames
    tickers = [f.replace("_integrated_signals.csv", "") for f in signal_files]

    # Store results
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_stocks": len(tickers),
            "risk_per_trade": 0.03,
            "stop_loss_pct": 0.03,
            "initial_balance": 10000,
        },
        "runs": [],
    }
    successful_backtests = 0

    # Run backtest for each ticker
    for ticker in tickers:
        try:
            print(f"\nBacktesting {ticker}...")

            # Run backtest with trade logging
            performance, trade_log = backtest_strategy(
                ticker=ticker,
                initial_balance=10000,
                risk_per_trade=0.03,
                stop_loss_pct=0.03,
                return_trade_log=True,  # New parameter to get trade log
            )

            if performance and trade_log is not None:
                # Format trades for JSON
                formatted_trades = []
                for trade in trade_log:
                    if trade["side"] == "sell":  # Only include completed trades
                        formatted_trade = {
                            "entry_date": trade["entry_date"].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "entry_price": float(trade["entry_price"]),
                            "position": trade["position"],
                            "size": float(trade["size"]),
                            "exit_date": trade["exit_date"].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "exit_price": float(trade["exit_price"]),
                            "pnl": float(trade["pnl"]),
                            "pnl_pct": float(trade["pnl_pct"]),
                        }
                        formatted_trades.append(formatted_trade)

                # Calculate additional metrics
                winning_trades = sum(1 for t in formatted_trades if t["pnl"] > 0)
                total_trades = len(formatted_trades)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                avg_profit_pct = (
                    sum(t["pnl_pct"] for t in formatted_trades) / total_trades
                    if total_trades > 0
                    else 0
                )

                # Add run results
                run_result = {
                    "ticker": ticker.replace(".", "_"),
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate": win_rate,
                    "avg_profit_pct": avg_profit_pct,
                    "final_net_worth": float(performance["final_balance"]),
                    "trades": formatted_trades,
                }
                results["runs"].append(run_result)
                successful_backtests += 1

        except Exception as e:
            print(f"Error backtesting {ticker}: {str(e)}")
            continue

    # Update metadata with final statistics
    results["metadata"]["successful_backtests"] = successful_backtests
    if successful_backtests > 0:
        results["metadata"].update(
            {
                "avg_return": sum(run["final_net_worth"] for run in results["runs"])
                / successful_backtests
                / 10000
                - 1,
                "avg_win_rate": sum(run["win_rate"] for run in results["runs"])
                / successful_backtests,
                "avg_trades_per_stock": sum(
                    run["total_trades"] for run in results["runs"]
                )
                / successful_backtests,
            }
        )

    # Save results to JSON file
    save_backtest_results(results)

    # Print summary
    print("\nOverall Results:")
    print(
        f"Successfully backtested {successful_backtests} out of {len(tickers)} stocks"
    )
    if successful_backtests > 0:
        print(f"Average Return: {results['metadata']['avg_return']:.2%}")
        print(f"Average Win Rate: {results['metadata']['avg_win_rate']:.2%}")
        print(
            f"Average Trades per Stock: {results['metadata']['avg_trades_per_stock']:.1f}"
        )


if __name__ == "__main__":
    main()
