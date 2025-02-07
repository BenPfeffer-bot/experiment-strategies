# File: src/strategies/meta_strategy.py

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import indicator modules
from src.indicators.rsi import RSIMultiLength
from src.indicators.fibonacci import FibonacciTrailingStop
from src.indicators.macd import compute_macd
from src.indicators.moving_average import MovingAverageForecast
from src.indicators.variance import compute_variance
from src.indicators.kelter_channel import compute_kelter_channels

# Import configuration variables
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR


def load_price_data(ticker):
    """
    Helper function to load price data (assumes CSV with 'Close' column).
    """
    filepath = os.path.join(MARKET_DATA_DIR, f"{ticker}_data.csv")
    df = pd.read_csv(filepath)
    # If there's a 'Date' column, use it as datetime index. Otherwise assume index is datetime.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.set_index("Date", inplace=True)
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    # Normalize dates and remove duplicates
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def generate_signals(prices: pd.Series):
    """
    Generate a simple integrated signal from the various indicators.

    Returns a DataFrame with indicator values and a final combined signal:
      - 1 for long signal,
      - -1 for short signal,
      - 0 for no signal.
    """
    # Ensure index is normalized and sorted
    prices.index = prices.index.normalize()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.sort_index()

    # Initialize indicator classes
    rsi_indicator = RSIMultiLength(
        max_length=20, min_length=10, overbought=70, oversold=30
    )
    fib_stop = FibonacciTrailingStop(
        L=20, R=1, use_labels=False, F=-0.382, use_f=False, trigger="close"
    )
    ma_forecaster = MovingAverageForecast(
        window=20, max_memory=50, forecast_length=100, up_per=80, mid_per=50, dn_per=20
    )

    # Compute RSI signals
    rsi_data = rsi_indicator.compute(prices)
    # Define oversold/overbought conditions for RSI
    rsi_long = rsi_data["avg_rsi"] < 40  # oversold -> potential long
    rsi_short = rsi_data["avg_rsi"] > 60  # overbought -> potential short

    # Create an OHLC DataFrame from prices (used for Fibonacci, variance, and kelter channels)
    df_ohlc = pd.DataFrame(
        {
            "close": prices,
            "high": prices * 1.01,  # example: 1% above close
            "low": prices * 0.99,  # example: 1% below close
        },
        index=prices.index,
    )

    # Compute Fibonacci Trailing Stop
    fib_data = fib_stop.compute(df_ohlc)
    # Trend direction from Fibonacci: uptrend if dir == 1, downtrend if dir == -1.
    fib_long = fib_data["dir"] == 1
    fib_short = fib_data["dir"] == -1

    # Compute MACD
    df_macd = pd.DataFrame({"close": prices}, index=prices.index)
    macd_data = compute_macd(df_macd)
    # Simple MACD signal:
    macd_long = macd_data["macd"] > macd_data["signal"]
    macd_short = macd_data["macd"] < macd_data["signal"]

    # Compute Moving Average Forecast
    ma_series, forecasts = ma_forecaster.compute(prices)
    ma_long = prices > ma_series
    ma_short = prices < ma_series

    # Compute Variance (volatility) signal
    variance_series = compute_variance(df_ohlc, length=20, src="close")
    # Use the median variance over the entire series: low variance (below median) is considered favorable
    variance_median = variance_series.median()
    variance_long = variance_series < variance_median
    variance_short = variance_series > variance_median

    # Compute Kelter Channels signal
    channels = compute_kelter_channels(df_ohlc, period=14, multiplier=1.5)
    # If price is below lower_channel, signal long; if above upper_channel, signal short.
    kelter_long = prices < channels["lower_channel"]
    kelter_short = prices > channels["upper_channel"]

    # Combine signals with equal weighting.
    combined_signal = []
    # Adjust thresholds since we're adding more indicators. Here, if sum >= 3 then long, <= -3 then short.
    for t in range(len(prices)):
        score = 0
        # Long considerations
        if rsi_long.iloc[t]:
            score += 1
        if fib_long.iloc[t]:
            score += 1
        if macd_long.iloc[t]:
            score += 1
        if ma_long.iloc[t]:
            score += 1
        if variance_long.iloc[t]:
            score += 1
        if kelter_long.iloc[t]:
            score += 1

        # Short considerations
        if rsi_short.iloc[t]:
            score -= 1
        if fib_short.iloc[t]:
            score -= 1
        if macd_short.iloc[t]:
            score -= 1
        if ma_short.iloc[t]:
            score -= 1
        if variance_short.iloc[t]:
            score -= 1
        if kelter_short.iloc[t]:
            score -= 1

        # Decide final signal based on aggregated score
        if score >= 3:
            combined_signal.append(1)
        elif score <= -3:
            combined_signal.append(-1)
        else:
            combined_signal.append(0)

    # Build a DataFrame with all outputs
    df_signals = pd.DataFrame(
        {
            "price": prices,
            "avg_rsi": rsi_data["avg_rsi"],
            "rsi_signal": np.where(rsi_long, 1, np.where(rsi_short, -1, 0)),
            "fib_dir": fib_data["dir"],
            "macd_macd": macd_data["macd"],
            "macd_signal": macd_data["signal"],
            "ma": ma_series,
            "variance": variance_series,
            "kelter_upper": channels["upper_channel"],
            "kelter_lower": channels["lower_channel"],
            "combined_signal": combined_signal,
        },
        index=prices.index,
    )

    return df_signals


def main():
    """
    Iterates over each ticker, computes the integrated signals, and saves a CSV.
    """
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df_prices = load_price_data(ticker)
        # Assume the price extraction uses 'Close'
        prices = df_prices["Close"]
        signals_df = generate_signals(prices)

        processed_dir = os.path.join(MARKET_DATA_DIR, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        output_csv = os.path.join(processed_dir, f"{ticker}_integrated_signals.csv")
        signals_df.to_csv(output_csv, index_label="Date")
        print(f"Integrated signals for {ticker} saved to {output_csv}")


if __name__ == "__main__":
    main()
