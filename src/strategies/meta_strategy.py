# File: src/strategies/meta_strategy.py

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import indicator modules
from src.indicators.rsi import RSIMultiLength
from src.indicators.fibonacci import FibonacciTrailingStop
from src.indicators.macd import compute_macd
from src.indicators.moving_average import MovingAverageForecast

# Import configuration variables
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR

def load_price_data(ticker):
    """
    Helper function to load price data (assumes CSV with 'Close' column).
    """
    filepath = os.path.join(MARKET_DATA_DIR, f"{ticker}_data.csv")
    df = pd.read_csv(filepath)
    # Assuming the CSV has a column 'Close', and an index or date column.
    # If no datetime index, create one.
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    return df

def generate_signals(prices: pd.Series):
    """
    Generate a simple integrated signal from the indicators.
    
    Returns a DataFrame with indicator values and a final signal:
      - 1 for long signal,
      - -1 for short signal,
      - 0 for no signal.
    """
    # Initialize indicator classes
    rsi_indicator = RSIMultiLength(max_length=20, min_length=10, overbought=70, oversold=30)
    fib_stop = FibonacciTrailingStop(L=20, R=1, use_labels=False, F=-0.382, use_f=False, trigger='close')
    ma_forecaster = MovingAverageForecast(window=20, max_memory=50, forecast_length=100,
                                          up_per=80, mid_per=50, dn_per=20)
    
    # Compute RSI signals
    rsi_data = rsi_indicator.compute(prices)
    # Define a simple boolean for oversold/overbought condition
    rsi_long = rsi_data['avg_rsi'] < 40  # oversold for a potential long
    rsi_short = rsi_data['avg_rsi'] > 60  # overbought for a potential short
    
    # Compute Fibonacci Trailing Stop, need OHLC dataframe.
    # Here we create minimal OHLC data from Close price as an example.
    df_ohlc = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,  # Just an example modification
        'low': prices * 0.99
    }, index=prices.index)
    fib_data = fib_stop.compute(df_ohlc)
    # Choose trend direction from Fibonacci indicator: uptrend if dir==1, downtrend if dir==-1
    fib_long = fib_data['dir'] == 1
    fib_short = fib_data['dir'] == -1

    # Compute MACD
    # Create a DataFrame with close prices for MACD computation
    df_macd = pd.DataFrame({'close': prices}, index=prices.index)
    macd_data = compute_macd(df_macd)
    # Simple MACD signal:
    macd_long = (macd_data['macd'] > macd_data['signal'])
    macd_short = (macd_data['macd'] < macd_data['signal'])
    
    # Compute Moving Average Forecast
    ma_series, forecasts = ma_forecaster.compute(prices)
    # Use a simple rule: if price is above the moving average, consider long, otherwise short.
    ma_long = prices > ma_series
    ma_short = prices < ma_series
    
    # Combine signals with equal weighting (could be tuned further)
    combined_signal = []
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
        # Short considerations
        if rsi_short.iloc[t]:
            score -= 1
        if fib_short.iloc[t]:
            score -= 1
        if macd_short.iloc[t]:
            score -= 1
        if ma_short.iloc[t]:
            score -= 1
        
        # Final signal threshold: if total score >= 2, generate long signal; <= -2, short signal; otherwise flat
        if score >= 2:
            combined_signal.append(1)
        elif score <= -2:
            combined_signal.append(-1)
        else:
            combined_signal.append(0)
    
    # Build a DataFrame with all outputs
    df_signals = pd.DataFrame({
        'price': prices,
        'avg_rsi': rsi_data['avg_rsi'],
        'rsi_signal': np.where(rsi_long, 1, np.where(rsi_short, -1, 0)),
        'fib_dir': fib_data['dir'],
        'macd_macd': macd_data['macd'],
        'macd_signal': macd_data['signal'],
        'ma': ma_series,
        'combined_signal': combined_signal
    }, index=prices.index)
    
    return df_signals

def main():
    """
    Iterates over each ticker, computes the integrated signals, and saves a CSV.
    """
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df_prices = load_price_data(ticker)
        
        # Assume price extraction from 'Close'
        prices = df_prices['Close']
        
        signals_df = generate_signals(prices)
        
        if not os.path.exists(f"{MARKET_DATA_DIR}/processed"):
            os.makedirs(f"{MARKET_DATA_DIR}/processed")
        
        # For visualization purposes, you might also want to save a plot or the CSV.
        output_csv = os.path.join(MARKET_DATA_DIR, f"processed/{ticker}_integrated_signals.csv")
        signals_df.to_csv(output_csv)
        print(f"Integrated signals for {ticker} saved to {output_csv}")

if __name__ == "__main__":
    main()
