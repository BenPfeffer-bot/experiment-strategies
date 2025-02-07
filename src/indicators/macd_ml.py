"""
mcd_ml.py

MACD Based Price Forecasting (LuxAlgo variant)
-------------------------------------------------
This work is licensed under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
© LuxAlgo (Original Pine Script by LuxAlgo)

This module demonstrates:
  - Calculation of MACD and Signal lines from closing price data.
  - A simplified memory‐mechanism to track the trend’s price differences.
  - A forecasting routine that uses percentiles of the stored differences to produce forecast lines:
      • Upper forecast (at percentile upPer)
      • Mid forecast (at percentile midPer)
      • Lower forecast (at percentile dnPer)
  - Plotting of the current price series with forecast lines displayed at trigger events.
  
Note: This is a simplified adaptation. In the original Pine Script, a “holder”
contains a history of memory vectors (one per bar during the trend). Here we keep only
the most recent memory vector for the current trend.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR

# -----------------------------
# Parameters / Settings
# -----------------------------
# MACD settings
FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9

# Trend Determination mode: either "MACD" or "MACD - Signal"
TREND_MODE = "MACD - Signal"  # set to "MACD" for MACD crossing 0

# Forecast settings
MAX_MEMORY = 50         # maximum number of bars to remember
FCAST = 100             # forecasting length (number of future bars to forecast)
UP_PERCENTILE = 80      # top percentile for forecast
MID_PERCENTILE = 50     # average percentile for forecast
DN_PERCENTILE = 20      # bottom percentile for forecast

# -----------------------------
# Helper classes & functions
# -----------------------------
class MemoryHolder:
    """
    Holds a memory vector (list of differences) and provides forecast computation.
    """
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.data = []  # list to hold differences (most recent first)

    def populate(self, diff):
        # Insert new difference at beginning
        self.data.insert(0, diff)
        # Trim memory to maximum allowed length
        if len(self.data) > self.max_memory:
            self.data.pop()

    def forecast(self, init_price, fcast, upPer, midPer, dnPer):
        """
        Computes forecast lines.
        For simplicity, we use the current memory vector for all forecast steps.
        (In the original code, different snapshots could be used.)
        
        Returns:
           Three lists of forecast values (upper, mid, lower) of length fcast.
        """
        if len(self.data) == 0:
            # Not enough memory to forecast
            return None, None, None
        # Compute percentiles from the memory vector
        arr = np.array(self.data)
        q_upper = np.percentile(arr, upPer)
        q_mid   = np.percentile(arr, midPer)
        q_lower = np.percentile(arr, dnPer)
        # For this simplified example, we assume forecast is constant over the horizon.
        forecast_upper = [init_price + q_upper] * fcast
        forecast_mid   = [init_price + q_mid]   * fcast
        forecast_lower = [init_price + q_lower] * fcast
        return forecast_upper, forecast_mid, forecast_lower

def compute_macd(df, fast_length, slow_length, signal_length):
    """
    Computes MACD, Signal and Histogram from df['close'].
    """
    # Fast and Slow EMAs
    fast_ema = df['close'].ewm(span=fast_length, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_length, adjust=False).mean()
    macd = fast_ema - slow_ema
    # Signal: using simple moving average (rolling mean)
    signal = macd.rolling(window=signal_length, min_periods=signal_length).mean()
    hist = macd - signal
    return macd, signal, hist

def detect_cross(series1, series2):
    """
    Returns boolean Series where a cross occurs.
    A cross is detected when the sign of (series1 - series2) changes.
    """
    diff = series1 - series2
    cross = (np.sign(diff.shift(1)) != np.sign(diff))
    # Exclude the very first bar (NaN shifted)
    cross.iloc[0] = False
    return cross

# -----------------------------
# Main forecasting routine
# -----------------------------
def forecast_macd_prices(df):
    """
    Iterates over each bar in df, updating memory for current trend, and when a trigger is detected,
    performs a forecast.
    
    Returns:
       forecasts: a list of dictionaries with keys:
          'trigger_index': index (integer location) of the trigger event.
          'trend': "uptrend" or "downtrend"
          'init_price': reference price for trend.
          'forecast_upper', 'forecast_mid', 'forecast_lower': forecast lines (list)
    """
    # Compute MACD, Signal and Histogram
    macd, signal, hist = compute_macd(df, FAST_LENGTH, SLOW_LENGTH, SIGNAL_LENGTH)
    df = df.copy()
    df["macd"] = macd
    df["signal"] = signal
    df["hist"] = hist

    # Pre-compute trigger series
    if TREND_MODE == "MACD":
        # Trigger when MACD crosses 0.
        trigger_series = detect_cross(df["macd"], pd.Series(0, index=df.index))
    else:
        # Trigger when MACD crosses Signal.
        trigger_series = detect_cross(df["macd"], df["signal"])

    forecasts = []

    # Initialize memory holders and reference prices for up- and downtrends
    memory_up = MemoryHolder(MAX_MEMORY)
    memory_down = MemoryHolder(MAX_MEMORY)
    uptrend_init_price = df['close'].iloc[0]
    downtrend_init_price = df['close'].iloc[0]
    # Trend index counters (not used further in this simplified version)
    up_idx = 0
    dn_idx = 0

    # Iterate through rows (by index location)
    for i in range(len(df)):
        if np.isnan(df["macd"].iloc[i]) or np.isnan(df["signal"].iloc[i]):
            continue

        cur_close = df["close"].iloc[i]
        cur_macd = df["macd"].iloc[i]
        cur_signal = df["signal"].iloc[i]

        # Determine trend based on selected mode
        if TREND_MODE == "MACD":
            is_uptrend = cur_macd > 0
        else:  # TREND_MODE == "MACD - Signal"
            is_uptrend = cur_macd > cur_signal

        is_downtrend = not is_uptrend

        # Update reference price at trend "reset" (i.e. when trend starts anew)
        if is_uptrend and (i == 0 or (df["macd"].iloc[i-1] <= df["signal"].iloc[i-1] if TREND_MODE=="MACD - Signal" else df["macd"].iloc[i-1] <= 0)):
            uptrend_init_price = cur_close
            # Reset uptrend memory
            memory_up = MemoryHolder(MAX_MEMORY)
            up_idx = 0
        if is_downtrend and (i == 0 or (df["macd"].iloc[i-1] >= df["signal"].iloc[i-1] if TREND_MODE=="MACD - Signal" else df["macd"].iloc[i-1] >= 0)):
            downtrend_init_price = cur_close
            # Reset downtrend memory
            memory_down = MemoryHolder(MAX_MEMORY)
            dn_idx = 0

        # Populate memory: record the difference from the trend’s initial price
        if is_uptrend:
            diff = cur_close - uptrend_init_price
            memory_up.populate(diff)
            up_idx += 1
        else:
            diff = cur_close - downtrend_init_price
            memory_down.populate(diff)
            dn_idx += 1

        # Check for trigger event at this bar
        if trigger_series.iloc[i]:
            # Choose memory and reference price based on current trend
            if is_uptrend:
                forecasts_upper, forecasts_mid, forecasts_lower = memory_up.forecast(uptrend_init_price, FCAST, UP_PERCENTILE, MID_PERCENTILE, DN_PERCENTILE)
                trend_str = "uptrend"
            else:
                forecasts_upper, forecasts_mid, forecasts_lower = memory_down.forecast(downtrend_init_price, FCAST, UP_PERCENTILE, MID_PERCENTILE, DN_PERCENTILE)
                trend_str = "downtrend"
            if forecasts_upper is not None:
                forecasts.append({
                    "trigger_index": i,
                    "trend": trend_str,
                    "init_price": uptrend_init_price if is_uptrend else downtrend_init_price,
                    "forecast_upper": forecasts_upper,
                    "forecast_mid": forecasts_mid,
                    "forecast_lower": forecasts_lower
                })
    return df, forecasts

# -----------------------------
# Plotting functions
# -----------------------------
def plot_and_save_forecasting(df, forecasts, filename):
    """
    Create a plot with the price series and, for each trigger event, plot forecast lines.
    """
    plt.figure(figsize=(14,7))
    plt.plot(df.index, df["close"], label="Close Price", color="black")
    
    # For each forecast event, plot forecast lines starting from the trigger date
    for event in forecasts:
        idx = event["trigger_index"]
        # The x-axis (forecast horizon) as future dates:
        if idx < len(df.index):
            start_time = df.index[idx]
        else:
            continue
        # Create x-axis for forecast (here we use integer offsets)
        x_forecast = np.arange(idx, idx + len(event["forecast_mid"]))
        # Convert x_forecast to dates if possible
        forecast_dates = df.index[:len(x_forecast)]
        
        # Plot forecast lines
        plt.plot(x_forecast, event["forecast_upper"], linestyle="--", color="#3179f5",
                 label="Forecast Upper" if event==forecasts[0] else "")
        plt.plot(x_forecast, event["forecast_mid"], linestyle="--", color="gray",
                 label="Forecast Mid" if event==forecasts[0] else "")
        plt.plot(x_forecast, event["forecast_lower"], linestyle="--", color="#ff5d00",
                 label="Forecast Lower" if event==forecasts[0] else "")
        # Mark the trigger event
        plt.axvline(x=idx, color="purple", linestyle=":", lw=1,
                    label="Trigger" if event==forecasts[0] else "")

    plt.xlabel("Bar Index")
    plt.ylabel("Price")
    plt.title("MACD Based Price Forecasting")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Forecast plot saved as {filename}")

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # For demonstration, create synthetic price data (random walk)
    # np.random.seed(42)
    # n_bars = 300
    # dates = pd.date_range(start="2023-01-01", periods=n_bars, freq="D")
    # # create a random walk price series starting at 100
    # price = 100 + np.cumsum(np.random.randn(n_bars))
    # df_sample = pd.DataFrame({"close": price}, index=dates)
    for ticker in TICKERS:
        prices = pd.read_csv(f"{MARKET_DATA_DIR}/{ticker}_data.csv")['Close']
        dates = prices.index
        close = prices
        df_sample = pd.DataFrame({"close": close}, index=dates)
        
        # Run forecasting routine
        df_macd, forecast_events = forecast_macd_prices(df_sample)
        
        if not os.path.exists(f"{FIGURES_DIR}/macd_ml"):
            os.makedirs(f"{FIGURES_DIR}/macd_ml")
        
            # Print forecast events summary
        print("Forecast events detected:")
        for event in forecast_events:
            print(f"Trigger at bar index {event['trigger_index']}, trend: {event['trend']}, "
                f"Reference Price: {event['init_price']:.2f}")
    
        # Plot the results
        plot_and_save_forecasting(df_sample, forecast_events, filename=f"{FIGURES_DIR}/macd_ml/{ticker}_macd_ml_forecast.png")
