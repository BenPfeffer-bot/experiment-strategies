"""
This module implements a Fibonacci Trailing Stop indicator inspired by LuxAlgo.
It calculates swing pivots over a given window (L, R), obtains Fibonacci-adjusted
levels based on the last two pivots, and then computes a trailing stop (st) which
switches direction according to the trigger price (either close or wick).
  
License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR

# --------------------------
# Helper functions for pivots
# --------------------------
def is_pivot_high(idx, L, R, highs):
    """
    Returns True if the bar at idx is a pivot high:
    It must be the maximum over [idx - L, idx + R].
    """
    if idx < L or idx > len(highs) - R - 1:
        return False
    window = highs[idx - L: idx + R + 1]
    return highs[idx] == max(window)

def is_pivot_low(idx, L, R, lows):
    """
    Returns True if the bar at idx is a pivot low:
    It must be the minimum over [idx - L, idx + R].
    """
    if idx < L or idx > len(lows) - R - 1:
        return False
    window = lows[idx - L: idx + R + 1]
    return lows[idx] == min(window)

# --------------------------
# Data structures for pivots
# --------------------------
class Pivot:
    def __init__(self, bar_index, price, direction):
        """
        direction: 1 for high pivot, -1 for low pivot.
        """
        self.bar = bar_index
        self.price = price
        self.direction = direction

# --------------------------
# Main Indicator Class
# --------------------------
class FibonacciTrailingStop:
    def __init__(self,
                 L=20,
                 R=1,
                 use_labels=False,   # labelling not plotted in this Python version
                 F=-0.382,           # Default Fibonacci level when use_f is False
                 use_f=False,        # if True use f parameter, else use F.
                 f=1.618,            # alternative Fibonacci multiplier when use_f is True
                 trigger='close'     # 'close' or 'wick' (if 'wick', use high in uptrend, low in downtrend).
                 ):
        """
        Parameters:
          L : int
             Left window for pivot detection.
          R : int
             Right window for pivot detection.
          use_labels : bool
             Toggle for creating swing labels (not plotted in this version).
          F : float
             Fibonacci adjustment factor (default when use_f is False).
          use_f : bool
             Toggle to use f instead of F.
          f : float
             Alternative Fibonacci multiplier.
          trigger : str
             Which price to use for trailing stop decisions ('close' or 'wick').
        """
        self.L = L
        self.R = R
        # For secondary (fill) pivots, use half windows as in Pine code.
        self.Ls = max(1, math.ceil(L / 2))
        self.Rs = max(1, math.ceil(R / 2))
        
        self.use_labels = use_labels
        self.iF = f if use_f else F
        self.trigger = trigger.lower()
        
        # Internal storage of pivots (most recent first)
        self.pivots = []
    
    def compute(self, df):
        """
        Compute Fibonacci Trailing Stop and related levels.
        
        Parameters:
          df: pd.DataFrame with at least columns: 'high', 'low', 'close'
          
        Returns:
          A DataFrame with columns:
            st: Trailing Stop line.
            other_side: Secondary level (from Fibonacci fills).
            dir: Direction indicator (1 for uptrend, -1 for downtrend, 0 for initial).
        """
        n_bars = len(df)
        st_series = np.full(n_bars, np.nan)
        other_side_series = np.full(n_bars, np.nan)
        dir_series = np.zeros(n_bars, dtype=int)
        
        # Initialize variables:
        # Current direction (0 = undefined, 1 = uptrend, -1 = downtrend)
        direction = 0  
        # Initialize trailing stop with the first bar's close value.
        st = df['close'].iloc[0]
        # Initialize swing maximum and minimum with first bar's high/low.
        swing_max = df['high'].iloc[0]
        swing_min = df['low'].iloc[0]
        other_side = st  # initial other side
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # For secondary pivot detection using half-window, precompute arrays:
        ph_small = np.full(n_bars, np.nan)
        pl_small = np.full(n_bars, np.nan)
        for i in range(self.Ls, n_bars - self.Rs):
            if is_pivot_high(i, self.Ls, self.Rs, highs):
                ph_small[i] = highs[i]
            if is_pivot_low(i, self.Ls, self.Rs, lows):
                pl_small[i] = lows[i]
        
        # Loop over bars
        for t in range(n_bars):
            # Check for confirmed pivots using full window (L, R)
            if is_pivot_high(t, self.L, self.R, highs):
                # For a high pivot, the confirmed pivot bar is t - R.
                pivot_bar = t - self.R
                pivot_price = highs[pivot_bar]
                if self.pivots:
                    first = self.pivots[0]
                    # If the last pivot is a high and the new pivot is higher,
                    # update the stored pivot (as in Pine).
                    if first.direction == 1 and pivot_price > first.price:
                        first.bar = pivot_bar
                        first.price = pivot_price
                    elif first.direction == -1 and pivot_price > first.price:
                        # Otherwise, add a new high pivot at the beginning.
                        self.pivots.insert(0, Pivot(pivot_bar, pivot_price, 1))
                    else:
                        # If same high-life, no update.
                        pass
                else:
                    self.pivots.insert(0, Pivot(pivot_bar, pivot_price, 1))
                    
            if is_pivot_low(t, self.L, self.R, lows):
                # For a low pivot, confirmed pivot bar is t - R.
                pivot_bar = t - self.R
                pivot_price = lows[pivot_bar]
                if self.pivots:
                    first = self.pivots[0]
                    if first.direction == -1 and pivot_price < first.price:
                        first.bar = pivot_bar
                        first.price = pivot_price
                    elif first.direction == 1 and pivot_price < first.price:
                        self.pivots.insert(0, Pivot(pivot_bar, pivot_price, -1))
                    else:
                        pass
                else:
                    self.pivots.insert(0, Pivot(pivot_bar, pivot_price, -1))
            
            # Only proceed if we have at least two pivots
            if len(self.pivots) >= 2:
                # Use the two most recent pivots
                p0 = self.pivots[0]
                p1 = self.pivots[1]
                current_max = max(p0.price, p1.price)
                current_min = min(p0.price, p1.price)
                # For the very first two pivots, set st and other_side as the average.
                if len(self.pivots) == 2:
                    st = (current_max + current_min) / 2.0
                    other_side = st
                diff = current_max - current_min
                # Adjust the extrema by the Fibonacci factor.
                current_max_adj = current_max + diff * self.iF
                current_min_adj = current_min - diff * self.iF
                
                # Determine swing direction from the pivot order:
                # If the most recent pivot is lower than the previous, d becomes negative (down swing), else positive.
                d = -diff if p0.price < p1.price else diff
                
                # Determine which price to use as the trigger.
                if self.trigger == 'close':
                    price = closes[t]
                else:
                    # 'wick' mode: use high in uptrend, low in downtrend,
                    # if direction is undefined (0) then use close.
                    if direction < 1:
                        price = highs[t]
                    elif direction > -1:
                        price = lows[t]
                    else:
                        price = closes[t]
                
                # Update trailing stop based on current direction and price action.
                if direction < 1:
                    if price > st:
                        st = current_min_adj
                        direction = 1
                    else:
                        st = min(st, current_max_adj)
                if direction > -1:
                    if price < st:
                        st = current_max_adj
                        direction = -1
                    else:
                        st = max(st, current_min_adj)
                
                # Update "other_side" using secondary pivot values if available from ph_small/pl_small.
                ph_val = ph_small[t] if not np.isnan(ph_small[t]) else None
                pl_val = pl_small[t] if not np.isnan(pl_small[t]) else None
                if direction < 1:
                    if ph_val is not None:
                        other_side = min(other_side, ph_val, st)
                    else:
                        other_side = min(other_side, st)
                elif direction > -1:
                    if pl_val is not None:
                        other_side = max(other_side, pl_val, st)
                    else:
                        other_side = max(other_side, st)
            
            # Store computed values for this bar.
            st_series[t] = st
            other_side_series[t] = other_side
            dir_series[t] = direction
        
        # Build a results DataFrame (indexed same as df)
        results = pd.DataFrame({
            'st': st_series,
            'other_side': other_side_series,
            'dir': dir_series
        }, index=df.index)
        return results

    def plot_and_save(self, df, computed, filename='fibonacci_trailing_stop.png'):
        """
        Plot the price series (close) along with the trailing stop and other side.
        
        Parameters:
          df: pd.DataFrame containing OHLC data.
          computed: DataFrame returned from compute().
          filename: str, output filename for the image.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
        plt.plot(computed.index, computed['st'], label='Fib Trailing Stop', color='teal', linewidth=2)
        plt.plot(computed.index, computed['other_side'], label='Other Side', color='orange', linestyle='--')
        plt.title('Fibonacci Trailing Stop Indicator')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as {filename}")

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Generate sample OHLC data using a random walk for illustration.
    # np.random.seed(42)
    # dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    # # Create a simple random walk for close prices.
    for ticker in TICKERS:
        prices = pd.read_csv(f"{MARKET_DATA_DIR}/{ticker}_data.csv")['Close']
        dates = prices.index
        close = prices
        # Create high and low as offsets from close.
        high = close + (np.random.rand(len(dates)) * 2)
        low = close - (np.random.rand(len(dates)) * 2)
        df_sample = pd.DataFrame({'close': close, 'high': high, 'low': low}, index=dates)
        
        # Instantiate the Fibonacci Trailing Stop indicator
        fib_stop = FibonacciTrailingStop(L=20, R=1, use_labels=False, F=-0.382, use_f=False, trigger='close')
        
        # Compute the indicator
        results = fib_stop.compute(df_sample)
        
        if not os.path.exists(f"{FIGURES_DIR}/fibonacci"):
            os.makedirs(f"{FIGURES_DIR}/fibonacci")
            
        # Plot and save the result
        fib_stop.plot_and_save(df_sample, results, filename=f"{FIGURES_DIR}/fibonacci/{ticker}_fibonacci_trailing_stop.png")
            
    # Print sample output for the last 5 bars.
    # print("Date       |  ST       | Other Side | Direction")
    # for d, row in results.tail(5).iterrows():
    #     print(f"{d.date()} | {row['st']:8.2f} | {row['other_side']:10.2f} | {int(row['dir'])}")
