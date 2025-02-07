"""
This module implements the Fibonacci Bollinger Bands (FBB) indicator.
Inspired by LuxAlgo's Pine Script version, it calculates the following:
    
  - basis: A VWMA (if volume data is available) or simple moving average of the source.
  - dev:  The standard deviation of the source over the specified length multiplied by a user-defined multiplier.
  - Upper bands: basis + (factor * dev) for factors 0.236, 0.382, 0.5, 0.618, 0.764, and 1.
  - Lower bands: basis - (factor * dev) for the same factors.
  
If volume data is present in the DataFrame, a VWMA is used; otherwise a simple moving average is computed.
  
License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import math
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR  

def compute_vwma(series: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """
    Computes the Volume Weighted Moving Average (VWMA).

    Parameters:
      series : pd.Series
         Price data (or any source series).
      volume : pd.Series
         Volume data.
      length : int
         Window length.

    Returns:
      pd.Series containing the VWMA.
    """
    # Compute rolling sum of price*volume and volume
    pv = series * volume
    vwma = pv.rolling(window=length, min_periods=length).sum() / volume.rolling(window=length, min_periods=length).sum()
    return vwma

def compute_fibo_bollinger_bands(df: pd.DataFrame, 
                                 length: int=200, 
                                 src: str=None, 
                                 mult: float=3.0) -> pd.DataFrame:
    """
    Computes Fibonacci Bollinger Bands (FBB) on the given OHLC (and optionally volume) DataFrame.

    Parameters:
      df : pd.DataFrame
         Must contain at least 'high', 'low', 'close'. Optionally 'volume'.
      length : int, default=200
         Rolling window length.
      src : str, optional
         The column name or a derived series to be used as the source.
         If not provided, hlc3 is used ((high+low+close)/3).
      mult : float, default=3.0
         Multiplier for the standard deviation.

    Returns:
      bands : pd.DataFrame
         A DataFrame with columns:
           - basis
           - dev
           - upper_1, upper_2, upper_3, upper_4, upper_5, upper_6
           - lower_1, lower_2, lower_3, lower_4, lower_5, lower_6
    """
    # Use hlc3 by default if src not provided
    if src is None:
        hlc3 = (df['high'] + df['low'] + df['close']) / 3.0
    else:
        # If src is a column name, use it; if not assume it's a series.
        hlc3 = df[src] if isinstance(src, str) and src in df.columns else src

    # Compute basis: use VWMA if volume exists, otherwise a simple moving average.
    if 'volume' in df.columns:
        basis = compute_vwma(hlc3, df['volume'], length)
    else:
        basis = hlc3.rolling(window=length, min_periods=length).mean()

    # Compute rolling standard deviation
    dev = hlc3.rolling(window=length, min_periods=length).std() * mult
    
    # Compute Fibonacci factors for band scaling.
    factors = [0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
    bands = pd.DataFrame(index=df.index)
    bands['basis'] = basis
    bands['dev'] = dev
    # Upper bands
    bands['upper_1'] = basis + factors[0] * dev
    bands['upper_2'] = basis + factors[1] * dev
    bands['upper_3'] = basis + factors[2] * dev
    bands['upper_4'] = basis + factors[3] * dev
    bands['upper_5'] = basis + factors[4] * dev
    bands['upper_6'] = basis + factors[5] * dev
    # Lower bands
    bands['lower_1'] = basis - factors[0] * dev
    bands['lower_2'] = basis - factors[1] * dev
    bands['lower_3'] = basis - factors[2] * dev
    bands['lower_4'] = basis - factors[3] * dev
    bands['lower_5'] = basis - factors[4] * dev
    bands['lower_6'] = basis - factors[5] * dev

    return bands

def plot_fibo_bollinger_bands(df: pd.DataFrame, bands: pd.DataFrame, filename: str='fibo_bollingbands.png'):
    """
    Plot the Fibonacci Bollinger Bands along with the price series.

    Parameters:
      df : pd.DataFrame
         DataFrame that contains at least a 'close' column.
      bands : pd.DataFrame
         DataFrame returned from compute_fibo_bollinger_bands(), containing the bands.
      filename : str, default='fibo_bollingbands.png'
         File name to save the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')
    # Plot the price series (using close price)
    plt.plot(df.index, df['close'], label='Close Price', color='gray', linewidth=1.5)
    
    # Plot basis using fuchsia color
    plt.plot(bands.index, bands['basis'], label='Basis (VWMA/SMA)', color='fuchsia', linewidth=2)
    
    # Upper bands: first 5 as white, 6th as red (thicker)
    plt.plot(bands.index, bands['upper_1'], label='Upper 0.236', color='white', linewidth=1)
    plt.plot(bands.index, bands['upper_2'], label='Upper 0.382', color='white', linewidth=1)
    plt.plot(bands.index, bands['upper_3'], label='Upper 0.5',   color='white', linewidth=1)
    plt.plot(bands.index, bands['upper_4'], label='Upper 0.618', color='white', linewidth=1)
    plt.plot(bands.index, bands['upper_5'], label='Upper 0.764', color='white', linewidth=1)
    plt.plot(bands.index, bands['upper_6'], label='Upper 1.0',   color='red',   linewidth=2)
    
    # Lower bands: first 5 as white, 6th as green (thicker)
    plt.plot(bands.index, bands['lower_1'], label='Lower 0.236', color='white', linewidth=1)
    plt.plot(bands.index, bands['lower_2'], label='Lower 0.382', color='white', linewidth=1)
    plt.plot(bands.index, bands['lower_3'], label='Lower 0.5',   color='white', linewidth=1)
    plt.plot(bands.index, bands['lower_4'], label='Lower 0.618', color='white', linewidth=1)
    plt.plot(bands.index, bands['lower_5'], label='Lower 0.764', color='white', linewidth=1)
    plt.plot(bands.index, bands['lower_6'], label='Lower 1.0',   color='green', linewidth=2)
    
    plt.title('Fibonacci Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best', fontsize='small', ncol=2, framealpha=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Fibonacci Bollinger Bands plot saved as {filename}")

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Generate example data:
    # np.random.seed(42)
    # dates = pd.date_range(start="2023-01-01@", periods=300, freq="D")
    # Create a random walk for close prices
    for ticker in TICKERS:
        prices = pd.read_csv(f"{MARKET_DATA_DIR}/{ticker}_data.csv")['Close']
        dates = prices.index
        close = prices
        
        # Create high and low as offsets from close.
        high = close + (np.random.rand(len(dates)) * 2)
        low = close - (np.random.rand(len(dates)) * 2)
        volume = pd.read_csv(f"{MARKET_DATA_DIR}/{ticker}_data.csv")['Volume']
        
        # Build DataFrame
        df_sample = pd.DataFrame({
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        }, index=dates)
        
        # Compute Fibonacci Bollinger Bands
        bands = compute_fibo_bollinger_bands(df_sample, length=200, mult=3.0)

        # Ensure folder exists before generating plots
        if not os.path.exists(f"{FIGURES_DIR}/fibo_bollingbands"):
            os.makedirs(f"{FIGURES_DIR}/fibo_bollingbands")
        
        # Plot the indicator over the close prices.
        plot_fibo_bollinger_bands(df_sample, bands, filename=f"{FIGURES_DIR}/fibo_bollingbands/{ticker}_fibo_bollingbands.png")
            
        # Print a summary of a few last values
        print(bands.tail(5))
