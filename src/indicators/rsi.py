"""
This module implements a Multi-Length RSI indicator inspired by LuxAlgo.
It computes an average RSI over a range of lookback lengths (from min_length to max_length)
while also tracking the frequency (percentage) of readings above an overbought level 
or below an oversold level. It then smoothes these readings to produce upper and lower channels.

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR

class RSIMultiLength:
    def __init__(self, max_length=20, min_length=10, overbought=70, oversold=30):
        """
        Parameters:
        • max_length: int
            The maximum lookback length.
        • min_length: int
            The minimum lookback length.
        • overbought: float
            The RSI value above which the indicator considers the input overbought.
        • oversold: float
            The RSI value below which the indicator considers the input oversold.
        """
        self.max_length = max_length
        self.min_length = min_length
        self.overbought = overbought
        self.oversold = oversold
        
        # Total number of RSI lengths used.
        self.N = self.max_length - self.min_length + 1

    def compute(self, src: pd.Series):
        """
        Process a price (or source) series and compute the multi-length RSI average and channels.
        
        For each bar, the code maintains (per RSI length):
          - A running "RMA" (recursive moving average) of the price changes (num) 
            and the absolute changes (den) with a smoothing factor alpha = 1/period.
          - It then computes an RSI for that period as: RSI = 50 * (num/den) + 50.
        
        Over all periods from min_length to max_length, we compute:
          - avg_rsi: the average RSI value.
          - overbuy: number of periods with RSI > overbought.
          - oversell: number of periods with RSI < oversold.
          
        Then, an exponential smoothing is used to derive channels:
          - buy_rsi_ma: smoothed upper channel (moves toward avg_rsi if many RSIs are overbought).
          - sell_rsi_ma: smoothed lower channel (moves toward avg_rsi if many RSIs are oversold).
        
        Returns:
        A dictionary with keys:
          • avg_rsi: pd.Series of the average RSI.
          • buy_rsi: pd.Series of the upper channel.
          • sell_rsi: pd.Series of the lower channel.
          • overbought_pct: pd.Series representing (overbuy_count / N)*100.
          • oversold_pct: pd.Series representing (oversell_count / N)*100.
        """
        n_bars = len(src)
        # Initialize arrays to keep the running RMA values for each RSI period.
        num = np.zeros(self.N)
        den = np.zeros(self.N)
        
        # Output lists
        avg_rsi_list = []
        buy_rsi_list = []
        sell_rsi_list = []
        overbought_pct_list = []
        oversold_pct_list = []
        
        # For channel smoothing (EMA-like update)
        prev_buy_rsi = None
        prev_sell_rsi = None

        # Iterate over each time step
        for t in range(n_bars):
            # Compute price difference (diff). For the first bar, assume diff = 0.
            diff = 0.0 if t == 0 else src.iloc[t] - src.iloc[t-1]
            
            overbuy = 0
            oversell = 0
            sum_rsi = 0.0
            
            # Loop over RSI lengths (using index k and period = min_length + k)
            for k in range(self.N):
                period = self.min_length + k
                alpha = 1.0 / period
                # Update recursive moving averages
                num[k] = alpha * diff + (1 - alpha) * num[k]
                den[k] = alpha * abs(diff) + (1 - alpha) * den[k]
                
                # Compute RSI value for this lookback; avoid dividing by zero
                if den[k] != 0:
                    rsi_val = 50 * (num[k] / den[k]) + 50
                else:
                    rsi_val = 50.0
                
                sum_rsi += rsi_val
                
                if rsi_val > self.overbought:
                    overbuy += 1
                if rsi_val < self.oversold:
                    oversell += 1
            
            # Average RSI over all lengths.
            avg_rsi = sum_rsi / self.N
            
            # Smooth the channels:
            if prev_buy_rsi is None:
                buy_rsi = avg_rsi
                sell_rsi = avg_rsi
            else:
                buy_rsi = prev_buy_rsi + (overbuy / self.N) * (avg_rsi - prev_buy_rsi)
                sell_rsi = prev_sell_rsi + (oversell / self.N) * (avg_rsi - prev_sell_rsi)
            
            # Save percentages (as in the original: count/N * 100)
            ob_pct = (overbuy / self.N) * 100
            os_pct = (oversell / self.N) * 100
            
            avg_rsi_list.append(avg_rsi)
            buy_rsi_list.append(buy_rsi)
            sell_rsi_list.append(sell_rsi)
            overbought_pct_list.append(ob_pct)
            oversold_pct_list.append(os_pct)
            
            # Update previous channel values for next iteration
            prev_buy_rsi = buy_rsi
            prev_sell_rsi = sell_rsi
        
        # Build Series for output using the same index as input
        avg_rsi_series = pd.Series(avg_rsi_list, index=src.index, name='Average Multi-Length RSI')
        buy_rsi_series = pd.Series(buy_rsi_list, index=src.index, name='Upper Channel')
        sell_rsi_series = pd.Series(sell_rsi_list, index=src.index, name='Lower Channel')
        overbought_pct_series = pd.Series(overbought_pct_list, index=src.index, name='Overbought %')
        oversold_pct_series = pd.Series(oversold_pct_list, index=src.index, name='Oversold %')
        
        return {
            'avg_rsi': avg_rsi_series,
            'buy_rsi': buy_rsi_series,
            'sell_rsi': sell_rsi_series,
            'overbought_pct': overbought_pct_series,
            'oversold_pct': oversold_pct_series
        }
    
    def plot_and_save(self, src: pd.Series, computed: dict, filename='rsi_multilength.png'):
        """
        Plot the multi-length RSI along with upper and lower channels and overbought/oversold percentages.
        
        Parameters:
        • src: pd.Series
            The price (or source) series.
        • computed: dict
            Dictionary containing computed Series (avg_rsi, buy_rsi, sell_rsi, overbought_pct, oversold_pct).
        • filename: str
            The name of the image file to save the plot.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot the RSI channels and average
        plt.plot(src.index, computed['avg_rsi'], label='Average RSI', color='purple', linewidth=1.5)
        plt.plot(src.index, computed['buy_rsi'], label='Upper Channel', color='green', linestyle='--')
        plt.plot(src.index, computed['sell_rsi'], label='Lower Channel', color='red', linestyle='--')
        
        # Fill between the channels (using a light transparent fill)
        plt.fill_between(src.index, computed['buy_rsi'], computed['sell_rsi'],
                         color='lavender', alpha=0.5)
        
        # Plot horizontal line at 50 (reference level)
        plt.axhline(50, color='gray', linestyle=':', linewidth=1)
        
        # Plot the overbought/oversold percentage areas as subplots above the main RSI (if desired)
        # Here, we simply annotate the last available percentages on the main plot.
        last_date = src.index[-1]
        ob_pct = computed['overbought_pct'].iloc[-1]
        os_pct = computed['oversold_pct'].iloc[-1]
        plt.text(last_date, 90, f'Overbought: {ob_pct:.2f} %', color='green', fontsize=10, va='center')
        plt.text(last_date, 10, f'Oversold: {os_pct:.2f} %', color='red', fontsize=10, va='center')
        
        plt.title('RSI Multi-Length [LuxAlgo]')
        plt.xlabel('Date')
        plt.ylabel('RSI Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot to file
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as {filename}")


# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":
    # Create a sample price series with a random walk
    # np.random.seed(42)
    # dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    # prices = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)
    for ticker in TICKERS:
        prices = pd.read_csv(f"{MARKET_DATA_DIR}/{ticker}_data.csv")['Close']
            
        # Instantiate RSIMultiLength indicator with default parameters
        rsi_indicator = RSIMultiLength(max_length=20, min_length=10, overbought=70, oversold=30)
        
        # Compute the RSI and channels
        computed_values = rsi_indicator.compute(prices)
        
        if not os.path.exists(f"{FIGURES_DIR}/rsi"):
            os.makedirs(f"{FIGURES_DIR}/rsi")
        
        # Plot and save the results
        rsi_indicator.plot_and_save(prices, computed_values, filename=f"{FIGURES_DIR}/rsi/{ticker}_rsi_multilength.png")
        
        # Print some sample output for the last 5 data points
        print("Date       | Average RSI | Upper Channel | Lower Channel | Overbought % | Oversold %")
    # for date in prices.index[-5:]:
    #     print(f"{date.date()} | {computed_values['avg_rsi'].loc[date]:8.2f}   | {computed_values['buy_rsi'].loc[date]:8.2f}   | "
    #         f"{computed_values['sell_rsi'].loc[date]:8.2f}   | {computed_values['overbought_pct'].loc[date]:6.2f} %   | {computed_values['oversold_pct'].loc[date]:6.2f} %")
