"""
This module implements a Moving Average indicator with forecasting capabilities.
Inspired by LuxAlgos MACD-Based Price Forecasting indicator, this code uses a simple
moving average for trend determination and stores recent deviations from an initial 
price to forecast upper, mid, and lower price levels using percentile-based interpolation.

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import FIGURES_DIR, TICKERS, MARKET_DATA_DIR

class MovingAverageForecast:
    def __init__(self, window=20, max_memory=50, forecast_length=100,
                 up_per=80, mid_per=50, dn_per=20):
        """
        Initialize the indicator.

        Parameters:
        • window : int
            Length of the moving average window.
        • max_memory : int
            Maximum number of deviation samples to keep for each trend.
        • forecast_length : int
            The horizon over which to forecast (this parameter is available for future expansion).
        • up_per : float
            Upper forecast percentile.
        • mid_per : float
            Mid (median/average) forecast percentile.
        • dn_per : float
            Lower forecast percentile.
        """
        self.window = window
        
        self.max_memory = max_memory
        self.forecast_length = forecast_length
        self.up_per = up_per
        self.mid_per = mid_per
        self.dn_per = dn_per

        # Memory to store deltas for uptrend and downtrend respectively.
        self.memory = {'up': [], 'down': []}
        # Initial reference prices for each trend; these update at trend-change points.
        self.uptrend_init_price = None
        self.downtrend_init_price = None

    def _update_memory(self, trend, current_price, init_price):
        """
        Update the memory for a given trend.

        Parameters:
        • trend : str
            Either 'up' or 'down'.
        • current_price : float
            Current price value.
        • init_price : float
            The reference price at the time the current trend started.

        Returns:
        The updated memory list for the trend.
        """
        delta = current_price - init_price
        self.memory[trend].append(delta)
        # Maintain the memory length
        if len(self.memory[trend]) > self.max_memory:
            self.memory[trend].pop(0)
        return self.memory[trend]

    def _compute_forecast_levels(self, init_price, mem_deltas):
        """
        Calculate forecast levels based on the current deviation memory.

        Parameters:
        • init_price : float
            The trend's initial reference price.
        • mem_deltas : list of float
            The stored deviations (delta values).

        Returns:
        A tuple (upper, mid, lower) representing forecasted price levels.
        """
        # Using numpy's percentile function for linear interpolation.
        upper = init_price + np.percentile(mem_deltas, self.up_per)
        mid = init_price + np.percentile(mem_deltas, self.mid_per)
        lower = init_price + np.percentile(mem_deltas, self.dn_per)
        return upper, mid, lower

    def compute(self, close_series: pd.Series):
        """
        Process the close price series, compute the moving average,
        and generate forecasted levels based on trend.

        Parameters:
        • close_series : pd.Series
            Price series (indexed by time).

        Returns:
        • ma_series : pd.Series
            The computed moving average.
        • forecasts : list of dict
            For each available index, returns a dictionary with:
            {
                'price': current price,
                'trend': 'up' or 'down',
                'forecast': { 'upper': float, 'mid': float, 'lower': float } or None
            }

        Note: Forecast values are computed only if enough data is available.
        """
        ma_series = close_series.rolling(window=self.window, min_periods=self.window).mean()
        forecasts = []

        # Iterate over the series by index
        for idx in range(len(close_series)):
            price = close_series.iloc[idx]
            current_ma = ma_series.iloc[idx]
            forecast_levels = None

            if np.isnan(current_ma):
                # Not enough data to compute the moving average yet
                forecasts.append({
                    'price': price,
                    'trend': None,
                    'forecast': None
                })
                continue

            # Determine trend: uptrend if price > MA, else downtrend
            if price > current_ma:
                trend = 'up'
                # Set reference if the trend just started
                if self.uptrend_init_price is None or \
                   (idx > 0 and close_series.iloc[idx - 1] <= ma_series.iloc[idx - 1]):
                    self.uptrend_init_price = price
                    # Clear previous memory for a fresh trend
                    self.memory['up'] = []
                mem = self._update_memory('up', price, self.uptrend_init_price)
                if len(mem) >= 1:  # Only forecast if there is memory available
                    upper, mid, lower = self._compute_forecast_levels(self.uptrend_init_price, mem)
                    forecast_levels = {'upper': upper, 'mid': mid, 'lower': lower}
                else:
                    forecast_levels = {'upper': None, 'mid': None, 'lower': None}
            else:
                trend = 'down'
                if self.downtrend_init_price is None or \
                   (idx > 0 and close_series.iloc[idx - 1] >= ma_series.iloc[idx - 1]):
                    self.downtrend_init_price = price
                    self.memory['down'] = []
                mem = self._update_memory('down', price, self.downtrend_init_price)
                if len(mem) >= 1:
                    upper, mid, lower = self._compute_forecast_levels(self.downtrend_init_price, mem)
                    forecast_levels = {'upper': upper, 'mid': mid, 'lower': lower}
                else:
                    forecast_levels = {'upper': None, 'mid': None, 'lower': None}

            forecasts.append({
                'price': price,
                'trend': trend,
                'forecast': forecast_levels
            })

        return ma_series, forecasts

    def plot_and_save(self, close_series: pd.Series, ma_series: pd.Series, forecasts: list, filename='moving_average_forecast.png'):
        """
        Plot the price series, moving average, and forecast levels, then save the plot to a file.

        Parameters:
        • close_series : pd.Series
            Original price series.
        • ma_series : pd.Series
            Moving average computed from the price series.
        • forecasts : list of dict
            Forecast results generated from the compute method.
        • filename : str
            The name of the file to save the plot.
        """
        # Prepare arrays for forecast levels
        upper_forecast = []
        mid_forecast = []
        lower_forecast = []

        # Iterate through forecasts to extract forecast levels
        for forecast in forecasts:
            if forecast['forecast'] is None or forecast['forecast']['upper'] is None:
                upper_forecast.append(np.nan)
                mid_forecast.append(np.nan)
                lower_forecast.append(np.nan)
            else:
                upper_forecast.append(forecast['forecast']['upper'])
                mid_forecast.append(forecast['forecast']['mid'])
                lower_forecast.append(forecast['forecast']['lower'])

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(close_series.index, close_series.values, label='Price', color='black', linewidth=1.5)
        plt.plot(ma_series.index, ma_series.values, label=f'MA (window={self.window})', color='blue', linestyle='--')
        plt.plot(close_series.index, upper_forecast, label='Forecast Upper', color='green', linestyle=':')
        plt.plot(close_series.index, mid_forecast, label='Forecast Mid', color='orange', linestyle='-.')
        plt.plot(close_series.index, lower_forecast, label='Forecast Lower', color='red', linestyle=':')

        plt.title('Moving Average Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as {filename}")


# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":
    # For demonstration, generate a random price series using a simple random walk.
    try:
        for ticker in TICKERS:
            prices = pd.read_csv(f"{MARKET_DATA_DIR}/{ticker}_data.csv")['Close']
            
            # dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
            # prices = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)

            # Instantiate the Moving Average Forecast indicator.
            ma_forecaster = MovingAverageForecast(window=20, max_memory=50,
                                                forecast_length=100,
                                                up_per=80, mid_per=50, dn_per=20)

            ma_series, forecasts = ma_forecaster.compute(prices)

            if not os.path.exists(f"{FIGURES_DIR}/moving_avg"):
                os.makedirs(f"{FIGURES_DIR}/moving_avg")
            
            # Plot and save the result to a file.
            ma_forecaster.plot_and_save(prices, ma_series, forecasts, filename=f"{FIGURES_DIR}/moving_avg/{ticker}_moving_average_forecast.png")
    except Exception as e:
        print(f"Error: {e}")

    # Print sample output (last 5 data points)
    # for ts, forecast in list(zip(prices.index, forecasts))[-5:]:
    #     print(f"{ts.date()} | Price: {forecast['price']:.2f} | "
    #           f"Trend: {forecast['trend']} | Forecast: {forecast['forecast']}")
