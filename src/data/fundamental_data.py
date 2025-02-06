import yfinance as yf
import pandas as pd
import logging
from typing import Optional, Dict
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import FUNDAMENTAL_DATA_DIR, TICKERS

class FundamentalDataLoader:
    """
    A class to fetch and process fundamental data for a given ticker symbol using yfinance.
    
    Best practices implemented:
      • Modularity: Encapsulated functionality via class methods.
      • Readability: Clear docstrings, type annotations, and inline comments.
      • Error handling and logging: Logging useful status messages and error conditions.
      • Configuration: Accepts the ticker symbol during initialization.
    """
    def __init__(self, ticker: str, fundamental_data_dir: Optional[Path] = None):
        """
        Initialize the FundamentalDataLoader.

        Args:
            ticker (str): The ticker symbol for the company (e.g., 'AAPL', 'MSFT').
        """
        self.ticker: str = ticker if ticker else TICKERS
        self.fundamental_data_dir: Path = fundamental_data_dir if fundamental_data_dir else FUNDAMENTAL_DATA_DIR
        self.info: Optional[Dict] = None

        # Setup a logger for this module with INFO level output
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_fundamentals(self) -> Dict:
        """
        Fetch the fundamental data for the ticker using yfinance.

        Returns:
            dict: A dictionary containing fundamental information.
        """
        self.logger.info(f"Fetching fundamental data for {self.ticker}")
        try:
            ticker_obj = yf.Ticker(self.ticker)
            self.info = ticker_obj.info
            if not self.info:
                self.logger.warning(f"No fundamental information returned for {self.ticker}.")
            else:
                self.logger.info(f"Successfully fetched fundamental data for {self.ticker}.")
        except Exception as e:
            self.logger.error(f"Error fetching fundamental data for {self.ticker}: {e}")
            raise
        return self.info

    def get_key_metrics(self) -> pd.Series:
        """
        Extract key financial metrics from the fetched fundamental data.

        Returns:
            pd.Series: A Series containing selected metrics (marketCap, trailingPE, 
                       forwardPE, dividendYield, beta).
        """
        if not self.info:
            self.logger.error("Fundamental data not available. Run fetch_fundamentals() first.")
            return pd.Series()
        
        keys = ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 'beta']
        metrics = {key: self.info.get(key, None) for key in keys}
        self.logger.info("Extracted key metrics.")
        return pd.Series(metrics)

    def save_to_json(self, filename: str) -> None:
        """
        Persist the fundamental data to a JSON file.

        Args:
            filename (str): The file path where to save the fundamental data.
        """
        if not self.info:
            self.logger.error("No data to save. Execute fetch_fundamentals() first.")
            return
        
        try:
            # Convert the fundamental info dictionary to a Series and export to JSON.
            pd.Series(self.info).to_json(filename)
            self.logger.info(f"Fundamentals saved to {filename}.")
        except Exception as e:
            self.logger.error(f"Error saving data to {filename}: {e}")
            raise


if __name__ == "__main__":
    # Example usage for ticker 'AAPL'
    # We have added a list of tickers in the config file with all the tickers we want to fetch data for
    # We can now loop through the list of tickers and fetch the data for each ticker
    # We ensure its saved in the cache directory
    for ticker in TICKERS:
        loader = FundamentalDataLoader(ticker=ticker)
        try:
            fundamentals = loader.fetch_fundamentals()
            fundamentals_series = pd.Series(fundamentals)
            print("Fundamental Data Sample:")
            print(fundamentals_series.head())
            
            key_metrics = loader.get_key_metrics()
            print("\nKey Metrics:")
            print(key_metrics)
            
            loader.save_to_json(f"{loader.fundamental_data_dir}/{loader.ticker}_fundamentals.json")
        except Exception as ex:
            print(f"An error occurred: {ex}")
