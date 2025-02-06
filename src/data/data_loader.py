# functions to fetch and ingest market data (historical)

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import yfinance as yf

from src.utils.config import CACHE_DIR, OUTPUT_DIR, TICKERS, MARKET_DATA_DIR

class FetchData:
    """
    Fetch data from Yahoo Finance
    """

    def __init__(self,
                cache_dir: Optional[Path] = None,
                output_dir: Optional[Path] = None,
                market_data_dir: Optional[Path] = None,
                ticker: Optional[List[str]] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                period: str = "max",
                interval: str = "1d",
    ) -> None:
        """
        Initialize the DataLoader instance.

        Args:
            ticker (str): The ticker symbol (e.g., '^STOXX50E' for EURO STOXX 50).
            start_date (Optional[str], optional): Start date in 'YYYY-MM-DD' format.
                If provided, it overrides the period parameter. Defaults to None.
            end_date (Optional[str], optional): End date in 'YYYY-MM-DD' format.
                If not provided, it defaults to the current date. Defaults to None.
            period (str, optional): The period of data to fetch (e.g., '5y').
                Used only if start_date is None. Defaults to "max".
            interval (str, optional): The data interval (e.g., '1d', '1h'). Defaults to "1d".
        """
        self.cache_dir: Path = cache_dir if cache_dir else CACHE_DIR
        self.output_dir: Path = output_dir if output_dir else OUTPUT_DIR
        self.market_data_dir: Path = market_data_dir if market_data_dir else MARKET_DATA_DIR
        self.ticker: List[str] = ticker if ticker else TICKERS
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval
        self.data: Optional[pd.DataFrame] = None

        # Setup a basic logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_data(self) -> pd.DataFrame:
            """
            Fetch historical data from Yahoo Finance.

            Returns:
                pd.DataFrame: DataFrame containing historical price data.
            
            Raises:
                Exception: If data fetching fails.
            """
            self.logger.info(f"Fetching data for ticker: {self.ticker}")

            try:
                ticker_obj = yf.Ticker(self.ticker)

                # If a start date is provided, use history with the given range
                if self.start_date:
                    self.data = ticker_obj.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval=self.interval,
                    )
                else:
                    self.data = ticker_obj.history(period=self.period, interval=self.interval)

                if self.data.empty:
                    self.logger.warning("Fetched data is empty. Verify the ticker and date parameters.")
                else:
                    self.logger.info(f"Successfully fetched {len(self.data)} records for {self.ticker}.")

                # Perform basic cleaning: reset index to have a default integer index and date as a column
                self.data.reset_index(inplace=True)

                return self.data

            except Exception as e:
                self.logger.error(f"Error fetching data for {self.ticker}: {str(e)}")
                raise

    def save_to_csv(self, filename: str) -> None:
        """
        Save the fetched data to a CSV file.

        Args:
            filename (str): The filepath for saving the CSV.

        Raises:
            Exception: If saving fails.
        """
        if self.data is None:
            self.logger.error("No data available to save. Please run fetch_data() first.")
            return

        try:
            self.data.to_csv(filename, index=False)
            self.logger.info(f"Data successfully saved to {filename}.")
        except Exception as e:
            self.logger.error(f"Error saving data to {filename}: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage with EURO STOXX 50 data:
    # The ticker '^STOXX50E' is commonly used for EURO STOXX 50; adjust accordingly.
    # We have added a list of tickers in the config file with all the tickers we want to fetch data for
    # We can now loop through the list of tickers and fetch the data for each ticker
    # We ensure its saved in the cache directory
    for ticker in TICKERS:
        loader = FetchData(ticker=ticker, period="5y", interval="1d")   
        try:
            data = loader.fetch_data()
            print(data.head())
            loader.save_to_csv(f"{loader.market_data_dir}/{loader.ticker}_data.csv")
        except Exception as e:
            print(f"An error occurred: {e}")