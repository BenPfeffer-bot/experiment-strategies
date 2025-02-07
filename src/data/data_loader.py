# functions to fetch and ingest market data (historical)

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import sys
import os
import time

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import yfinance as yf

from src.utils.config import (
    CACHE_DIR,
    OUTPUT_DIR,
    TICKERS,
    MARKET_DATA_DIR,
    RAW_DATA_DIR,
)


class FetchData:
    """
    Fetch data from Yahoo Finance
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        market_data_dir: Optional[Path] = None,
        ticker: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "max",
        interval: str = "1d",
        max_retries: int = 3,
        retry_delay: int = 5,
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
            max_retries (int, optional): Maximum number of retries for fetching data. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
        """
        self.cache_dir: Path = cache_dir if cache_dir else CACHE_DIR
        self.output_dir: Path = output_dir if output_dir else OUTPUT_DIR
        self.market_data_dir: Path = (
            market_data_dir if market_data_dir else MARKET_DATA_DIR
        )
        self.ticker: List[str] = ticker if ticker else TICKERS
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval
        self.data: Optional[pd.DataFrame] = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._setup_rate_limiter()

        # Setup a basic logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_rate_limiter(self):
        """Setup rate limiting to avoid API throttling"""
        self.last_request_time = 0
        self.min_request_interval = 2  # seconds between requests

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched data quality
        """
        if data is None or data.empty:
            return False

        # Check for minimum required columns
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not all(col in data.columns for col in required_cols):
            return False

        # Check for excessive missing values
        missing_threshold = 0.1
        missing_ratio = data[list(required_cols)].isnull().mean()
        if any(missing_ratio > missing_threshold):
            return False

        # Check for data staleness
        if "Date" in data.columns:
            latest_date = pd.to_datetime(data["Date"]).max()
            staleness_days = (pd.Timestamp.now() - latest_date).days
            if staleness_days > 5:  # Data shouldn't be more than 5 days old
                return False

        return True

    def fetch_data(self) -> pd.DataFrame:
        """
        Enhanced fetch_data with retries and validation
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    f"Attempt {attempt + 1} of {self.max_retries} to fetch data for {self.ticker}"
                )

                ticker_obj = yf.Ticker(self.ticker)

                if self.start_date:
                    self.data = ticker_obj.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval=self.interval,
                    )
                else:
                    self.data = ticker_obj.history(
                        period=self.period, interval=self.interval
                    )

                if self._validate_data(self.data):
                    self.logger.info(
                        f"Successfully fetched and validated data for {self.ticker}"
                    )
                    self.data = self._clean_and_standardize(self.data)
                    return self.data
                else:
                    raise ValueError("Data validation failed")

            except Exception as e:
                self.logger.error(f"Error fetching data for {self.ticker}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def _clean_and_standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and clean the fetched data
        """
        # Reset index to have date as column
        data.reset_index(inplace=True)

        # Ensure datetime format
        data["Date"] = pd.to_datetime(data["Date"])

        # Handle missing values
        data = data.fillna(method="ffill").fillna(method="bfill")

        # Add derived columns
        data["Returns"] = data["Close"].pct_change()
        data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Volatility"] = data["Returns"].rolling(window=20).std() * np.sqrt(252)

        # Remove outliers
        for col in ["Open", "High", "Low", "Close"]:
            data = self._remove_outliers(data, col)

        return data

    def _remove_outliers(
        self, data: pd.DataFrame, column: str, n_std: float = 3
    ) -> pd.DataFrame:
        """
        Remove outliers using z-score method
        """
        mean = data[column].mean()
        std = data[column].std()
        data = data[
            (data[column] <= mean + (n_std * std))
            & (data[column] >= mean - (n_std * std))
        ]
        return data

    def save_to_csv(self, filename: str) -> None:
        """
        Save the fetched data to a CSV file.

        Args:
            filename (str): The filepath for saving the CSV.

        Raises:
            Exception: If saving fails.
        """
        if self.data is None:
            self.logger.error(
                "No data available to save. Please run fetch_data() first."
            )
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
            loader.save_to_csv(f"{RAW_DATA_DIR}/{loader.ticker}_data.csv")
        except Exception as e:
            print(f"An error occurred: {e}")
