import pandas as pd
import numpy as np
import logging
import io
from typing import Dict, List, Optional
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import OUTPUT_DIR, MARKET_DATA_DIR, FUNDAMENTAL_DATA_DIR, NEWS_DATA_DIR, TICKERS, OUTPUT_PREPROCESSED_DATA_DIR

class DataIntegrator:
    """
    A class that processes and integrates multiple datasets:
      • Market data: time-series financial data.
      • Fundamental data: key company metrics (single-record dictionary).
      • News data: news articles related to the company.
    
    The integration pipeline:
      1. Clean market data (handle missing values, remove outliers, feature engineering, normalization).
      2. Process fundamental data by flattening key metrics.
      3. Convert news data to a DataFrame, process date fields, and aggregate (e.g., count articles by day).
      4. Merge the three datasets into a unified DataFrame by aligning on dates (with fundamentals repeated as static features).
    """
    
    def __init__(self, 
                 market_df: pd.DataFrame, 
                 fundamental_data: Dict, 
                 news_data: List[Dict],
                 output_dir: Optional[Path] = None):
        self.market_df = self._normalize_columns(market_df.copy())
        self.fundamental_data = fundamental_data
        # Convert DataFrame to list of dicts if needed
        self.news_data = news_data.to_dict('records') if isinstance(news_data, pd.DataFrame) else news_data
        self.output_dir = output_dir if output_dir else OUTPUT_DIR

        # Set up module-level logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent column names and case sensitivity"""
        column_map = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'dividends': 'Dividends',
            'stock splits': 'Stock Splits'
        }
        return df.rename(columns=lambda x: column_map.get(x.lower(), x))

    def clean_market_data(self, missing_method: str = "ffill", z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Clean the market data.
        - Convert date column.
        - Fill missing values.
        - Remove outliers based on z-score.
        - Create engineered features: Daily_Return and a moving average (SMA_10).
        - Normalize numeric columns with Min-Max scaling.
        
        Args:
            missing_method (str): Method to fill missing values ('ffill', 'bfill' or a value).
            z_threshold (float): Threshold for outlier removal.
        
        Returns:
            pd.DataFrame: Cleaned market data with date as index.
        """
        self.logger.info("Cleaning market data")
        df = self.market_df.copy()

        # Convert Date column to datetime and set as index (if present)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        
        # Handle missing values on all columns
        self.logger.info("Handling missing values in market data")
        if missing_method == "ffill":
            df.ffill(inplace=True)
        elif missing_method == "bfill":
            df.bfill(inplace=True)
        else:
            df.fillna(missing_method, inplace=True)
        
        # Remove outliers from numerical columns using z-score filtering
        self.logger.info("Removing outliers from market data")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        df = df[(np.abs(z_scores) < z_threshold).all(axis=1)]
        
        # Feature Engineering: Daily returns and 10-day moving average of 'Close' price
        self.logger.info("Performing feature engineering on market data")
        if "Close" in df.columns:
            df["Daily_Return"] = df["Close"].pct_change()
            df["SMA_10"] = df["Close"].rolling(window=10).mean()
        
        # Normalize numeric columns with Min-Max scaling
        self.logger.info("Normalizing market data")
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
        
        # Add validation for essential columns
        if "Close" not in df.columns:
            raise ValueError("Market data must contain 'Close' column")
            
        # Add fallback for empty data after outlier removal
        original_count = len(self.market_df)
        if len(df) < 0.5 * original_count:  # If we removed more than 50% of data
            self.logger.warning("Excessive outliers detected, using original data")
            df = self.market_df.copy()
        
        self.market_df = df
        return df
    
    def process_fundamentals(self) -> pd.DataFrame:
        """
        Process fundamental data by flattening the provided dictionary.
        For this example, we extract selected key metrics.
        
        Returns:
            pd.DataFrame: A single-row DataFrame of key fundamental metrics.
        """
        self.logger.info("Processing fundamental data")
        # Define the keys we are interested in
        keys = ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 'beta']
        flat_data = {key: self.fundamental_data.get(key, None) for key in keys}
        fundamental_df = pd.DataFrame([flat_data])
        self.logger.info("Fundamental data processed")
        return fundamental_df

    def process_news_data(self) -> pd.DataFrame:
        """
        Process news data by converting the list of dictionaries into a DataFrame.
        Extracts the publication date, title, and summary.
        Then, resamples to aggregate news count on a daily basis.
        
        Returns:
            pd.DataFrame: Aggregated news data with a date index and a 'news_count' column.
        """
        self.logger.info("Processing news data")
        if not self.news_data:
            self.logger.warning("No news data available to process")
            return pd.DataFrame()
        
        records = []
        for item in self.news_data:
            content = item.get("content", {})
            record = {
                "pubDate": pd.to_datetime(content.get("pubDate", None)),
                "title": content.get("title", ""),
                "summary": content.get("summary", "")
            }
            records.append(record)
        
        news_df = pd.DataFrame(records)
        # Drop records without a valid publication date
        news_df.dropna(subset=["pubDate"], inplace=True)
        news_df.set_index("pubDate", inplace=True)
        # Aggregate by day: count number of news articles per day
        self.logger.info("Aggregating news data by day")
        news_agg = news_df.resample("D").agg({"title": "count"}).rename(columns={"title": "news_count"})
        return news_agg

    def combine_data(self) -> pd.DataFrame:
        """
        Combine the cleaned market data, processed fundamental data, and aggregated news data.
        - The market data is cleaned and indexed by date.
        - News data is merged on the date index (with missing days filled as zero articles).
        - Fundamental data (static for the company) is added to each record.
        
        Returns:
            pd.DataFrame: A unified DataFrame ready for ML experiments.
        """
        self.logger.info("Combining market, fundamental, and news data")
        market_df = self.clean_market_data()
        fundamental_df = self.process_fundamentals()
        news_df = self.process_news_data()

        # Merge market data with aggregated news data via date index (left join)
        combined_df = market_df.copy()
        if not news_df.empty:
            combined_df = combined_df.join(news_df, how="left")
            combined_df["news_count"].fillna(0, inplace=True)
        
        # Append fundamental data as static features (repeated for every date)
        fundamental_static = fundamental_df.iloc[0].to_dict() if not fundamental_df.empty else {}
        for key, value in fundamental_static.items():
            combined_df[key] = value
        
        self.logger.info("Data combination complete")
        return combined_df

if __name__ == "__main__":
    # We have in the cache directory the market data, fundamental data and news data for each ticker
    # We can now loop through the list of tickers and process the data for each ticker
    # We will also save the combined data in the processed directory
    for ticker in TICKERS:
        # Add validation for market data existence
        market_path = f"{MARKET_DATA_DIR}/{ticker}_data.csv"
        if not Path(market_path).exists():
            print(f"Skipping {ticker} - missing market data")
            continue
            
        market_df = pd.read_csv(market_path)
        fundamental_data = pd.read_json(f"{FUNDAMENTAL_DATA_DIR}/{ticker}_fundamentals.json")
        news_df = pd.read_json(f"{NEWS_DATA_DIR}/{ticker}_news.json")
        news_data = news_df.to_dict('records')  # Convert DataFrame to list of dicts

        # Initialize the DataIntegrator with our data
        integrator = DataIntegrator(
            market_df=market_df, 
            fundamental_data=fundamental_data, 
            news_data=news_data,
            output_dir=OUTPUT_PREPROCESSED_DATA_DIR
        )

        # Combine the data
        combined_data = integrator.combine_data()
        print("Combined Data Sample:")
        print(combined_data.head())

        # Add data consistency check
        if combined_data.empty or combined_data.isna().all().any():
            print(f"Skipping {ticker} - failed to generate valid combined data")
            continue
            
        # Save with index to preserve dates
        combined_data.to_csv(f"{integrator.output_dir}/{ticker}_combined_data.csv", index=True)

        # We can now print the combined data
        print("Combined Data Sample:")
        print(combined_data.head())

