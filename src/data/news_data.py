import yfinance as yf
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import NEWS_DATA_DIR, TICKERS

class NewsDataLoader:
    """
    A class to fetch and process recent news data for a specified ticker using yfinance.
    
    Best practices implemented:
      • Modularity: Encapsulated functionality via class methods.
      • Readability: Clear docstrings, type annotations, and inline comments.
      • Error handling and logging: Logging useful status messages and error conditions.
      • Extendibility: Allows filtering of news by keywords.
    """
    
    def __init__(self, ticker: str, news_data_dir: Optional[Path] = None):
        """
        Initialize the NewsDataLoader with a ticker symbol.
        
        Args:
            ticker (str): The ticker symbol for which to fetch news.
        """
        self.ticker: str = ticker if ticker else TICKERS
        self.news_data_dir: Path = news_data_dir if news_data_dir else NEWS_DATA_DIR
        self.news: Optional[List[Dict]] = None
        
        # Set up logger for this module
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_news(self) -> List[Dict]:
        """
        Fetch recent news articles for the given ticker using yfinance.
        
        Returns:
            List[Dict]: A list of news articles (each represented as a dictionary).
        """
        self.logger.info(f"Fetching news for ticker: {self.ticker}")
        try:
            ticker_obj = yf.Ticker(self.ticker)
            # yfinance returns a list of dictionaries containing news items
            self.news = ticker_obj.news
            if not self.news:
                self.logger.warning(f"No news found for ticker: {self.ticker}")
            else:
                self.logger.info(f"Retrieved {len(self.news)} news items for ticker: {self.ticker}")
        except Exception as e:
            self.logger.error(f"Error fetching news for {self.ticker}: {e}")
            raise
        return self.news if self.news is not None else []
    
    def filter_news_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Filter the fetched news articles by a specified keyword. The search is performed
        on the title and summary of the news articles.
        
        Args:
            keyword (str): The keyword to filter news articles.
        
        Returns:
            List[Dict]: A list of news items that contain the specified keyword.
        """
        if not self.news:
            self.logger.error("News data not available. Run fetch_news() first.")
            return []
        
        filtered = []
        for item in self.news:
            # Assuming news items may contain keys such as 'title' and 'summary'
            title = item.get("title", "").lower()
            summary = item.get("summary", "").lower()
            if keyword.lower() in title or keyword.lower() in summary:
                filtered.append(item)
        self.logger.info(f"Filtered news; found {len(filtered)} items containing keyword '{keyword}'.")
        return filtered

    def save_to_json(self, filename: str) -> None:
        """
        Save the fetched news data to a JSON file.
        
        Args:
            filename (str): Path to the file where the news data will be saved.
        """
        if not self.news:
            self.logger.error("No news data available to save. Run fetch_news() first.")
            return
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.news, f, indent=4)
            self.logger.info(f"News data saved to {filename}.")
        except Exception as e:
            self.logger.error(f"Error saving news data to {filename}: {e}")
            raise

if __name__ == "__main__":
    # Example usage: Fetching and filtering news for 'AAPL'
    # We have added a list of tickers in the config file with all the tickers we want to fetch data for
    # We can now loop through the list of tickers and fetch the data for each ticker
    # We ensure its saved in the cache directory
    for ticker in TICKERS:
        news_loader = NewsDataLoader(ticker=ticker)
        try:
            # Fetch news data from yfinance
            news_items = news_loader.fetch_news()
            print(f"Total news items fetched: {len(news_items)}")
            
            # Filter news for a specific keyword (e.g., 'earnings')
            keyword = "earnings" # TODO: Make this a parameter
            filtered_news = news_loader.filter_news_by_keyword(keyword)
            print(f"\nNews items containing the keyword '{keyword}':")
            for item in filtered_news:
                print(f"- {item.get('title', 'No Title')}")
            
            # Save the complete news list to a JSON file
            news_loader.save_to_json(f"{news_loader.news_data_dir}/{news_loader.ticker}_news.json")
        except Exception as ex:
            print(f"An error occurred: {ex}")
            raise
