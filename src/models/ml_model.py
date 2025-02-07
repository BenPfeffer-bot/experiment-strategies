# File: src/models/ml_model.py
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union
import os
import sys
import joblib
import talib
from sklearn.utils.class_weight import compute_sample_weight
from collections import defaultdict
from sklearn import set_config
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.utils.config import MARKET_DATA_DIR, TICKERS

set_config(enable_metadata_routing=True)


def load_signals_data(ticker: str) -> pd.DataFrame:
    """
    Load integrated signals CSV data.
    """
    filepath = f"db/cache/market_data/processed/{ticker}_integrated_signals.csv"
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and features."""
    try:
        # Calculate returns and log returns
        df["returns"] = df["price"].pct_change()
        df["log_returns"] = np.log(df["price"]).diff()

        # Calculate RSI
        df["rsi"] = talib.RSI(df["price"].values, timeperiod=14)

        # Calculate MACD
        macd, signal, hist = talib.MACD(
            df["price"].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["macd"] = macd

        # Calculate volatility (20-day rolling standard deviation)
        df["volatility"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

        # Calculate volatility ratio (current volatility / 100-day mean volatility)
        volatility_100d = df["returns"].rolling(window=100).std() * np.sqrt(252)
        df["volatility_ratio"] = (
            df["volatility"] / volatility_100d.rolling(window=20).mean()
        )

        # Calculate moving averages
        for period in [5, 10, 20]:
            ma_col = f"ma_{period}"
            df[ma_col] = df["price"].rolling(window=period).mean()
            # Calculate distance to moving average as percentage
            df[f"dist_to_ma_{period}"] = (df["price"] - df[ma_col]) / df[ma_col] * 100

        # Set ma to ma_20 for compatibility
        df["ma"] = df["ma_20"]

        # Forward fill NaN values
        df = df.fillna(method="ffill")
        # Fill remaining NaN values with 0
        df = df.fillna(0)

        return df

    except Exception as e:
        logger.error(f"Error calculating technical features: {str(e)}")
        raise


class MLModel:
    def __init__(self, features: List[str] = None):
        """Initialize the ML model with specified features."""
        if features is None:
            self.features = [
                "rsi",
                "macd",
                "volatility",
                "volatility_ratio",
                "dist_to_ma_5",
                "dist_to_ma_10",
                "dist_to_ma_20",
            ]
        else:
            self.features = features

        self.target = "target"
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.scaler = StandardScaler()

    def set_features(self, features: List[str]) -> None:
        """Set or update the features list."""
        self.features = features

    def set_target(self, target: str) -> None:
        """Set or update the target variable."""
        self.target = target

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction by scaling and handling missing values."""
        # Select only the required features
        X = data[self.features].copy()

        # Handle missing values
        X = X.fillna(method="ffill").fillna(0)

        # Scale the features
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit(self, data: pd.DataFrame) -> None:
        """Train the model on the provided data."""
        X = self._prepare_features(data)
        y = data[self.target].values
        self.model.fit(X, y)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the provided data."""
        try:
            X = self._prepare_features(data)
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            return np.zeros(len(data))  # Return neutral signals on error

    def generate_signal(self, data: pd.DataFrame) -> np.ndarray:
        """Generate trading signals for backtesting. Wraps the predict method."""
        return self.predict(data)

    def train_ml_model(self, ticker: str) -> None:
        """Train the ML model for a specific ticker."""
        try:
            # Load data
            df = load_signals_data(ticker)

            # Calculate technical features
            df = calculate_technical_features(df)

            # Create target variable (1 for price increase, 0 for decrease)
            df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)

            # Drop rows with NaN values
            df = df.dropna()

            # Prepare features and target
            X = self._prepare_features(df)
            y = df[self.target].values

            # Train the model
            self.model.fit(X, y)

            # Calculate and log training accuracy
            train_predictions = self.model.predict(X)
            train_accuracy = accuracy_score(y, train_predictions)
            logger.info(f"Training accuracy for {ticker}: {train_accuracy:.2f}")

        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            raise


if __name__ == "__main__":
    for ticker in TICKERS:
        print(f"\nTraining ML strategy for {ticker}")
        try:
            model = MLModel()
            model.train_ml_model(ticker)
            print(f"Model trained successfully with {len(model.feature_cols)} features")
        except Exception as e:
            print(f"Error training model for {ticker}: {str(e)}")
