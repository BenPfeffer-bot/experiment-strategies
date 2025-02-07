import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import MARKET_DATA_DIR, TICKERS


class MLStrategy:
    def __init__(self, ticker: str, lookback_period: int = 20):
        self.ticker = ticker
        self.lookback_period = lookback_period
        self.model = None
        self.scaler = StandardScaler()

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for ML model
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features["price"] = df["price"]
        features["returns"] = df["price"].pct_change()
        features["log_returns"] = np.log(df["price"] / df["price"].shift(1))

        # Technical indicators (reusing existing ones)
        features["rsi"] = df["avg_rsi"]
        features["macd"] = df["macd_macd"]
        features["macd_signal"] = df["macd_signal"]
        features["ma"] = df["ma"]

        # Volatility features
        features["volatility"] = (
            features["returns"].rolling(window=self.lookback_period).std()
        )
        features["volatility_ratio"] = features["volatility"] / features[
            "volatility"
        ].shift(self.lookback_period)

        # Momentum features
        for period in [5, 10, 20]:
            features[f"momentum_{period}"] = df["price"].pct_change(period)
            features[f"volume_momentum_{period}"] = (
                df["volume"].pct_change(period) if "volume" in df.columns else 0
            )

        # Mean reversion features
        for period in [5, 10, 20]:
            rolling_mean = df["price"].rolling(window=period).mean()
            features[f"dist_to_ma_{period}"] = (
                df["price"] - rolling_mean
            ) / rolling_mean

        # Target variable (next day return)
        features["target"] = np.where(df["price"].shift(-1) > df["price"], 1, 0)

        return features.dropna()

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare data for training/prediction
        """
        features_df = self._create_features(df)

        # Separate features and target
        X = features_df.drop(["target"], axis=1)
        y = features_df["target"]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, X.columns

    def train(self, df: pd.DataFrame, model_type: str = "rf"):
        """
        Train the ML model with cross-validation
        """
        X_scaled, y, feature_names = self.prepare_data(df)

        # Choose model
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )

        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=tscv, scoring="accuracy"
        )

        # Train final model on full dataset
        self.model.fit(X_scaled, y)

        # Feature importance analysis
        if hasattr(self.model, "feature_importances_"):
            importances = pd.Series(
                self.model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)

            print(f"\nTop 10 important features for {self.ticker}:")
            print(importances.head(10))

        return cv_scores.mean(), cv_scores.std()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled, _, _ = self.prepare_data(df)
        return self.model.predict(X_scaled)

    def get_prediction_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled, _, _ = self.prepare_data(df)
        return self.model.predict_proba(X_scaled)

    def save_model(self, path: str):
        """
        Save trained model and scaler
        """
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load_model(self, path: str):
        """
        Load trained model and scaler
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        saved_objects = joblib.load(path)
        self.model = saved_objects["model"]
        self.scaler = saved_objects["scaler"]


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate model performance
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    for ticker in TICKERS:
        print(f"\nTraining ML strategy for {ticker}")

        # Load data
        try:
            df = pd.read_csv(
                os.path.join(
                    MARKET_DATA_DIR, "processed", f"{ticker}_integrated_signals.csv"
                ),
                parse_dates=["Date"],
                index_col="Date",
            )

            # Initialize and train model
            ml_strategy = MLStrategy(ticker)
            cv_mean, cv_std = ml_strategy.train(df)

            print(f"Cross-validation accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})")

            # Save model
            model_path = os.path.join("models", "ml", f"{ticker}_model.joblib")
            ml_strategy.save_model(model_path)

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
