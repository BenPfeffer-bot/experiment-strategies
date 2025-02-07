import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn import hmm
import joblib
import os
import sys
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


@dataclass
class MarketRegime:
    """Market regime characteristics"""

    label: int
    trend: str
    volatility: str
    liquidity: str
    avg_return: float
    avg_volume: float

    @property
    def description(self) -> str:
        """Generate a human-readable description of the regime"""
        return (
            f"{self.trend} | {self.volatility} Volatility | {self.liquidity} Liquidity"
        )


class MarketRegimeDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.hmm_model = None
        self.n_regimes = 4  # Fixed number of regimes

    def detect_regimes(
        self, data: pd.DataFrame
    ) -> Tuple[List[MarketRegime], pd.DataFrame]:
        """Detect market regimes using HMM"""
        self.logger.info("Starting regime detection...")

        try:
            # Prepare features for HMM
            features = self._prepare_features(data)

            # Initialize HMM with more robust parameters
            self.hmm_model = hmm.GaussianHMM(
                n_components=4,
                covariance_type="diag",  # Use diagonal covariance for stability
                n_iter=1000,
                tol=1e-5,
                random_state=42,
            )

            # Fit HMM with scaled data and error handling
            try:
                scaled_features = self.scaler.fit_transform(features)
                self.hmm_model.fit(scaled_features)
            except Exception as e:
                self.logger.warning(
                    f"HMM detection failed: {str(e)}. Using fallback method."
                )
                return self._fallback_regime_detection(data)

            # Predict regimes
            regime_labels = self.hmm_model.predict(scaled_features)

            # Calculate regime characteristics with error handling
            try:
                regimes = self._characterize_regimes(data, regime_labels)
                regime_data = self._create_regime_data(data, regime_labels, regimes)
                return regimes, regime_data
            except Exception as e:
                self.logger.error(f"Error in regime characterization: {str(e)}")
                return self._fallback_regime_detection(data)

        except Exception as e:
            self.logger.error(f"Regime detection failed: {str(e)}")
            return self._fallback_regime_detection(data)

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection"""
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Calculate technical indicators
        returns = pd.Series(data["returns"].fillna(0))
        volume = pd.Series(
            data["volume"].fillna(data["volume"].mean())
            if "volume" in data.columns
            else np.ones(len(data))
        )

        # Calculate features with explicit windows
        volatility = returns.rolling(window=20, min_periods=1).std().fillna(0)
        trend = returns.rolling(window=20, min_periods=1).mean().fillna(0)
        volume_change = pd.Series(volume).pct_change().fillna(0)

        # Stack features into a matrix
        feature_matrix = np.column_stack(
            [returns.values, volatility.values, trend.values, volume_change.values]
        )

        return feature_matrix

    def _characterize_regimes(
        self, data: pd.DataFrame, regime_labels: np.ndarray
    ) -> List[MarketRegime]:
        """Characterize each regime based on its properties"""
        regimes = []

        # Ensure data is properly aligned
        if len(data) != len(regime_labels):
            self.logger.error(
                f"Data length mismatch: data={len(data)}, labels={len(regime_labels)}"
            )
            return [
                MarketRegime(i, "Unknown", "Unknown", "Unknown", 0.0, 0.0)
                for i in range(self.n_regimes)
            ]

        for i in range(self.n_regimes):
            try:
                # Get data for this regime
                mask = regime_labels == i
                if not any(mask):
                    regimes.append(
                        MarketRegime(i, "Unknown", "Unknown", "Unknown", 0.0, 0.0)
                    )
                    continue

                regime_data = data.iloc[mask]

                # Calculate basic statistics
                returns = float(regime_data["returns"].mean())
                volatility = float(regime_data["returns"].std())
                volume = (
                    float(regime_data["volume"].mean())
                    if "volume" in regime_data
                    else 0.0
                )

                # Determine trend
                if abs(returns) < 0.0001:
                    trend = "Neutral"
                elif abs(returns) < 0.001:
                    trend = "Weak " + ("Uptrend" if returns > 0 else "Downtrend")
                else:
                    trend = "Strong " + ("Uptrend" if returns > 0 else "Downtrend")

                # Determine volatility level using percentiles
                all_vol = data["returns"].std()
                vol_percentile = stats.percentileofscore([all_vol], volatility)
                if vol_percentile < 33:
                    vol_level = "Low"
                elif vol_percentile < 66:
                    vol_level = "Medium"
                else:
                    vol_level = "High"

                # Create regime object
                regimes.append(
                    MarketRegime(
                        label=i,
                        trend=trend,
                        volatility=vol_level,
                        liquidity="Low",  # Simplified
                        avg_return=returns,
                        avg_volume=volume,
                    )
                )

            except Exception as e:
                self.logger.error(f"Error characterizing regime {i}: {str(e)}")
                regimes.append(
                    MarketRegime(i, "Unknown", "Unknown", "Unknown", 0.0, 0.0)
                )

        return regimes

    def _create_regime_data(
        self, data: pd.DataFrame, regime_labels: np.ndarray, regimes: List[MarketRegime]
    ) -> pd.DataFrame:
        """Create regime data DataFrame"""
        regime_data = pd.DataFrame(index=data.index)
        regime_data["regime"] = regime_labels

        # Add regime probabilities if HMM model is available
        if self.hmm_model is not None:
            try:
                scaled_features = self.scaler.transform(self._prepare_features(data))
                probs = self.hmm_model.predict_proba(scaled_features)
                regime_data["probability"] = np.max(probs, axis=1)
            except Exception as e:
                self.logger.warning(
                    f"Could not calculate regime probabilities: {str(e)}"
                )
                regime_data["probability"] = 1.0
        else:
            regime_data["probability"] = 1.0

        # Add regime descriptions
        regime_descriptions = {r.label: r.description for r in regimes}
        regime_data["description"] = regime_data["regime"].map(regime_descriptions)

        return regime_data

    def _fallback_regime_detection(
        self, data: pd.DataFrame
    ) -> Tuple[List[MarketRegime], pd.DataFrame]:
        """Fallback method for regime detection using simple rules"""
        self.logger.info("Using fallback regime detection method")

        # Calculate basic metrics
        returns = data["returns"].rolling(window=20).mean()
        volatility = data["returns"].rolling(window=20).std()

        # Define regime thresholds
        regime_labels = np.zeros(len(data))

        # Regime 0: Low vol, positive returns
        # Regime 1: Low vol, negative returns
        # Regime 2: High vol, positive returns
        # Regime 3: High vol, negative returns

        vol_threshold = volatility.median()
        for i in range(len(data)):
            if i < 20:  # Skip the first window where rolling calcs are invalid
                continue
            if volatility.iloc[i] > vol_threshold:
                if returns.iloc[i] > 0:
                    regime_labels[i] = 2
                else:
                    regime_labels[i] = 3
            else:
                if returns.iloc[i] > 0:
                    regime_labels[i] = 0
                else:
                    regime_labels[i] = 1

        # Create regime objects and data
        regimes = self._characterize_regimes(data, regime_labels)
        regime_data = self._create_regime_data(data, regime_labels, regimes)

        return regimes, regime_data

    def plot_regimes(
        self, data: pd.DataFrame, regime_data: pd.DataFrame, save_path: str = None
    ):
        """Plot regime detection results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Price with regime colors
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n_regimes))
        for i in range(self.n_regimes):
            mask = regime_data["regime"] == i
            ax1.scatter(
                data.index[mask],
                data["price"][mask],
                c=[colors[i]],
                label=f"Regime {i}",
                alpha=0.6,
            )
        ax1.set_title("Price with Market Regimes")
        ax1.legend()

        # Regime probabilities
        regime_data["probability"].plot(ax=ax2)
        ax2.set_title("Regime Probability")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_model(self, filepath: str):
        """Save the regime detection models"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "hmm_model": self.hmm_model,
                "scaler": self.scaler,
            },
            filepath,
        )

    @classmethod
    def load_model(cls, filepath: str):
        """Load saved regime detection models"""
        data = joblib.load(filepath)
        detector = cls()
        detector.hmm_model = data["hmm_model"]
        detector.scaler = data["scaler"]
        return detector
