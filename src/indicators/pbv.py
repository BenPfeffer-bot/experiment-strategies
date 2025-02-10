"""
Enhanced Price-to-Book Value (PBV) Indicator

This module implements an advanced Price-to-Book Value indicator with sector-relative
analysis and historical trend detection. It provides valuation metrics and generates
trading signals based on relative value analysis.

Features:
- Multiple PBV calculation methods
- Sector-relative analysis
- Historical trend detection
- Valuation-based signals
- Advanced visualization
- Performance optimization

License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import logging
from datetime import datetime
from enum import Enum
import json
from collections import defaultdict

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import (
    MARKET_DATA_DIR,
    TICKERS,
    FIGURES_DIR,
    RAW_DATA_DIR,
    FUNDAMENTAL_DATA_DIR,
)


class PBVMethod(Enum):
    """Types of PBV calculations"""

    STANDARD = "standard"  # Simple price to book value
    SECTOR_RELATIVE = "sector_relative"  # Relative to sector average
    ADJUSTED = "adjusted"  # Adjusted for sector and growth


class ValuationZone(Enum):
    """Valuation zone classifications"""

    DEEP_VALUE = "deep_value"
    VALUE = "value"
    FAIR = "fair"
    GROWTH = "growth"
    EXPENSIVE = "expensive"


@dataclass
class PBVParameters:
    """Parameters for PBV calculations"""

    lookback_period: int = 252  # One year of trading days
    smoothing_period: int = 20  # For moving averages
    pbv_method: PBVMethod = PBVMethod.STANDARD
    price_column: str = "Close"
    signal_threshold: float = 0.2  # 20% deviation for signals
    zone_percentiles: Tuple[float, float, float, float] = (20.0, 40.0, 60.0, 80.0)


@dataclass
class PBVResult:
    """Container for PBV calculation results"""

    # Required fields without defaults
    pbv_ratio: pd.Series
    smoothed_pbv: pd.Series
    valuation_zone: pd.Series
    signals: pd.Series
    zscore: pd.Series
    trend: pd.Series
    # Optional fields with defaults
    sector_average: Optional[pd.Series] = None
    relative_value: Optional[pd.Series] = None


@dataclass
class FundamentalData:
    """Container for fundamental data"""

    price_to_book: float
    book_value: float
    current_price: float
    market_cap: float
    earnings_growth: float
    revenue_growth: float
    return_on_equity: float
    sector: str
    industry: str


class PBVIndicator:
    """
    Enhanced Price-to-Book Value Indicator Implementation.

    This indicator calculates PBV ratios using multiple methods and provides
    valuation analysis with signal generation. Features include:
    - Multiple PBV calculation methods
    - Sector-relative analysis
    - Historical trend detection
    - Signal generation based on valuation
    - Advanced visualization options

    Attributes:
        params (PBVParameters): Configuration parameters
        logger (logging.Logger): Logger instance
        results (Optional[PBVResult]): Computed results
    """

    def __init__(self, params: Optional[PBVParameters] = None):
        """
        Initialize PBV indicator with given parameters.

        Args:
            params: Optional[PBVParameters]
                Parameters for calculations. If None, uses default values.
        """
        self.params = params if params else PBVParameters()
        self._validate_parameters()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize results container
        self.results: Optional[PBVResult] = None

    def _validate_parameters(self) -> None:
        """Validate PBV parameters."""
        if self.params.lookback_period < 1:
            raise ValueError("lookback_period must be positive")
        if self.params.smoothing_period < 1:
            raise ValueError("smoothing_period must be positive")
        if not 0 < self.params.signal_threshold < 1:
            raise ValueError("signal_threshold must be between 0 and 1")
        if len(self.params.zone_percentiles) != 4:
            raise ValueError("zone_percentiles must have exactly 4 values")
        if not all(0 <= x <= 100 for x in self.params.zone_percentiles):
            raise ValueError("all zone_percentiles must be between 0 and 100")
        if not all(
            self.params.zone_percentiles[i] < self.params.zone_percentiles[i + 1]
            for i in range(len(self.params.zone_percentiles) - 1)
        ):
            raise ValueError("zone_percentiles must be in ascending order")

    def _calculate_pbv(self, prices: np.ndarray, book_values: np.ndarray) -> np.ndarray:
        """
        Calculate PBV ratio.

        Args:
            prices: np.ndarray
                Price series
            book_values: np.ndarray
                Book value per share series

        Returns:
            np.ndarray of PBV ratios
        """
        # Create mask for valid calculations
        valid_mask = (
            (book_values > 0)
            & ~np.isnan(book_values)
            & ~np.isnan(prices)
            & (prices > 0)
        )

        # Initialize PBV array with NaN
        pbv = np.full_like(prices, np.nan, dtype=float)

        # Calculate PBV only where we have valid data
        pbv[valid_mask] = prices[valid_mask] / book_values[valid_mask]

        # Handle extreme values (clip to reasonable range)
        pbv = np.nan_to_num(pbv, nan=np.nan)  # Convert inf to nan
        valid_pbv = pbv[~np.isnan(pbv)]
        if len(valid_pbv) > 0:
            # Use percentiles to determine reasonable clipping range
            p1, p99 = np.percentile(valid_pbv[valid_pbv > 0], [1, 99])
            clip_min = max(0.1, p1 / 2)  # Don't allow extremely small values
            clip_max = min(100, p99 * 2)  # Don't allow extremely large values
            pbv = np.clip(pbv, clip_min, clip_max)

        return pbv

    def _calculate_sector_relative(
        self, pbv: np.ndarray, sectors: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate sector-relative PBV metrics.

        Args:
            pbv: np.ndarray
                PBV ratios
            sectors: pd.Series
                Sector classifications

        Returns:
            Tuple of (sector averages, relative values)
        """
        sector_avg = np.full_like(pbv, np.nan, dtype=float)
        relative_val = np.full_like(pbv, np.nan, dtype=float)

        # Calculate for each sector
        for sector in sectors.unique():
            sector_mask = sectors == sector
            if any(sector_mask):
                sector_pbv = pbv[sector_mask]
                valid_pbv = sector_pbv[~np.isnan(sector_pbv)]
                if len(valid_pbv) > 0:
                    # Use rolling median for more stable sector average
                    rolling_median = (
                        pd.Series(sector_pbv)
                        .rolling(window=min(len(sector_pbv), 252), min_periods=1)
                        .median()
                    )
                    sector_avg[sector_mask] = rolling_median

                    # Calculate relative value
                    relative_val[sector_mask] = np.where(
                        ~np.isnan(sector_pbv) & (rolling_median > 0),
                        sector_pbv / rolling_median,
                        np.nan,
                    )

        # Handle any remaining NaN values with reasonable defaults
        relative_val = np.nan_to_num(relative_val, nan=1.0, posinf=2.0, neginf=0.5)
        sector_avg = np.nan_to_num(sector_avg, nan=1.0, posinf=10.0, neginf=0.1)

        return sector_avg, relative_val

    def _calculate_adjusted_pbv(
        self, pbv: np.ndarray, growth_rates: np.ndarray
    ) -> np.ndarray:
        """
                Calculate growth-adjusted PBV.

                Args:
                    pbv: np.ndarray
                        PBV ratios
                    growth_rates: np.ndarray
                        Earnings growth rates

        Returns:
                    np.ndarray of adjusted PBV ratios
        """
        # Add small constant to avoid division by zero
        growth_factor = 1 + np.maximum(growth_rates, -0.99)
        adjusted_pbv = pbv / growth_factor

        return adjusted_pbv

    def compute(self, data: pd.DataFrame) -> PBVResult:
        """
        Compute PBV and related metrics.
        """
        try:
            # Validate required columns
            required_columns = [self.params.price_column, "book_value_per_share"]
            if self.params.pbv_method == PBVMethod.SECTOR_RELATIVE:
                required_columns.append("sector")
            if self.params.pbv_method == PBVMethod.ADJUSTED:
                required_columns.append("earnings_growth")

            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Extract data
            prices = data[self.params.price_column].values
            book_values = data["book_value_per_share"].values

            # Calculate basic PBV
            pbv = self._calculate_pbv(prices, book_values)

            # Apply method-specific calculations
            sector_average = relative_value = None
            if self.params.pbv_method == PBVMethod.SECTOR_RELATIVE:
                sector_average, relative_value = self._calculate_sector_relative(
                    pbv, data["sector"]
                )
                pbv = relative_value  # Use relative PBV for signals
            elif self.params.pbv_method == PBVMethod.ADJUSTED:
                growth_adjustment = 1 + data["earnings_growth"].values
                pbv = pbv / growth_adjustment

            # Calculate smoothed PBV
            smoothed_pbv = (
                pd.Series(pbv)
                .rolling(window=self.params.smoothing_period, min_periods=1)
                .mean()
                .fillna(method="bfill")
                .values
            )

            # Calculate z-score using rolling window
            pbv_series = pd.Series(pbv)
            rolling_mean = pbv_series.rolling(
                window=self.params.lookback_period, min_periods=1
            ).mean()
            rolling_std = pbv_series.rolling(
                window=self.params.lookback_period, min_periods=1
            ).std()

            zscore = np.zeros_like(pbv)
            valid_mask = (
                (rolling_std != 0)
                & ~np.isnan(rolling_std)
                & ~np.isnan(rolling_mean)
                & ~np.isnan(pbv)
            )
            zscore[valid_mask] = (
                pbv[valid_mask] - rolling_mean[valid_mask]
            ) / rolling_std[valid_mask]

            # Determine valuation zones using rolling window
            zones = np.full_like(pbv, ValuationZone.FAIR.value, dtype=object)
            rolling_window = min(self.params.lookback_period, len(pbv))

            # Calculate rolling percentiles for the entire series
            pbv_series = pd.Series(pbv)
            rolling_percentiles = []

            for percentile in self.params.zone_percentiles:
                rolling_perc = pbv_series.rolling(
                    window=rolling_window, min_periods=1
                ).quantile(percentile / 100)
                rolling_percentiles.append(rolling_perc)

            # Assign zones based on rolling percentiles
            for i in range(len(pbv)):
                if np.isnan(pbv[i]):
                    continue

                current_percentiles = [p.iloc[i] for p in rolling_percentiles]

                if pbv[i] <= current_percentiles[0]:
                    zones[i] = ValuationZone.DEEP_VALUE.value
                elif pbv[i] <= current_percentiles[1]:
                    zones[i] = ValuationZone.VALUE.value
                elif pbv[i] <= current_percentiles[2]:
                    zones[i] = ValuationZone.FAIR.value
                elif pbv[i] <= current_percentiles[3]:
                    zones[i] = ValuationZone.GROWTH.value
                else:
                    zones[i] = ValuationZone.EXPENSIVE.value

            # Calculate trend
            trend = np.zeros_like(pbv)
            trend[1:] = np.sign(np.diff(smoothed_pbv))

            # Generate signals based on valuation
            signals = np.zeros_like(pbv)
            if (
                self.params.pbv_method == PBVMethod.SECTOR_RELATIVE
                and relative_value is not None
            ):
                # Signals based on relative value
                signals[relative_value < (1 - self.params.signal_threshold)] = (
                    1  # Buy signal
                )
                signals[
                    relative_value > (1 + self.params.signal_threshold)
                ] = -1  # Sell signal
            else:
                # Signals based on z-score
                signals[zscore < -self.params.signal_threshold] = 1  # Buy signal
                signals[zscore > self.params.signal_threshold] = -1  # Sell signal

            # Create result series
            self.results = PBVResult(
                pbv_ratio=pd.Series(pbv, index=data.index, name="PBV"),
                smoothed_pbv=pd.Series(
                    smoothed_pbv, index=data.index, name="Smoothed PBV"
                ),
                valuation_zone=pd.Series(zones, index=data.index, name="Zone"),
                signals=pd.Series(signals, index=data.index, name="Signals"),
                zscore=pd.Series(zscore, index=data.index, name="Z-Score"),
                trend=pd.Series(trend, index=data.index, name="Trend"),
                sector_average=pd.Series(
                    sector_average, index=data.index, name="Sector Average"
                )
                if sector_average is not None
                else None,
                relative_value=pd.Series(
                    relative_value, index=data.index, name="Relative Value"
                )
                if relative_value is not None
                else None,
            )

            return self.results

        except Exception as e:
            self.logger.error(f"Error computing PBV: {str(e)}")
            raise

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current indicator state.

        Returns:
            Dict containing current values for PBV metrics
        """
        if self.results is None:
            raise ValueError("PBV not computed. Call compute() first.")

        state = {
            "pbv": float(self.results.pbv_ratio.iloc[-1]),
            "smoothed_pbv": float(self.results.smoothed_pbv.iloc[-1]),
            "zone": self.results.valuation_zone.iloc[-1],
            "signal": float(self.results.signals.iloc[-1]),
            "zscore": float(self.results.zscore.iloc[-1]),
            "trend": float(self.results.trend.iloc[-1]),
        }

        if self.results.sector_average is not None:
            state["sector_average"] = float(self.results.sector_average.iloc[-1])
        if self.results.relative_value is not None:
            state["relative_value"] = float(self.results.relative_value.iloc[-1])

        return state

    def plot(
        self,
        price_data: Optional[pd.Series] = None,
        show_signals: bool = True,
        show_zones: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot PBV indicator with price data.

        Args:
            price_data: Optional[pd.Series]
                Price data to plot
            show_signals: bool
                If True, shows trading signals
            show_zones: bool
                If True, shows valuation zones
            save_path: Optional[str]
                If provided, saves plot to this path
            title: Optional[str]
                Custom plot title
        """
        if self.results is None:
            raise ValueError("PBV not computed. Call compute() first.")

        try:
            # Create figure with subplots
            n_plots = 2 + (1 if show_zones else 0)
            fig, axes = plt.subplots(
                n_plots,
                1,
                figsize=(12, 4 * n_plots),
                height_ratios=[2, 1] + ([1] if show_zones else []),
            )

            # Plot price and PBV
            ax_price = axes[0]
            if price_data is not None:
                ax_price.plot(
                    price_data.index,
                    price_data,
                    label="Price",
                    color="black",
                    linewidth=1,
                )
                ax_price.set_title("Price and PBV")

                # Plot PBV overlay
                ax_pbv = ax_price.twinx()
                ax_pbv.plot(
                    self.results.pbv_ratio.index,
                    self.results.pbv_ratio,
                    label="PBV",
                    color="blue",
                    alpha=0.5,
                )
                ax_pbv.plot(
                    self.results.smoothed_pbv.index,
                    self.results.smoothed_pbv,
                    label="Smoothed PBV",
                    color="red",
                    linestyle="--",
                )
                ax_pbv.set_ylabel("PBV Ratio")

            # Plot sector comparison if available
            ax_rel = axes[1]
            if self.results.sector_average is not None:
                ax_rel.plot(
                    self.results.sector_average.index,
                    self.results.sector_average,
                    label="Sector Average",
                    color="gray",
                    linestyle="--",
                )
                ax_rel.plot(
                    self.results.relative_value.index,
                    self.results.relative_value,
                    label="Relative Value",
                    color="purple",
                )
            else:
                ax_rel.plot(
                    self.results.zscore.index,
                    self.results.zscore,
                    label="Z-Score",
                    color="purple",
                )

            # Plot signals if requested
            if show_signals:
                signals = self.results.signals
                buy_points = self.results.pbv_ratio[signals == 1]
                sell_points = self.results.pbv_ratio[signals == -1]

                ax_rel.scatter(
                    buy_points.index,
                    [0] * len(buy_points),
                    color="green",
                    marker="^",
                    s=100,
                    label="Buy Signal",
                )
                ax_rel.scatter(
                    sell_points.index,
                    [0] * len(sell_points),
                    color="red",
                    marker="v",
                    s=100,
                    label="Sell Signal",
                )

            ax_rel.set_title("Relative Value Analysis")
            ax_rel.grid(True, alpha=0.3)
            ax_rel.legend()

            # Plot valuation zones if requested
            if show_zones:
                ax_zone = axes[2]
                zones = pd.Series(
                    np.where(
                        self.results.valuation_zone == ValuationZone.EXPENSIVE.value,
                        2,
                        np.where(
                            self.results.valuation_zone == ValuationZone.GROWTH.value,
                            1,
                            np.where(
                                self.results.valuation_zone
                                == ValuationZone.VALUE.value,
                                -1,
                                np.where(
                                    self.results.valuation_zone
                                    == ValuationZone.DEEP_VALUE.value,
                                    -2,
                                    0,
                                ),
                            ),
                        ),
                    ),
                    index=self.results.valuation_zone.index,
                )
                ax_zone.plot(zones.index, zones, color="blue", label="Valuation Zone")
                ax_zone.set_yticks([-2, -1, 0, 1, 2])
                ax_zone.set_yticklabels(
                    ["Deep Value", "Value", "Fair", "Growth", "Expensive"]
                )
                ax_zone.set_title("Valuation Zone")
                ax_zone.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            raise ValueError(f"Error plotting PBV: {str(e)}")


def load_fundamental_data(filepath: str) -> FundamentalData:
    """Load and parse fundamental data from JSON file"""
    with open(filepath, "r") as f:
        data = json.load(f)

    return FundamentalData(
        price_to_book=data.get("priceToBook", 0),
        book_value=data.get("bookValue", 0),
        current_price=data.get("currentPrice", 0),
        market_cap=data.get("marketCap", 0),
        earnings_growth=data.get("earningsGrowth", 0),
        revenue_growth=data.get("revenueGrowth", 0),
        return_on_equity=data.get("returnOnEquity", 0),
        sector=data.get("sector", "Unknown"),
        industry=data.get("industry", "Unknown"),
    )


if __name__ == "__main__":
    """Example usage of the PBV indicator"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create output directory
        pbv_output_dir = os.path.join(FIGURES_DIR, "pbv")
        os.makedirs(pbv_output_dir, exist_ok=True)

        # Dictionary to store sector and industry data
        sector_data = defaultdict(list)
        industry_data = defaultdict(list)

        # First pass: collect sector and industry data
        logger.info("Collecting sector and industry data...")
        for ticker in TICKERS:
            try:
                fundamental_data_path = os.path.join(
                    FUNDAMENTAL_DATA_DIR, f"{ticker}_fundamentals.json"
                )
                fund_data = load_fundamental_data(fundamental_data_path)

                if fund_data.price_to_book > 0:
                    sector_data[fund_data.sector].append(fund_data.price_to_book)
                    industry_data[fund_data.industry].append(fund_data.price_to_book)
            except Exception as e:
                logger.warning(
                    f"Could not load fundamental data for {ticker}: {str(e)}"
                )

        # Calculate sector and industry medians
        sector_medians = {
            sector: np.median(values) for sector, values in sector_data.items()
        }
        industry_medians = {
            industry: np.median(values) for industry, values in industry_data.items()
        }

        # Second pass: process each ticker
        for ticker in TICKERS:
            try:
                logger.info(f"Processing PBV for {ticker}")

                # Load market data
                market_data_path = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
                market_df = pd.read_csv(market_data_path)
                market_df["Date"] = pd.to_datetime(market_df["Date"], utc=True)
                market_df.set_index("Date", inplace=True)

                # Load fundamental data
                fundamental_data_path = os.path.join(
                    FUNDAMENTAL_DATA_DIR, f"{ticker}_fundamentals.json"
                )
                fund_data = load_fundamental_data(fundamental_data_path)

                # Get sector and industry medians
                sector_median = sector_medians.get(
                    fund_data.sector, fund_data.price_to_book
                )
                industry_median = industry_medians.get(
                    fund_data.industry, fund_data.price_to_book
                )

                # Create fundamental DataFrame
                fundamental_df = pd.DataFrame(
                    {
                        "book_value": fund_data.book_value,
                        "sector": fund_data.sector,
                        "industry": fund_data.industry,
                        "earnings_growth": fund_data.earnings_growth,
                        "revenue_growth": fund_data.revenue_growth,
                        "return_on_equity": fund_data.return_on_equity,
                        "price_to_book": fund_data.price_to_book,
                        "sector_median_pbv": sector_median,
                        "industry_median_pbv": industry_median,
                    },
                    index=[market_df.index[-1]],
                )

                # Expand fundamental data to match market data timeframe
                fundamental_df = fundamental_df.reindex(market_df.index, method="ffill")

                # Calculate historical book values
                if fund_data.price_to_book > 0 and fund_data.current_price > 0:
                    # Calculate current book value per share
                    current_book_value = (
                        fund_data.current_price / fund_data.price_to_book
                    )

                    # Calculate historical book values based on price changes
                    price_ratio = market_df["Close"] / market_df["Close"].iloc[-1]
                    fundamental_df["book_value_per_share"] = (
                        current_book_value * price_ratio
                    )
                else:
                    # If we don't have valid P/B ratio, use static book value
                    fundamental_df["book_value_per_share"] = fund_data.book_value

                # Merge data
                df = pd.merge(
                    market_df,
                    fundamental_df,
                    left_index=True,
                    right_index=True,
                    how="left",
                )

                # Forward fill any missing data
                df = df.ffill()

                # Calculate actual PBV ratio for verification
                df["actual_pbv"] = df["Close"] / df["book_value_per_share"]

                # Initialize indicator with custom parameters
                params = PBVParameters(
                    lookback_period=252,
                    smoothing_period=20,
                    pbv_method=PBVMethod.SECTOR_RELATIVE,
                    price_column="Close",
                    signal_threshold=0.2,
                    zone_percentiles=(20.0, 40.0, 60.0, 80.0),
                )
                pbv_indicator = PBVIndicator(params)

                # Compute indicator
                results = pbv_indicator.compute(df)

                # Get current state
                current_state = pbv_indicator.get_current_state()
                logger.info(
                    f"{ticker} current PBV: {current_state['pbv']:.2f} "
                    f"(Sector: {fund_data.sector}, Sector Median: {sector_median:.2f}, "
                    f"Industry: {fund_data.industry}, Industry Median: {industry_median:.2f}, "
                    f"ROE: {fund_data.return_on_equity:.1%}, "
                    f"Growth: {fund_data.earnings_growth:.1%}, "
                    f"Zone: {current_state['zone']}, "
                    f"Z-Score: {current_state['zscore']:.2f})"
                )

                # Plot indicator
                save_path = os.path.join(pbv_output_dir, f"{ticker}_pbv.png")
                pbv_indicator.plot(
                    price_data=df["Close"],
                    show_signals=True,
                    show_zones=True,
                    save_path=save_path,
                    title=f"Price-to-Book Value Analysis - {ticker}",
                )

                logger.info(f"PBV plot saved for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
