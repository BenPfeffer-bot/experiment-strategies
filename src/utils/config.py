"""
Configuration Module

This module handles project-wide configuration and paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Data directories
DATA_DIR = PROJECT_ROOT / "db"
CACHE_DIR = DATA_DIR / "cache"
FIGURES_DIR = DATA_DIR / "figures"

# Fetching
MARKET_DATA_DIR = CACHE_DIR / "market_data"
RAW_DATA_DIR = MARKET_DATA_DIR / "raw"


FUNDAMENTAL_DATA_DIR = CACHE_DIR / "fundamental_data"
NEWS_DATA_DIR = CACHE_DIR / "news_data"

# Processing
OUTPUT_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR = CACHE_DIR / "compiled"
# OUTPUT_MARKET_DATA_DIR = OUTPUT_DIR / "market_data"
# OUTPUT_FUNDAMENTAL_DATA_DIR = OUTPUT_DIR / "fundamental_data"
# OUTPUT_NEWS_DATA_DIR = OUTPUT_DIR / "news_data"


# Backtesting
# BACKTESTING_DIR = DATA_DIR / "backtesting"

# List of tickers to analyze
TICKERS = [
    "RMS.PA",
    "VOW3.DE",
    "ASML.AS",
    "SAN.MC",
    "BNP.PA",
    "ITX.MC",
    "MC.PA",
    "RACE.MI",
    "ENI.MI",
    "MUV2.DE",
    "ENEL.MI",
    "BAS.DE",
    "DG.PA",
    "SU.PA",
    "BMW.DE",
    "NOKIA.HE",
    "BBVA.MC",
    "AI.PA",
    "SGO.PA",
    "ADS.DE",
    "BAYN.DE",
    "DHL.DE",
    "OR.PA",
    "CS.PA",
    "BN.PA",
    "INGA.AS",
    "ADYEN.AS",
    "TTE.PA",
    "IFX.DE",
    "RI.PA",
    "WKL.AS",
    "DB1.DE",
    "SIE.DE",
    "AIR.PA",
    "PRX.AS",
    "ABI.BR",
    "STLAM.MI",
    "NDA-FI.HE",
    "MBG.DE",
    "ISP.MI",
    "IBE.MC",
    "KER.PA",
    "SAP.DE",
    "AD.AS",
    "ALV.DE",
    "UCG.MI",
    "EL.PA",
    "SAN.PA",
    "BBVA.MC",
    "ENEL.MI",
]

# Trading parameters
INITIAL_CAPITAL = 100000
TRANSACTION_COSTS = 0.001
RISK_FREE_RATE = 0.02  # Annual risk-free rate

# Technical analysis parameters
LOOKBACK_PERIODS = {"short": 20, "medium": 50, "long": 200}

# Risk management parameters
POSITION_SIZE_LIMITS = {"min": 0.01, "max": 0.3}

STOP_LOSS_LIMITS = {"min": 0.01, "max": 0.05}

# Market regime parameters
REGIME_DETECTION = {"n_regimes": 4, "lookback_period": 60, "min_samples": 252}

# Optimization parameters
OPTIMIZATION = {"n_initial_points": 10, "n_iterations": 50, "exploration_weight": 0.1}

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    CACHE_DIR,
    FIGURES_DIR,
    OUTPUT_DIR,
    MARKET_DATA_DIR,
    FUNDAMENTAL_DATA_DIR,
    NEWS_DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    # BACKTESTING_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
