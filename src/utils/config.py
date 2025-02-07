"""
Configuration Module

This module handles project-wide configuration and paths.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "db"
CACHE_DIR = DATA_DIR / "cache"
FIGURES_DIR = DATA_DIR / "figures"

# Fetching
MARKET_DATA_DIR = CACHE_DIR / "market_data"
FUNDAMENTAL_DATA_DIR = CACHE_DIR / "fundamental_data"
NEWS_DATA_DIR = CACHE_DIR / "news_data"

# Processing
OUTPUT_DIR = DATA_DIR / "processed"
# OUTPUT_MARKET_DATA_DIR = OUTPUT_DIR / "market_data"
# OUTPUT_FUNDAMENTAL_DATA_DIR = OUTPUT_DIR / "fundamental_data"
# OUTPUT_NEWS_DATA_DIR = OUTPUT_DIR / "news_data"
OUTPUT_PREPROCESSED_DATA_DIR = OUTPUT_DIR / "preprocessed_data"

# Backtesting
BACKTESTING_DIR = DATA_DIR / "backtesting"

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

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    CACHE_DIR,
    FIGURES_DIR,
    OUTPUT_DIR,
    MARKET_DATA_DIR,
    FUNDAMENTAL_DATA_DIR,
    NEWS_DATA_DIR,
    OUTPUT_PREPROCESSED_DATA_DIR,
    BACKTESTING_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
