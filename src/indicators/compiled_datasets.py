import os
import sys
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import MARKET_DATA_DIR, TICKERS, FIGURES_DIR, RAW_DATA_DIR
from src.indicators.macd import MACD
from src.indicators.rsi import RSIMultiLength
from src.indicators.kelter_channel import KelterChannel
from src.indicators.fibonacci import FibonacciTrailingStop
