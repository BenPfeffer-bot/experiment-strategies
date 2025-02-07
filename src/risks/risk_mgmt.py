# File: src/risk/risk_management.py
import numpy as np
import pandas as pd

def calculate_rolling_volatility(prices: pd.Series, window: int = 20):
    """
    Compute rolling annualized volatility from daily returns.
    """
    returns = prices.pct_change().dropna()
    daily_vol = returns.rolling(window=window).std()
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol

def calculate_position_size(account_size, risk_percent, entry_price, stop_loss, volatility=None):
    """
    Calculate the number of shares (or units) to trade.
    If volatility is provided (annualized), the stop-loss can be adapted as a multiple of volatility.
    
    Parameters:
      account_size: Total capital in currency.
      risk_percent: Fraction of account to risk per trade.
      entry_price: Entry price of the asset.
      stop_loss: Stop loss price.
      volatility: Optional annualized volatility.
    
    Returns:
      Position size (units).
    
    Example:
      If risk_amount = 1000, risk per unit = 2, position size = 500.
      With volatility adjustments, the stop loss can widen/narrow.
    """
    risk_amount = account_size * risk_percent
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        raise ValueError("Stop loss must differ from entry price.")
    position_size = risk_amount / risk_per_unit
    return position_size
