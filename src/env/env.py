import gym
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import rsi, macd_ml
from src.utils.config import DATA_DIR, TICKERS
from stable_baselines3 import PPO

# Define a custom trading environment
class EuroStoxxEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.current_index = 50  # start after enough data for indicators
        # state: [close, short_ma, long_ma, macd, rsi, atr, regime_score, ...]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0: sell, 1: hold, 2: buy
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.position = 0  # positive for long, negative for short
        self.entry_price = None

    def _get_state(self):
        row = self.data.iloc[self.current_index]
        # Assume the data contains precomputed indicators: 'close', 'short_ma', 'long_ma', 'macd', 'rsi', 'atr', 'regime'
        return np.array([row['close'], row['short_ma'], row['long_ma'], row['macd'], row['rsi'], row['atr'], row['regime']])

    def step(self, action):
        # Execute trade decision
        current_price = self.data.iloc[self.current_index]['close']

        # Reset reward and simulate a trade decision
        reward = 0

        if action == 2 and self.position == 0:  # Buy signal
            self.position = 1
            self.entry_price = current_price
        elif action == 0 and self.position == 1:  # Sell to close long
            pnl = current_price - self.entry_price
            reward = pnl  # Reward can be risk-adjusted later
            self.balance += pnl
            self.position = 0
            self.entry_price = None

        # Hold action or inaction does not change the position immediately, but we can add time-based rewards/penalties
        # Optionally, for risk management: penalize if the trade is moving adversely by more than an ATR multiple
        if self.position == 1:
            # Using a simple risk metric: difference between current_price and entry_price scaled by atr
            atr = self.data.iloc[self.current_index]['atr']
            adverse_move = self.entry_price - current_price
            if adverse_move > 1.5 * atr:
                # Force an exit with a penalty
                pnl = current_price - self.entry_price
                reward += pnl - 10  # penalty
                self.balance += pnl
                self.position = 0
                self.entry_price = None

        # Move to the next day
        self.current_index += 1
        done = self.current_index >= len(self.data)
        state = self._get_state() if not done else np.zeros(self.observation_space.shape)
        return state, reward, done, {}

    def reset(self):
        self.current_index = 50
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        return self._get_state()

# Data Preparation:
def load_data():
    # Load historical EURO STOXX 50 data and compute technical indicators
    data = pd.read_csv(f"{DATA_DIR}/{TICKERS[0]}.csv", parse_dates=['date'])
    # Compute indicators: for instance:
    data['short_ma'] = data['close'].rolling(window=20).mean()
    data['long_ma'] = data['close'].rolling(window=50).mean()
    data['macd'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['rsi'] = rsi(data['close'], period=14)  # custom function
    data['macd_ml'] = macd_ml(data['close'], period=14)           # custom function
    # Regime indicator (e.g., trending = 1, range = 0)
    data['regime'] = np.where(data['short_ma'] > data['long_ma'], 1, 0)
    data.dropna(inplace=True)
    return data

# Main training loop for RL Agent:
data = load_data()
env = EuroStoxxEnv(data)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)  # tune timesteps as needed

# Evaluate the trained agent on out-of-sample data and visualize performance.
