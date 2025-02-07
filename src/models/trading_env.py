import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import os
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import MARKET_DATA_DIR, TICKERS


@dataclass
class MarketState:
    price: float
    position: float
    balance: float
    pnl: float
    entry_price: float
    features: np.ndarray


class TradingEnv(gym.Env):
    """
    Enhanced Trading Environment with realistic market dynamics and optimized performance
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.001,
        position_size_pct: float = 0.2,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        window_size: int = 20,
        n_threads: int = 4,
    ):
        super(TradingEnv, self).__init__()

        # Initialize environment parameters
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.window_size = window_size

        # Parallel processing
        self.n_threads = n_threads
        self.executor = ThreadPoolExecutor(max_workers=n_threads)

        # Pre-calculate features
        self._preprocess_data()

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: price data + technical indicators + account info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * len(self.feature_columns) + 4,),
            dtype=np.float32,
        )

        # Initialize state cache
        self.state_cache = {}
        self.reset()

    def _preprocess_data(self):
        """Pre-calculate and normalize features"""
        # Define feature columns
        self.feature_columns = [
            "price",
            "ma",
            "kelter_upper",
            "kelter_lower",
            "avg_rsi",
            "macd_macd",
            "macd_signal",
            "fib_dir",
            "variance",
        ]

        # Calculate normalization parameters
        self.feature_means = {}
        self.feature_stds = {}

        for col in self.feature_columns:
            if col in self.df.columns:
                self.feature_means[col] = self.df[col].mean()
                self.feature_stds[col] = self.df[col].std()
            else:
                self.feature_means[col] = 0
                self.feature_stds[col] = 1

        # Pre-normalize features
        self.normalized_features = pd.DataFrame()
        for col in self.feature_columns:
            if col in self.df.columns:
                self.normalized_features[col] = (
                    self.df[col] - self.feature_means[col]
                ) / self.feature_stds[col]
            else:
                self.normalized_features[col] = 0

    def _get_observation(self) -> np.ndarray:
        """Create observation with cached historical data and current state"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape[0])

        # Check cache
        if self.current_step in self.state_cache:
            cached_state = self.state_cache[self.current_step]
            # Update account state
            account_state = np.array(
                [
                    self.balance / self.initial_balance,
                    1 if self.position > 0 else 0,
                    1 if self.position < 0 else 0,
                    self.current_pnl / self.initial_balance,
                ]
            )
            return np.concatenate([cached_state, account_state])

        # Get historical normalized features
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        obs_slice = self.normalized_features.iloc[start_idx:end_idx]

        if len(obs_slice) < self.window_size:
            # Pad with zeros if not enough history
            pad_length = self.window_size - len(obs_slice)
            obs_slice = pd.concat(
                [
                    pd.DataFrame(
                        np.zeros((pad_length, len(self.feature_columns))),
                        columns=self.feature_columns,
                    ),
                    obs_slice,
                ]
            )

        # Flatten features
        flattened_features = obs_slice.values.flatten()

        # Cache the features
        self.state_cache[self.current_step] = flattened_features

        # Add account state
        account_state = np.array(
            [
                self.balance / self.initial_balance,
                1 if self.position > 0 else 0,
                1 if self.position < 0 else 0,
                self.current_pnl / self.initial_balance,
            ]
        )

        return np.concatenate([flattened_features, account_state]).astype(np.float32)

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward with improved risk-adjusted metrics"""
        reward = 0

        # PnL-based reward
        if self.position != 0:
            current_price = self.df.iloc[self.current_step]["price"]
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if self.position < 0:
                pnl_pct = -pnl_pct

            # Risk-adjusted reward
            volatility = self.df["variance"].iloc[self.current_step]
            if volatility > 0:
                reward += pnl_pct / np.sqrt(volatility)  # Sharpe-ratio-like adjustment
            else:
                reward += pnl_pct

        # Transaction cost penalty
        if action != 0:
            reward -= self.transaction_cost_pct

        # Risk management reward/penalty
        if self.position != 0:
            if abs(reward) > self.take_profit_pct:
                reward *= 1.1  # Bonus for reaching take profit
            elif abs(reward) > self.stop_loss_pct:
                reward *= 0.9  # Penalty for hitting stop loss

        # Trend alignment reward
        if self.position != 0:
            trend = (
                self.df["ma"].iloc[self.current_step]
                > self.df["ma"].iloc[self.current_step - 1]
            )
            if (self.position > 0 and trend) or (self.position < 0 and not trend):
                reward *= 1.05  # Bonus for trend alignment

        return reward

    def _apply_slippage(self, price: float, action: int) -> float:
        """Apply realistic slippage based on volatility"""
        volatility = self.df["variance"].iloc[self.current_step]
        adjusted_slippage = self.slippage_pct * (1 + np.sqrt(volatility))
        slippage = np.random.uniform(-adjusted_slippage, adjusted_slippage)

        if action == 1:  # Buy
            return price * (1 + slippage)
        elif action == 2:  # Sell
            return price * (1 - slippage)
        return price

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment with parallel state updates"""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, {"balance": self.balance}

        current_price = self.df.iloc[self.current_step]["price"]

        # Apply transaction costs and slippage
        execution_price = self._apply_slippage(current_price, action)
        transaction_cost = execution_price * self.transaction_cost_pct

        # Execute trades
        if action == 1 and self.position <= 0:  # Buy
            max_position = (self.balance * self.position_size_pct) / execution_price
            self.position = max_position
            self.entry_price = execution_price
            self.balance -= execution_price * max_position + transaction_cost

        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:
                self.balance += execution_price * self.position - transaction_cost
                self.position = 0
                self.entry_price = 0

        # Calculate reward
        reward = self._calculate_reward(action)

        # Update current PnL
        if self.position != 0:
            self.current_pnl = (current_price - self.entry_price) * self.position
        else:
            self.current_pnl = 0

        # Move to next step
        self.current_step += 1

        # Clean old cache entries
        if len(self.state_cache) > self.window_size * 2:
            min_key = min(self.state_cache.keys())
            del self.state_cache[min_key]

        # Check for episode end
        done = self.current_step >= len(self.df) - 1

        # Prepare info dict
        info = {
            "balance": self.balance,
            "position": self.position,
            "pnl": self.current_pnl,
            "price": current_price,
            "execution_price": execution_price,
            "transaction_cost": transaction_cost,
        }

        return self._get_observation(), reward, done, info

    def reset(self):
        """Reset the environment and clear caches"""
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = self.window_size
        self.current_pnl = 0
        self.state_cache.clear()
        return self._get_observation()

    def render(self, mode="human"):
        """Render the environment state with additional metrics"""
        print(f"\nStep: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position:.4f}")
        print(f"Current PnL: ${self.current_pnl:.2f}")
        if self.position != 0:
            print(f"Entry Price: ${self.entry_price:.2f}")
        current_price = self.df.iloc[self.current_step]["price"]
        print(f"Current Price: ${current_price:.2f}")

        # Additional metrics
        if self.position != 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            if self.position < 0:
                pnl_pct = -pnl_pct
            print(f"PnL %: {pnl_pct:.2f}%")

        volatility = self.df["variance"].iloc[self.current_step]
        print(f"Current Volatility: {np.sqrt(volatility):.4f}")
        print(
            f"Risk-Adjusted PnL: {self.current_pnl / (np.sqrt(volatility) if volatility > 0 else 1):.2f}"
        )


if __name__ == "__main__":
    # Test the environment
    for ticker in TICKERS[:1]:
        df = pd.read_csv(
            os.path.join(
                MARKET_DATA_DIR, "processed", f"{ticker}_integrated_signals.csv"
            ),
            parse_dates=["Date"],
            index_col="Date",
        )

        env = TradingEnv(
            df,
            initial_balance=10000,
            transaction_cost_pct=0.001,
            slippage_pct=0.001,
            n_threads=4,
        )

        # Run a test episode
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
        print(f"Final balance: ${info['balance']:.2f}")
