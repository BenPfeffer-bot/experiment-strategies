import numpy as np
import sys
import os
import pandas as pd
from typing import Tuple, List, Dict
import joblib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import TICKERS, MARKET_DATA_DIR


@dataclass
class TrainingMetrics:
    total_reward: float
    avg_reward: float
    max_reward: float
    min_reward: float
    final_epsilon: float


class QLearningAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,  # 0=Hold, 1=Buy, 2=Sell
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,  # Increased batch size
        memory_size: int = 100000,  # Increased memory size
        n_threads: int = 4,  # Number of threads for parallel processing
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.n_threads = n_threads

        # Vectorized Q-table using NumPy array
        self.q_table = np.zeros((memory_size, action_size))
        self.state_dict = {}  # Maps state tuples to indices
        self.state_counter = 0

        # Prioritized experience replay
        self.memory = {
            "state": np.zeros((memory_size, state_size)),
            "action": np.zeros(memory_size, dtype=np.int32),
            "reward": np.zeros(memory_size),
            "next_state": np.zeros((memory_size, state_size)),
            "done": np.zeros(memory_size, dtype=np.bool_),
            "priority": np.ones(memory_size),  # Priority for experience replay
        }
        self.memory_size = memory_size
        self.memory_counter = 0

        # State normalization
        self.running_mean = np.zeros(state_size)
        self.running_var = np.ones(state_size)
        self.count = 0

        # State cache
        self.state_cache = {}
        self.cache_lock = threading.Lock()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=n_threads)

        # Training history
        self.training_history = []

    def _get_state_index(self, state_key: tuple) -> int:
        """Get or create index for state in Q-table"""
        if state_key not in self.state_dict:
            self.state_dict[state_key] = self.state_counter
            self.state_counter += 1
            if self.state_counter >= len(self.q_table):
                # Expand Q-table if needed
                self.q_table = np.vstack(
                    [self.q_table, np.zeros((self.memory_size, self.action_size))]
                )
        return self.state_dict[state_key]

    def _normalize_state(self, state: np.ndarray) -> tuple:
        """Normalize state with caching"""
        state_key = tuple(state)

        with self.cache_lock:
            if state_key in self.state_cache:
                return self.state_cache[state_key]

            state_array = np.asarray(state, dtype=np.float32)
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1e6, neginf=-1e6)

            # Update running statistics
            self.count += 1
            delta = state_array - self.running_mean
            self.running_mean += delta / self.count
            self.running_var = (
                self.running_var * (self.count - 1)
                + delta * (state_array - self.running_mean)
            ) / self.count

            # Normalize and clip
            running_std = np.sqrt(np.maximum(self.running_var / self.count, 1e-8))
            normalized_state = np.clip(
                (state_array - self.running_mean) / running_std, -10, 10
            )
            normalized_tuple = tuple(np.round(normalized_state, 6))

            # Cache the result
            self.state_cache[state_key] = normalized_tuple

            # Limit cache size
            if len(self.state_cache) > self.memory_size:
                self.state_cache.pop(next(iter(self.state_cache)))

            return normalized_tuple

    def _process_batch(self, batch_data: tuple) -> tuple:
        """Process a batch of experiences in parallel"""
        states, actions, rewards, next_states, dones = batch_data

        # Vectorized Q-value computation
        current_states = np.array([self._normalize_state(s) for s in states])
        next_states = np.array([self._normalize_state(s) for s in next_states])

        current_indices = np.array(
            [self._get_state_index(tuple(s)) for s in current_states]
        )
        next_indices = np.array([self._get_state_index(tuple(s)) for s in next_states])

        current_q = self.q_table[current_indices, actions]
        next_q_max = np.max(self.q_table[next_indices], axis=1)

        # Compute targets
        targets = rewards + self.gamma * next_q_max * (1 - dones)

        # Compute TD errors for prioritized replay
        td_errors = np.abs(targets - current_q)

        return current_indices, actions, targets, td_errors

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience with priority"""
        index = self.memory_counter % self.memory_size

        # Compute priority based on TD error
        current_q = self.get_q(state, action)
        next_q_max = max(self.get_q(next_state, a) for a in range(self.action_size))
        target = reward + self.gamma * next_q_max * (1 - done)
        priority = abs(target - current_q) + 1e-6  # Small constant for stability

        self.memory["state"][index] = state
        self.memory["action"][index] = action
        self.memory["reward"][index] = reward
        self.memory["next_state"][index] = next_state
        self.memory["done"][index] = done
        self.memory["priority"][index] = priority

        self.memory_counter += 1

    def get_q(self, state: np.ndarray, action: int) -> float:
        """Get Q-value using vectorized operations"""
        state_key = self._normalize_state(state)
        state_idx = self._get_state_index(state_key)
        return self.q_table[state_idx, action]

    def choose_action(self, state: np.ndarray) -> int:
        """Choose action with epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        state_key = self._normalize_state(state)
        state_idx = self._get_state_index(state_key)
        return int(np.argmax(self.q_table[state_idx]))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Learn from experience using prioritized replay and parallel processing"""
        self.remember(state, action, reward, next_state, done)

        if self.memory_counter >= self.batch_size:
            # Sample batch based on priorities
            probs = self.memory["priority"][
                : min(self.memory_counter, self.memory_size)
            ]
            probs = probs / np.sum(probs)

            batch_indices = np.random.choice(
                len(probs), size=self.batch_size, p=probs, replace=False
            )

            # Prepare batch data
            batch_data = (
                self.memory["state"][batch_indices],
                self.memory["action"][batch_indices],
                self.memory["reward"][batch_indices],
                self.memory["next_state"][batch_indices],
                self.memory["done"][batch_indices],
            )

            # Process batch in parallel
            current_indices, actions, targets, td_errors = self._process_batch(
                batch_data
            )

            # Update Q-values
            self.q_table[current_indices, actions] += self.alpha * (
                targets - self.q_table[current_indices, actions]
            )

            # Update priorities
            self.memory["priority"][batch_indices] = td_errors

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def train(self, env, episodes: int = 100) -> TrainingMetrics:
        """Train the agent with parallel experience collection"""
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                # Learn in parallel
                self.executor.submit(
                    self.learn, state, action, reward, next_state, done
                )

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            self.training_history.append(
                {
                    "episode": episode,
                    "total_reward": total_reward,
                    "epsilon": self.epsilon,
                }
            )

            if episode % 10 == 0:
                print(
                    f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}"
                )

        # Wait for all learning tasks to complete
        self.executor.shutdown(wait=True)

        return TrainingMetrics(
            total_reward=sum(episode_rewards),
            avg_reward=np.mean(episode_rewards),
            max_reward=max(episode_rewards),
            min_reward=min(episode_rewards),
            final_epsilon=self.epsilon,
        )

    def save(self, filepath: str):
        """Save the trained agent"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "q_table": self.q_table,
                "state_dict": self.state_dict,
                "state_counter": self.state_counter,
                "epsilon": self.epsilon,
                "training_history": self.training_history,
                "running_mean": self.running_mean,
                "running_var": self.running_var,
                "count": self.count,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load a trained agent"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved agent found at {filepath}")

        saved_data = joblib.load(filepath)
        self.q_table = saved_data["q_table"]
        self.state_dict = saved_data["state_dict"]
        self.state_counter = saved_data["state_counter"]
        self.epsilon = saved_data["epsilon"]
        self.training_history = saved_data["training_history"]
        self.running_mean = saved_data["running_mean"]
        self.running_var = saved_data["running_var"]
        self.count = saved_data["count"]


if __name__ == "__main__":
    from src.models.trading_env import TradingEnv

    # Test the agent on a sample environment
    for ticker in TICKERS[:1]:  # Test with first ticker
        try:
            # Load data
            df = pd.read_csv(
                os.path.join(
                    MARKET_DATA_DIR, "processed", f"{ticker}_integrated_signals.csv"
                ),
                parse_dates=["Date"],
                index_col="Date",
            )

            # Create environment
            env = TradingEnv(
                df,
                initial_balance=10000,
                transaction_cost_pct=0.001,
                slippage_pct=0.001,
            )

            # Initialize and train agent
            state_size = env.observation_space.shape[0]
            agent = QLearningAgent(state_size=state_size)

            print(f"\nTraining RL agent for {ticker}")
            metrics = agent.train(env, episodes=100)

            print("\nTraining Results:")
            print(f"Total Reward: {metrics.total_reward:.2f}")
            print(f"Average Reward: {metrics.avg_reward:.2f}")
            print(f"Max Reward: {metrics.max_reward:.2f}")
            print(f"Min Reward: {metrics.min_reward:.2f}")
            print(f"Final Epsilon: {metrics.final_epsilon:.3f}")

            # Save the trained agent
            save_path = os.path.join(
                MARKET_DATA_DIR, "models", f"{ticker}_rl_agent.joblib"
            )
            agent.save(save_path)
            print(f"\nAgent saved to {save_path}")

        except Exception as e:
            print(f"Error training RL agent for {ticker}: {str(e)}")
