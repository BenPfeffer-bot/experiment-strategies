import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.models.trading_env import TradingEnv
from src.utils.config import MARKET_DATA_DIR, TICKERS


def train_rl_agent(df_path, total_timesteps=100000):
    # Load and validate data
    try:
        df = pd.read_csv(df_path, parse_dates=["Date"], index_col="Date")
        required_columns = [
            "price",
            "avg_rsi",
            "rsi_signal",
            "fib_dir",
            "macd_macd",
            "macd_signal",
            "ma",
            "variance",
            "kelter_upper",
            "kelter_lower",
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove any NaN values
        df = df.dropna()
        if len(df) == 0:
            raise ValueError("No valid data points after removing NaN values")

        env = TradingEnv(df)

        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./models/checkpoints/",
            name_prefix="trading_model",
        )

        # Initialize model with custom network architecture
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
        )

        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

        # Save the final model
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join(
            "models", f"ppo_trading_{os.path.basename(df_path)}.zip"
        )
        model.save(model_path)
        print(f"RL agent trained and saved to {model_path}!")

    except Exception as e:
        print(f"Error training RL agent: {str(e)}")
        return None

    return model


if __name__ == "__main__":
    for ticker in TICKERS:
        print(f"\nTraining model for {ticker}")
        sample_data_path = os.path.join(
            MARKET_DATA_DIR, "processed", f"{ticker}_integrated_signals.csv"
        )
        model = train_rl_agent(sample_data_path)
