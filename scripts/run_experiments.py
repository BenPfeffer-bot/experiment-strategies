import os
import sys
import json
import logging
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration variables (Assuming TICKERS is defined in src/utils/config.py)
from src.utils.config import TICKERS, MARKET_DATA_DIR

# Import our modules
from src.strategies.meta_strategy import main as generate_meta_signals
from src.models.ml_model import MLModel
from src.models.reinforcement_learning import QLearningAgent
from src.execution.executor import SimulatedBroker
from src.risks.risk_mgmt import RiskManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


class ExperimentManager:
    def __init__(
        self, experiment_name: str, base_dir: str = "experiments/base_strategy"
    ):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize MLflow with SQLite backend
        db_path = self.base_dir / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Initialize risk manager
        self.risk_manager = RiskManager()

    def setup_logging(self):
        """Configure logging"""
        log_file = (
            self.base_dir
            / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(self.experiment_name)

    def run_meta_strategy(self):
        """Run meta strategy signal generation"""
        self.logger.info("Generating meta strategy signals...")
        with mlflow.start_run(nested=True, run_name="meta_strategy"):
            try:
                generate_meta_signals()
                mlflow.log_param("strategy_type", "meta")
                self.logger.info("Meta strategy signals generated successfully")
            except Exception as e:
                self.logger.error(f"Error in meta strategy: {str(e)}")
                raise

    def run_ml_experiments(self) -> Dict:
        """Run machine learning experiments"""
        ml_results = {}
        self.logger.info("Running ML experiments...")

        with mlflow.start_run(nested=True, run_name="ml_training"):
            for ticker in TICKERS:
                try:
                    self.logger.info(f"Training ML model for {ticker}")

                    # Initialize ML model
                    model = MLModel()

                    # Load data for prediction
                    df = pd.read_csv(
                        os.path.join(
                            MARKET_DATA_DIR,
                            "processed",
                            f"{ticker}_integrated_signals.csv",
                        ),
                        parse_dates=["Date"],
                        index_col="Date",
                    )

                    # Train model
                    model.train_ml_model(ticker)

                    # Make predictions using the trained model
                    predictions = model.predict(df)

                    # Calculate returns based on signal strength
                    df["returns"] = df["price"].pct_change()
                    strategy_returns = np.zeros_like(predictions[:-1], dtype=float)

                    # Strong buy signals (2) get full long exposure
                    strategy_returns[predictions[:-1] == 2] = df["returns"].values[1:][
                        predictions[:-1] == 2
                    ]

                    # Strong sell signals (0) get full short exposure
                    strategy_returns[predictions[:-1] == 0] = -df["returns"].values[1:][
                        predictions[:-1] == 0
                    ]

                    # Hold signals (1) get no exposure
                    strategy_returns[predictions[:-1] == 1] = 0

                    # Calculate risk metrics
                    risk_metrics = self.risk_manager.calculate_risk_metrics(
                        pd.Series(strategy_returns)
                    )

                    # Get feature importance from both models in the ensemble
                    feature_importance = {}
                    for name, model_est in model.model.named_estimators_.items():
                        if hasattr(model_est, "feature_importances_"):
                            feature_importance[name] = dict(
                                zip(model.feature_cols, model_est.feature_importances_)
                            )

                    # Log metrics to MLflow
                    with mlflow.start_run(nested=True, run_name=f"ml_{ticker}"):
                        mlflow.log_params(
                            {
                                "ticker": ticker,
                                "n_features": len(model.feature_cols),
                                "features": ", ".join(model.feature_cols),
                            }
                        )
                        mlflow.log_metrics(
                            {
                                "sharpe_ratio": risk_metrics.sharpe_ratio,
                                "max_drawdown": risk_metrics.max_drawdown,
                                "volatility": risk_metrics.volatility,
                                "sortino_ratio": risk_metrics.sortino_ratio,
                                "calmar_ratio": risk_metrics.calmar_ratio,
                                "total_return": (1 + strategy_returns).prod() - 1,
                            }
                        )

                        # Log feature importance plots
                        for model_name, importances in feature_importance.items():
                            top_features = dict(
                                sorted(
                                    importances.items(),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )[:10]
                            )

                            fig = go.Figure(
                                go.Bar(
                                    x=list(top_features.keys()),
                                    y=list(top_features.values()),
                                    text=[f"{v:.2%}" for v in top_features.values()],
                                    textposition="auto",
                                )
                            )

                            fig.update_layout(
                                title=f"Top 10 Feature Importance - {model_name}",
                                xaxis_title="Features",
                                yaxis_title="Importance",
                                showlegend=False,
                            )

                            fig.write_html(
                                self.base_dir
                                / f"{ticker}_{model_name}_feature_importance.html"
                            )

                    ml_results[ticker] = {
                        "risk_metrics": risk_metrics,
                        "feature_importance": feature_importance,
                    }

                    # Generate and save performance visualization
                    self.plot_performance(
                        df["price"], strategy_returns, predictions, ticker, "ml"
                    )

                except Exception as e:
                    self.logger.error(f"ML training failed for {ticker}: {str(e)}")
                    ml_results[ticker] = {"status": "failed", "error": str(e)}

        return ml_results

    def run_rl_experiments(self) -> Dict:
        """Run reinforcement learning experiments"""
        rl_results = {}
        self.logger.info("Running RL experiments...")

        with mlflow.start_run(nested=True, run_name="rl_training"):
            for ticker in TICKERS:
                try:
                    self.logger.info(f"Training RL agent for {ticker}")

                    # Load data
                    df = pd.read_csv(
                        os.path.join(
                            MARKET_DATA_DIR,
                            "processed",
                            f"{ticker}_integrated_signals.csv",
                        ),
                        parse_dates=["Date"],
                        index_col="Date",
                    )

                    # Calculate returns
                    df["returns"] = df["price"].pct_change()
                    df = df.fillna(0)  # Fill NaN values from pct_change

                    # Initialize and train RL agent
                    state_size = len(df.columns)
                    agent = QLearningAgent(state_size=state_size, action_size=3)

                    # Run episodes
                    episode_rewards = []
                    for episode in range(100):  # Number of episodes
                        state = df.iloc[0].values
                        total_reward = 0

                        for t in range(len(df) - 1):
                            action = agent.choose_action(state)
                            next_state = df.iloc[t + 1].values
                            reward = (
                                df.iloc[t + 1]["returns"]
                                if "returns" in df
                                else df.iloc[t + 1]["price"] / df.iloc[t]["price"] - 1
                            )

                            if action == 1:  # Long
                                total_reward += reward
                            elif action == 2:  # Short
                                total_reward -= reward

                            agent.learn(
                                state, action, reward, next_state, t == len(df) - 2
                            )
                            state = next_state

                        episode_rewards.append(total_reward)

                        # Log episode metrics
                        mlflow.log_metric(
                            f"{ticker}_episode_reward", total_reward, step=episode
                        )

                    # Calculate final performance metrics
                    final_reward = np.mean(
                        episode_rewards[-10:]
                    )  # Average of last 10 episodes
                    rl_results[ticker] = {
                        "final_reward": final_reward,
                        "episode_rewards": episode_rewards,
                    }

                    # Plot learning curves
                    self.plot_learning_curve(episode_rewards, ticker)

                except Exception as e:
                    self.logger.error(f"RL training failed for {ticker}: {str(e)}")
                    rl_results[ticker] = {"status": "failed", "error": str(e)}

        return rl_results

    def plot_performance(
        self,
        prices: pd.Series,
        returns: np.ndarray,
        signals: np.ndarray,
        ticker: str,
        model_type: str,
    ):
        """Generate enhanced performance visualization"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Price & Signals",
                "Signal Strength",
                "Cumulative Returns",
                "Drawdown",
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2],
        )

        # 1. Price chart with signals
        dates = prices.index

        # Main price line
        fig.add_trace(
            go.Scatter(
                x=dates, y=prices, name="Price", line=dict(color="blue", width=1)
            ),
            row=1,
            col=1,
        )

        # Buy signals
        buy_dates = dates[signals == 2]
        buy_prices = prices[signals == 2]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up", size=10, color="green", line=dict(width=1)
                ),
            ),
            row=1,
            col=1,
        )

        # Sell signals
        sell_dates = dates[signals == 0]
        sell_prices = prices[signals == 0]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down", size=10, color="red", line=dict(width=1)
                ),
            ),
            row=1,
            col=1,
        )

        # 2. Signal strength indicator
        if model_type == "ml":
            signal_strength = pd.Series(signals, index=dates)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=signal_strength,
                    name="Signal Strength",
                    line=dict(color="purple"),
                ),
                row=2,
                col=1,
            )

        # 3. Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=dates[1:],
                y=cumulative_returns,
                name="Cumulative Returns",
                line=dict(color="green"),
            ),
            row=3,
            col=1,
        )

        # 4. Drawdown
        rolling_max = pd.Series(cumulative_returns).expanding().max()
        drawdown = pd.Series(cumulative_returns) / rolling_max - 1
        fig.add_trace(
            go.Scatter(
                x=dates[1:],
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red"),
            ),
            row=4,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{model_type.upper()} Model Performance - {ticker}",
                x=0.5,
                xanchor="center",
            ),
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Update axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Signal", row=2, col=1)
        fig.update_yaxes(title_text="Returns", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown", row=4, col=1)

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True, row=4, col=1)

        # Save plot
        fig.write_html(self.base_dir / f"{ticker}_{model_type}_performance.html")

    def plot_learning_curve(self, rewards: List[float], ticker: str):
        """Plot RL learning curve"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rewards, name="Episode Reward"))

        fig.update_layout(
            title=f"RL Learning Curve - {ticker}",
            xaxis_title="Episode",
            yaxis_title="Total Reward",
        )

        fig.write_html(self.base_dir / f"{ticker}_rl_learning.html")

    def save_results(self, results: Dict):
        """Save experiment results"""
        output_file = (
            self.base_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4, default=str)
        self.logger.info(f"Results saved to {output_file}")


def main():
    # Initialize experiment manager
    experiment_manager = ExperimentManager("trading_experiment")

    try:
        # Step 1: Generate integrated signals
        experiment_manager.run_meta_strategy()

        # Step 2: Run ML experiments
        ml_results = experiment_manager.run_ml_experiments()

        # Step 3: Run RL experiments
        # rl_results = experiment_manager.run_rl_experiments()

        # Combine and save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "ml_results": ml_results,
            # "rl_results": rl_results,
        }

        experiment_manager.save_results(results)

    except Exception as e:
        experiment_manager.logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
