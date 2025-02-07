import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
import joblib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import logging
from datetime import datetime

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


@dataclass
class OptimizationResult:
    best_params: Dict[str, float]
    best_score: float
    all_trials: List[Dict]
    optimization_history: List[float]
    parameter_importance: Dict[str, float]


class BayesianOptimizer:
    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float]],
        objective_function: Callable,
        n_initial_points: int = 10,
        n_iterations: int = 50,
        exploration_weight: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.param_space = param_space
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.random_state = random_state

        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )

        # Initialize optimization history
        self.X_samples = []
        self.y_samples = []
        self.all_trials = []

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for optimization process"""
        self.logger = logging.getLogger("BayesianOptimizer")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        log_dir = "logs/optimization"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            f"{log_dir}/optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range"""
        normalized = []
        for param_name, value in params.items():
            low, high = self.param_space[param_name]
            normalized.append((value - low) / (high - low))
        return np.array(normalized)

    def _denormalize_params(self, normalized: np.ndarray) -> Dict[str, float]:
        """Convert normalized parameters back to original scale"""
        params = {}
        for i, (param_name, (low, high)) in enumerate(self.param_space.items()):
            params[param_name] = normalized[i] * (high - low) + low
        return params

    def _expected_improvement(self, X: np.ndarray) -> float:
        """Calculate expected improvement at X"""
        mu, sigma = self.gp.predict(X.reshape(1, -1), return_std=True)

        if len(self.y_samples) == 0:
            return 0

        best_y = np.max(self.y_samples)

        with np.errstate(divide="warn"):
            imp = mu - best_y - self.exploration_weight
            Z = imp / sigma if sigma > 0 else 0
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _parallel_evaluate(self, params_list: List[Dict[str, float]]) -> List[float]:
        """Evaluate multiple parameter sets in parallel"""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(self.objective_function, params_list))
        return results

    def optimize(self) -> OptimizationResult:
        """Run Bayesian optimization"""
        # Initial random sampling
        self.logger.info("Starting initial random sampling...")
        initial_params = []
        for _ in range(self.n_initial_points):
            params = {}
            for param_name, (low, high) in self.param_space.items():
                params[param_name] = np.random.uniform(low, high)
            initial_params.append(params)

        # Evaluate initial points in parallel
        initial_scores = self._parallel_evaluate(initial_params)

        # Store results
        for params, score in zip(initial_params, initial_scores):
            normalized_params = self._normalize_params(params)
            self.X_samples.append(normalized_params)
            self.y_samples.append(score)
            self.all_trials.append({"params": params, "score": score})

        self.logger.info(
            f"Initial sampling complete. Best score: {max(initial_scores):.4f}"
        )

        # Main optimization loop
        for i in range(self.n_iterations):
            self.logger.info(f"Starting iteration {i + 1}/{self.n_iterations}")

            # Fit GP model
            self.gp.fit(np.array(self.X_samples), np.array(self.y_samples))

            # Find next point to evaluate
            def objective(x):
                return -self._expected_improvement(x)

            best_ei = float("inf")
            best_params = None

            # Multiple random starts for optimization
            for _ in range(5):
                x0 = np.random.rand(len(self.param_space))
                bounds = [(0, 1) for _ in range(len(self.param_space))]

                result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

                if result.fun < best_ei:
                    best_ei = result.fun
                    best_params = result.x

            # Evaluate new point
            params = self._denormalize_params(best_params)
            score = self.objective_function(params)

            # Store results
            self.X_samples.append(best_params)
            self.y_samples.append(score)
            self.all_trials.append({"params": params, "score": score})

            self.logger.info(f"Iteration {i + 1} complete. Score: {score:.4f}")

        # Calculate parameter importance
        param_importance = self._calculate_parameter_importance()

        # Find best result
        best_idx = np.argmax(self.y_samples)
        best_params = self._denormalize_params(self.X_samples[best_idx])
        best_score = self.y_samples[best_idx]

        self.logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        self.logger.info("Best parameters:")
        for param, value in best_params.items():
            self.logger.info(f"{param}: {value:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=self.all_trials,
            optimization_history=self.y_samples,
            parameter_importance=param_importance,
        )

    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance using GP length scales"""
        importance = {}
        length_scales = self.gp.kernel_.get_params()["k1__length_scale"]
        if not isinstance(length_scales, np.ndarray):
            length_scales = np.array([length_scales])

        total = np.sum(1 / length_scales)
        for i, (param_name, _) in enumerate(self.param_space.items()):
            importance[param_name] = (1 / length_scales[i]) / total

        return importance

    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history and parameter importance"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot optimization history
        ax1.plot(self.y_samples)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score")
        ax1.set_title("Optimization History")

        # Plot parameter importance
        importance = self._calculate_parameter_importance()
        sns.barplot(x=list(importance.values()), y=list(importance.keys()), ax=ax2)
        ax2.set_title("Parameter Importance")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_results(self, filepath: str):
        """Save optimization results"""
        results = {
            "param_space": self.param_space,
            "X_samples": self.X_samples,
            "y_samples": self.y_samples,
            "all_trials": self.all_trials,
            "gp_model": self.gp,
            "best_params": self._denormalize_params(
                self.X_samples[np.argmax(self.y_samples)]
            ),
            "best_score": max(self.y_samples),
        }
        joblib.dump(results, filepath)

    @classmethod
    def load_results(cls, filepath: str):
        """Load optimization results"""
        results = joblib.load(filepath)
        optimizer = cls(
            param_space=results["param_space"],
            objective_function=lambda x: 0,  # Dummy function
        )
        optimizer.X_samples = results["X_samples"]
        optimizer.y_samples = results["y_samples"]
        optimizer.all_trials = results["all_trials"]
        optimizer.gp = results["gp_model"]
        return optimizer
