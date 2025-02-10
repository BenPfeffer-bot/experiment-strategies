import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any
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
        kernel_params: Optional[Dict[str, float]] = None,
        gp_params: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
        strategy: Optional[object] = None,
    ):
        self.param_space = param_space
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.random_state = random_state
        self.data = data
        self.strategy = strategy

        # Initialize Gaussian Process with more stable kernel configuration
        if gp_params and "kernel" in gp_params:
            kernel = gp_params["kernel"]
        else:
            kernel = Matern(
                length_scale=[1.0] * len(param_space),
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )
            if kernel_params:
                kernel.set_params(**kernel_params)

        # Set up GP parameters
        gp_config = {
            "kernel": kernel,
            "alpha": 1e-6,
            "normalize_y": True,
            "n_restarts_optimizer": 3,
            "random_state": random_state,
        }

        # Update with user-provided GP parameters
        if gp_params:
            gp_config.update({k: v for k, v in gp_params.items() if k != "kernel"})

        self.gp = GaussianProcessRegressor(**gp_config)

        # Initialize optimization history
        self.X_samples = []
        self.y_samples = []
        self.all_trials = []

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for optimization process"""
        self.logger = logging.getLogger("BayesianOptimizer")

        # Only add handlers if they don't exist
        if not self.logger.handlers:
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

            # Prevent propagation to root logger to avoid duplicate messages
            self.logger.propagate = False

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

    def _evaluate_single(self, params):
        """Wrapper function for evaluating a single parameter set"""
        return self.objective_function(params, self.data, self.strategy)

    def _parallel_evaluate(self, params_list: List[Dict[str, float]]) -> List[float]:
        """Evaluate multiple parameter sets in parallel with improved error handling"""
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for params in params_list:
                    futures.append(executor.submit(self._evaluate_single, params))

                # Collect results with timeout and error handling
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        if np.isfinite(result):
                            results.append(result)
                        else:
                            results.append(float("inf"))
                    except Exception as e:
                        self.logger.warning(f"Evaluation failed: {str(e)}")
                        results.append(float("inf"))

                return results
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {str(e)}")
            # Fallback to sequential evaluation
            return [self._evaluate_single(params) for params in params_list]

    def optimize(self) -> OptimizationResult:
        """Run Bayesian optimization with improved robustness"""
        try:
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

            # Store valid results
            for params, score in zip(initial_params, initial_scores):
                if np.isfinite(score):
                    normalized_params = self._normalize_params(params)
                    self.X_samples.append(normalized_params)
                    self.y_samples.append(score)
                    self.all_trials.append({"params": params, "score": score})

            if not self.X_samples:
                self.logger.error("No valid initial samples found")
                return None

            self.logger.info(
                f"Initial sampling complete. Best score: {min(self.y_samples):.4f}"
            )

            # Main optimization loop
            for i in range(self.n_iterations):
                self.logger.info(f"Starting iteration {i + 1}/{self.n_iterations}")

                try:
                    # Fit GP model with error handling
                    X = np.array(self.X_samples)
                    y = np.array(self.y_samples)

                    # Normalize targets
                    y_mean = np.mean(y)
                    y_std = np.std(y) if np.std(y) > 0 else 1.0
                    y_normalized = (y - y_mean) / y_std

                    self.gp.fit(X, y_normalized)

                    # Find next point to evaluate
                    def objective(x):
                        return -self._expected_improvement(x)

                    best_ei = float("inf")
                    best_params = None

                    # Multiple random starts for optimization
                    for _ in range(3):  # Reduced from 5 to 3 for speed
                        x0 = np.random.rand(len(self.param_space))
                        bounds = [(0, 1) for _ in range(len(self.param_space))]

                        result = minimize(
                            objective,
                            x0,
                            bounds=bounds,
                            method="L-BFGS-B",
                            options={"maxiter": 50},
                        )

                        if result.fun < best_ei:
                            best_ei = result.fun
                            best_params = result.x

                    # Evaluate new point
                    params = self._denormalize_params(best_params)
                    score = self.objective_function(params, self.data, self.strategy)

                    if np.isfinite(score):
                        # Store results
                        self.X_samples.append(best_params)
                        self.y_samples.append(score)
                        self.all_trials.append({"params": params, "score": score})
                        self.logger.info(
                            f"Iteration {i + 1} complete. Score: {score:.4f}"
                        )
                    else:
                        self.logger.warning(f"Iteration {i + 1} produced invalid score")

                except Exception as e:
                    self.logger.error(f"Error in iteration {i + 1}: {str(e)}")
                    continue

            # Find best result
            if not self.y_samples:
                return None

            best_idx = np.argmin(self.y_samples)
            best_params = self._denormalize_params(self.X_samples[best_idx])
            best_score = self.y_samples[best_idx]

            # Calculate parameter importance with error handling
            try:
                param_importance = self._calculate_parameter_importance()
            except Exception as e:
                self.logger.warning(
                    f"Could not calculate parameter importance: {str(e)}"
                )
                param_importance = {
                    name: 1.0 / len(self.param_space) for name in self.param_space
                }

            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                all_trials=self.all_trials,
                optimization_history=self.y_samples,
                parameter_importance=param_importance,
            )

        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return None

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
                self.X_samples[np.argmin(self.y_samples)]
            ),
            "best_score": min(self.y_samples),
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
