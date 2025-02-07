import numpy as np
from scipy.optimize import minimize
import logging
from functools import partial

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    def __init__(
        self, objective_function, param_space, n_initial_points=10, n_iterations=20
    ):
        """
        Initialize the Bayesian Optimizer.

        Args:
            objective_function: Function to minimize
            param_space: List of tuples (min, max) or dict of param_name: (min, max)
            n_initial_points: Number of random points to evaluate before optimization
            n_iterations: Number of optimization iterations
        """
        self.objective_function = objective_function
        # Convert dict param_space to list if necessary
        if isinstance(param_space, dict):
            self.param_names = list(param_space.keys())
            self.param_space = [param_space[name] for name in self.param_names]
        else:
            self.param_names = None
            self.param_space = param_space
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_value = np.inf
        self.history = []

    def _random_sample(self):
        """Generate a random point within the parameter space."""
        point = [np.random.uniform(low, high) for low, high in self.param_space]
        if self.param_names:
            return dict(zip(self.param_names, point))
        return point

    def _evaluate_point(self, point):
        """Evaluate the objective function at a given point."""
        try:
            # Convert point to dict if using named parameters
            if self.param_names and not isinstance(point, dict):
                point = dict(zip(self.param_names, point))

            value = self.objective_function(point)
            self.history.append((point, value))

            if value < self.best_value:
                self.best_value = value
                self.best_params = point
            return value
        except Exception as e:
            logger.error(f"Error evaluating point {point}: {str(e)}")
            return np.inf

    def optimize(self):
        """Run the optimization process."""
        logger.info("Starting initial random sampling...")

        # Initial random sampling
        for _ in range(self.n_initial_points):
            point = self._random_sample()
            self._evaluate_point(point)

        # Main optimization loop
        for i in range(self.n_iterations):
            logger.info(f"Starting iteration {i + 1}/{self.n_iterations}")

            try:
                # Create a partial function for scipy's minimize
                objective = partial(self._evaluate_point)

                # Use scipy's minimize for local optimization
                if self.param_names:
                    # Convert best params back to list for minimize
                    x0 = (
                        [self.best_params[name] for name in self.param_names]
                        if self.best_params
                        else self._random_sample()
                    )
                else:
                    x0 = self.best_params if self.best_params else self._random_sample()

                result = minimize(
                    objective,
                    x0=x0 if isinstance(x0, list) else list(x0.values()),
                    bounds=self.param_space,
                    method="L-BFGS-B",
                )

                if result.success:
                    point = (
                        dict(zip(self.param_names, result.x))
                        if self.param_names
                        else result.x
                    )
                    value = self._evaluate_point(point)
                    if value < self.best_value:
                        self.best_value = value
                        self.best_params = point

            except Exception as e:
                logger.error(f"Optimization iteration failed: {str(e)}")
                continue

        return self.best_params, self.best_value
