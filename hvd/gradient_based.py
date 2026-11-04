"""Multiobjective gradient descent with box constraints."""

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .base import State
from .utils import get_logger, set_bounds


def minimum_norm_weights(gradients: np.ndarray) -> np.ndarray:
    """Find simplex weights minimizing ||sum_i weight_i * gradient_i||².

    The negative weighted gradient is MGDA's common descent direction for any
    number of objectives. A zero combination indicates Pareto stationarity.
    See https://mgda.inria.fr/mgda.
    """
    gradients = np.asarray(gradients, dtype=float)
    if gradients.ndim != 2 or not gradients.size or not np.isfinite(gradients).all():
        raise ValueError("Expected a finite (n_objectives, n_variables) gradient matrix")
    n_obj = len(gradients)
    scale = np.linalg.norm(gradients, axis=1).max()
    if scale == 0:
        return np.full(n_obj, 1 / n_obj)
    normalized = gradients / scale  # A common scale preserves the minimizer.
    gram = normalized @ normalized.T
    result = minimize(
        lambda weights: 0.5 * weights @ gram @ weights,
        np.full(n_obj, 1 / n_obj),
        jac=lambda weights: gram @ weights,
        bounds=[(0.0, 1.0)] * n_obj,
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1,
            "jac": lambda weights: np.ones(n_obj),
        },
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Minimum-norm gradient calculation failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


class MGDA:
    """Multiple-gradient descent with projected Armijo backtracking.

    Each point follows the negative minimum-norm convex combination of its
    objective gradients. Projected Armijo backtracking accepts a step only when
    every objective satisfies its decrease condition. Intrinsic equality and
    inequality constraints are outside this method; xl/xu enforce box bounds.
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        func: Callable,
        jac: Callable,
        xl: float | list[float] | np.ndarray,
        xu: float | list[float] | np.ndarray,
        N: int = 5,
        X0: np.ndarray | None = None,
        max_iters: int | None = None,
        xtol: float = 1e-8,
        verbose: bool = True,
        metrics: dict | None = None,
        step_size: float = 1.0,
        armijo: float = 1e-4,
        max_backtracks: int = 40,
    ) -> None:
        self.dim, self.n_obj = n_var, n_obj
        self.xl, self.xu = set_bounds(xl, n_var), set_bounds(xu, n_var)
        if np.any(self.xl > self.xu):
            raise ValueError("Lower bounds exceed upper bounds")
        if step_size <= 0 or not 0 < armijo < 1 or max_backtracks < 1 or xtol < 0:
            raise ValueError("Invalid line-search settings or tolerance")
        if X0 is None:
            if not np.isfinite(self.xl).all() or not np.isfinite(self.xu).all():
                raise ValueError("Finite bounds are required to sample an initial population")
            X0 = np.random.uniform(self.xl, self.xu, (N, n_var))
        X0 = np.asarray(X0, dtype=float)
        if X0.ndim != 2 or X0.shape[1] != n_var or not len(X0) or not np.isfinite(X0).all():
            raise ValueError("Invalid initial population")
        if np.any(X0 < self.xl) or np.any(X0 > self.xu):
            raise ValueError("Initial population is outside the box bounds")
        self.N = len(X0)
        self.max_iters = self.N * 10 if max_iters is None else max_iters
        self.xtol, self.initial_step_size = xtol, step_size
        self.armijo, self.max_backtracks = armijo, max_backtracks
        self.state = State(n_var, 0, 0, func, jac)
        self.state.update(X0)
        self.iter_count, self.stop_dict = 0, {}
        self.verbose, self.metrics = verbose, metrics or {}
        self.history_X, self.history_Y, self.history_R_norm = [], [], []
        self.history_metrics = defaultdict(list)
        self.logger = get_logger(logger_id=self.__class__.__name__, console=verbose)

    def run(self) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        while not self.terminate():
            self.iteration()
            self.log()
        return self.state.primal, self.state.Y, self.stop_dict

    def _compute_step(self, state: State) -> np.ndarray:
        if state.J.shape != (state.N, self.n_obj, self.dim):
            raise ValueError("Objective Jacobian has an unexpected shape")
        self.weights = np.array([minimum_norm_weights(jacobian) for jacobian in state.J])
        return -np.einsum("ni,nij->nj", self.weights, state.J)

    def iteration(self) -> None:
        self.step = self._compute_step(self.state)
        self.step_size = np.zeros(self.N)
        population = self.state.primal.copy()
        for index, direction in enumerate(self.step):
            if np.linalg.norm(direction) <= self.xtol:
                continue
            alpha = self.initial_step_size
            for _ in range(self.max_backtracks):
                candidate = np.clip(self.state.primal[index] + alpha * direction, self.xl, self.xu)
                displacement = candidate - self.state.primal[index]
                slopes = self.state.J[index] @ displacement
                if np.linalg.norm(displacement) <= self.xtol:
                    break
                if np.all(slopes < 0):
                    values = State.evaluate(self.state.func, candidate)
                    if np.isfinite(values).all() and np.all(
                        values <= self.state.Y[index] + self.armijo * slopes
                    ):
                        population[index] = candidate
                        self.step_size[index] = alpha
                        break
                alpha *= 0.5
        displacement = np.linalg.norm(population - self.state.primal, axis=1)
        if np.any(displacement > 0):
            self.state.update(population)
        self.iter_count += 1
        if displacement.max() <= self.xtol:
            self.stop_dict["xtol"] = float(displacement.max())

    def log(self) -> None:
        self.history_X.append(self.state.primal.copy())
        self.history_Y.append(self.state.Y.copy())
        self.history_R_norm.append(float(np.median(np.linalg.norm(self.step, axis=1))))
        for name, metric in self.metrics.items():
            self.history_metrics[name].append(metric.compute(Y=self.state.Y))
        if self.verbose:
            self.logger.info(f"iteration {self.iter_count}: gradient norm {self.history_R_norm[-1]:.6g}")

    def terminate(self) -> bool:
        if self.iter_count >= self.max_iters:
            self.stop_dict["iter_count"] = self.iter_count
        return bool(self.stop_dict)
