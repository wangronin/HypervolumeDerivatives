import autograd.numpy as np

from ..utils import timeit
from .base import ConstrainedMOOAnalytical, MOOAnalytical, _cumprod


class _DTLZ(MOOAnalytical):
    def __init__(self, n_objectives: int = 3, n_decision_vars: int = 11):
        self.n_objectives = n_objectives
        # the default decision space of 11-dimensional
        self.n_decision_vars = n_decision_vars if n_decision_vars is not None else self.n_objectives + 8
        self.lower_bounds = np.zeros(self.n_decision_vars)
        self.upper_bounds = np.ones(self.n_decision_vars)
        super().__init__()

    def get_pareto_set(self, N: int = 1000, kind="uniform") -> np.ndarray:
        M = self.n_objectives
        theta = (
            np.sort(np.random.rand(N) * 2 * np.pi)
            if kind == "uniform"
            else np.r_[np.linspace(0, 2 * np.pi, N - 1), 0]
        )
        x = np.c_[np.cos(theta), np.sin(theta)] * 0.4
        return np.c_[x + 0.5, np.tile(0.5, (N, self.n_decision_vars - M + 1))]

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        x = self.get_pareto_set(N)
        y = np.array([self._objective(xx) for xx in x])
        return y


class DTLZ1(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        D = len(x)
        M = self.n_objectives
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return (
            0.5
            * (1 + g)
            * _cumprod(np.concatenate([[1], x[0 : M - 1]]))[::-1]
            * np.concatenate([[1], 1 - x[0 : M - 1][::-1]])
        )

    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ1(DTLZ1, ConstrainedMOOAnalytical):
    n_eq_constr = 1

    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class DTLZ2(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])
        )

    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ2(DTLZ2, ConstrainedMOOAnalytical):
    n_eq_constr = 1

    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class DTLZ3(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        D = len(x)
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])
        )

    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ3(DTLZ3, ConstrainedMOOAnalytical):
    n_eq_constr = 1

    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class DTLZ4(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        x_ = x[0 : M - 1] ** 100
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x_ * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x_[::-1] * np.pi / 2)])
        )

    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ4(DTLZ4, ConstrainedMOOAnalytical):
    @timeit
    def eq_constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class Eq1IDTLZ1(Eq1DTLZ1):
    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        D = len(x)
        M = self.n_objectives
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return (1 + g) / 2 - 0.5 * (1 + g) * _cumprod(np.concatenate([[1], x[0 : M - 1]]))[
            ::-1
        ] * np.concatenate([[1], 1 - x[0 : M - 1][::-1]])


class Eq1IDTLZ2(Eq1DTLZ2):
    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (1 + g) / 2 - (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])
        )


class Eq1IDTLZ3(Eq1DTLZ3):
    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        D = len(x)
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return (1 + g) / 2 - (1 + g) * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[
            ::-1
        ] * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])


class Eq1IDTLZ4(Eq1DTLZ4):
    @timeit
    def _objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        x_ = x[0 : M - 1] ** 100
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (1 + g) / 2 - (1 + g) * _cumprod(np.concatenate([[1], np.cos(x_ * np.pi / 2)]))[
            ::-1
        ] * np.concatenate([[1], np.sin(x_[::-1] * np.pi / 2)])
