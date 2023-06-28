import autograd.numpy as np
from autograd import hessian, jacobian

from .utils import timeit


def _cumprod(x):
    # collect products
    cumprods = []
    for i in range(x.size):
        # get next number / column / row
        current_num = x[i]

        # deal with first case
        if i == 0:
            cumprods.append(current_num)
        else:
            # get previous number
            prev_num = cumprods[i - 1]

            # compute next number / column / row
            next_num = prev_num * current_num
            cumprods.append(next_num)
    return np.array(cumprods)


class MOOAnalytical:
    def __init__(self):
        self._objective_jacobian = jacobian(self.objective)
        self._objective_hessian = hessian(self.objective)
        self._constraint_jacobian = jacobian(self.constraint) if hasattr(self, "constraint") else None
        self._constraint_hessian = hessian(self.constraint) if hasattr(self, "constraint") else None
        self.CPU_time: int = 0  # in nanoseconds

    @timeit
    def objective_jacobian(self, x):
        return self._objective_jacobian(x)

    @timeit
    def objective_hessian(self, x):
        return self._objective_hessian(x)

    @timeit
    def constraint_jacobian(self, x):
        return self._constraint_jacobian(x)

    @timeit
    def constraint_hessian(self, x):
        return self._constraint_hessian(x)


class _Eq1DTLZ(MOOAnalytical):
    def __init__(self, n_objectives: int = 3, n_decision_vars: int = None):
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
        y = np.array([self.objective(xx) for xx in x])
        return y


class Eq1DTLZ1(_Eq1DTLZ):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
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
    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class Eq1DTLZ2(_Eq1DTLZ):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])
        )

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class Eq1DTLZ3(_Eq1DTLZ):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        D = len(x)
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])
        )

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class Eq1DTLZ4(_Eq1DTLZ):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        x_ = x[0 : M - 1] ** 100
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x_ * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x_[::-1] * np.pi / 2)])
        )

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class Eq1IDTLZ1(Eq1DTLZ1):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        D = len(x)
        M = self.n_objectives
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return (1 + g) / 2 - 0.5 * (1 + g) * _cumprod(np.concatenate([[1], x[0 : M - 1]]))[
            ::-1
        ] * np.concatenate([[1], 1 - x[0 : M - 1][::-1]])


class Eq1IDTLZ2(Eq1DTLZ2):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (1 + g) / 2 - (
            (1 + g)
            * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[::-1]
            * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])
        )


class Eq1IDTLZ3(Eq1DTLZ3):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        D = len(x)
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return (1 + g) / 2 - (1 + g) * _cumprod(np.concatenate([[1], np.cos(x[0 : M - 1] * np.pi / 2)]))[
            ::-1
        ] * np.concatenate([[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)])


class Eq1IDTLZ4(Eq1DTLZ4):
    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        M = self.n_objectives
        x_ = x[0 : M - 1] ** 100
        g = np.sum((x[M - 1 :] - 0.5) ** 2)
        return (1 + g) / 2 - (1 + g) * _cumprod(np.concatenate([[1], np.cos(x_ * np.pi / 2)]))[
            ::-1
        ] * np.concatenate([[1], np.sin(x_[::-1] * np.pi / 2)])


class CONV4(MOOAnalytical):
    def __init__(self):
        self.n_objectives = 4
        self.n_decision_vars = 4
        self.lower_bounds = -10 * np.ones(self.n_decision_vars)
        self.upper_bounds = 10 * np.ones(self.n_decision_vars)
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        a = np.eye(self.n_decision_vars)
        deltaa = np.ones(self.n_decision_vars)
        fa4 = np.array([2, 2, 2, 0])
        fa1 = np.array([0, 2, 2, 2])
        deltay = fa4 - fa1

        if np.all(x < 0):
            z = x + deltaa
            y = np.array([np.sum((z - a[i]) ** 2) - 1.1 * deltay[i] for i in range(4)])
        else:
            y = np.array([np.sum((x - a[i]) ** 2) for i in range(4)])
        return y
