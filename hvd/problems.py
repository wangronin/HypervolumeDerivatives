import autograd.numpy as np
import numpy as _np
from autograd import hessian, jacobian
from autograd.numpy import (abs, arange, cos, exp, mean, pi, prod, sign, sin,
                            sqrt, sum, tile)

from .utils import timeit

__author__ = ["Hao Wang"]


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
        self.CPU_time: int = 0  # in nanoseconds

    @timeit
    def objective_jacobian(self, x):
        return self._objective_jacobian(x)

    @timeit
    def objective_hessian(self, x):
        return self._objective_hessian(x)


class ConstrainedMOOAnalytical(MOOAnalytical):
    n_eq_constr = 0
    n_ieq_constr = 0

    def __init__(self):
        super().__init__()
        self._constraint_jacobian = jacobian(self.constraint) if hasattr(self, "constraint") else None
        self._constraint_hessian = hessian(self.constraint) if hasattr(self, "constraint") else None

    # TODO: `constraint` -> `eq_constraint`; add one more `ieq_constraint`
    @timeit
    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._constraint_jacobian(x)

    @timeit
    def constraint_hessian(self, x: np.ndarray) -> np.ndarray:
        return self._constraint_hessian(x)

    def constraint(self, x: np.ndarray) -> np.ndarray:
        pass


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
        y = np.array([self.objective(xx) for xx in x])
        return y


class DTLZ1(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

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
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ1(DTLZ1, ConstrainedMOOAnalytical):
    n_eq_constr = 1

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class DTLZ2(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

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
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ2(DTLZ2, ConstrainedMOOAnalytical):
    n_eq_constr = 1

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class DTLZ3(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

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
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ3(DTLZ3, ConstrainedMOOAnalytical):
    n_eq_constr = 1

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4


class DTLZ4(_DTLZ, ConstrainedMOOAnalytical):
    n_eq_constr = 22

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
        return np.vstack([self.lower_bounds - x, x - self.upper_bounds])


class Eq1DTLZ4(DTLZ4, ConstrainedMOOAnalytical):
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


class CONV3(MOOAnalytical):
    def __init__(self):
        self.n_objectives = 3
        self.n_decision_vars = 3
        self.lower_bounds = -3 * np.ones(self.n_decision_vars)
        self.upper_bounds = 3 * np.ones(self.n_decision_vars)
        self.a1 = -1 * np.ones(self.n_decision_vars)
        self.a2 = np.ones(self.n_decision_vars)
        self.a3 = np.r_[-1 * np.ones(self.n_decision_vars - 1), 1]
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        func = lambda x, a: np.sum((x - a) ** 2)
        return np.array([func(x, self.a1), func(x, self.a2), func(x, self.a3)])
        # return np.vstack([func(x, self.a1), func(x, self.a2), func(x, self.a3)]).T

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        w = np.random.rand(N, 3)
        X = w @ np.vstack([self.a1, self.a2, self.a3])
        return np.array([self.objective(x) for x in X])


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


class UF7(MOOAnalytical):
    def __init__(self, n_decision_vars: int = 30) -> None:
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, np.zeros(self.n_decision_vars - 1) - 1]
        self.upper_bounds = np.ones(self.n_decision_vars)
        self.encoding = np.ones(self.n_decision_vars)
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        N = x.shape[0]
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        y = x - np.sin(6 * np.pi * np.tile(x[:, [0]], (1, D)) + np.tile(np.arange(D) + 1, (N, 1)) * np.pi / D)
        return np.hstack(
            [
                x[:, 0] ** 0.2 + 2 * np.mean(y[:, J1] ** 2, 1),
                1 - x[:, 0] ** 0.2 + 2 * np.mean(y[:, J2] ** 2, 1),
            ]
        ).T

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        return np.c_[f, 1 - f]


class UF8(MOOAnalytical):
    def __init__(self, n_decision_vars: int = 30) -> None:
        self.n_objectives = 3
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, 0, np.zeros(self.n_decision_vars - 2) - 2]
        self.upper_bounds = np.r_[1, 1, np.zeros(self.n_decision_vars - 2) + 2]
        self.encoding = np.ones(self.n_decision_vars)
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        N = x.shape[0]
        D = self.n_decision_vars
        J1 = arange(4, D + 1, 3) - 1
        J2 = arange(5, D + 1, 3) - 1
        J3 = arange(3, D + 1, 3) - 1
        y = x - 2 * np.tile(x[:, [1]], (1, D)) * np.sin(
            2 * np.pi * np.tile(x[:, [0]], (1, D)) + np.tile(arange(D) + 1, (N, 1)) * np.pi / D
        )
        return np.hstack(
            [
                np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, J1] ** 2, 1),
                np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + 2 * np.mean(y[:, J2] ** 2, 1),
                np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(y[:, J3] ** 2, 1),
            ]
        ).T


# TODO: rename `constraint` -> `ieq` and `eq`
class CF1(ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.zeros(self.n_decision_vars)
        self.upper_bounds = np.ones(self.n_decision_vars)
        self.n_ieq_constr = 1

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.clip(x, self.lower_bounds, self.upper_bounds)
        N = x.shape[0]
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        term1 = np.tile(x[:, [0]], (1, len(J1))) ** (0.5 * (1 + 3 * (np.tile(J1 + 1, (N, 1)) - 2) / (D - 2)))
        term2 = np.tile(x[:, [0]], (1, len(J2))) ** (0.5 * (1 + 3 * (np.tile(J2 + 1, (N, 1)) - 2) / (D - 2)))
        F1 = x[:, 0] + 2 * np.mean((x[:, J1] - term1) ** 2, 1)
        F2 = 1 - x[:, 0] + 2 * np.mean((x[:, J2] - term2) ** 2, 1)
        return np.hstack([F1, F2])

    @timeit
    def constraint(self, x: np.ndarray) -> np.ndarray:
        # TODO: this function is calling the objective. Figure out a more efficient impplementation
        y = np.atleast_2d(self.objective(x))
        return 1 - y[:, 0] - y[:, 1] + np.abs(np.sin(10 * np.pi * (y[:, 0] - y[:, 1] + 1)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        return np.c_[f, 1 - f]


class CF2(ConstrainedMOOAnalytical):
    """Constrained Problem 2 in CEC 09 benchmark

    Reference:
        Zhang, Qingfu, Aimin Zhou, Shizheng Zhao, Ponnuthurai Nagaratnam Suganthan, Wudong Liu,
        and Santosh Tiwari. "Multiobjective optimization test instances for the CEC 2009 special
        session and competition." (2008): 1-30.
    """

    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, np.zeros(self.n_decision_vars - 1) - 1]
        self.upper_bounds = np.ones(self.n_decision_vars)
        self.n_ieq_constr = 1

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.clip(x, self.lower_bounds, self.upper_bounds)
        N = x.shape[0]
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        term1 = (x[:, J1] - sin(6 * pi * tile(x[:, [0]], (1, len(J1))) + tile(J1 + 1, (N, 1)) * pi / D)) ** 2
        term2 = (x[:, J2] - cos(6 * pi * tile(x[:, [0]], (1, len(J2))) + tile(J2 + 1, (N, 1)) * pi / D)) ** 2
        return np.hstack(
            [
                x[:, 0] + 2 * np.mean(term1, 1),
                1 - np.sqrt(x[:, 0]) + 2 * np.mean(term2, 1),
            ]
        )

    @timeit
    def constraint(self, x: np.ndarray) -> np.ndarray:
        # TODO: this function is calling the objective. Figure out a more efficient impplementation
        y = np.atleast_2d(self.objective(x))
        t = y[:, 1] + np.sqrt(y[:, 0]) - np.sin(2 * np.pi * (np.sqrt(y[:, 0]) - y[:, 1] + 1)) - 1
        return -t / (1 + np.exp(4 * abs(t)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        idx = (1 / 16 <= f) & (f <= 4 / 16) | (9 / 16 <= f) & (f <= 1)
        f = f[idx]
        return np.c_[f, 1 - np.sqrt(f)]


class CF3(ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, np.zeros(self.n_decision_vars - 1) - 2]
        self.upper_bounds = np.r_[1, np.zeros(self.n_decision_vars - 1) + 2]
        self.n_ieq_constr = 1

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.clip(x, self.lower_bounds, self.upper_bounds)
        N = x.shape[0]
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        Y = x - sin(6 * pi * tile(x[:, 0], (1, D)) + tile(np.arange(1, D + 1), (N, 1)) * pi / D)
        term1 = prod(cos(20 * Y[:, J1] * pi / sqrt(tile(J1 + 1, (N, 1)))), 1)
        term2 = prod(cos(20 * Y[:, J2] * pi / sqrt(tile(J2 + 1, (N, 1)))), 1)
        F1 = x[:, 0] + 2 / len(J1) * (4 * sum(Y[:, J1] ** 2, 1) - 2 * term1 + 2)
        F2 = 1 - x[:, 0] ** 2 + 2 / len(J2) * (4 * sum(Y[:, J2] ** 2, 1) - 2 * term2 + 2)
        return np.hstack([F1, F2])

    @timeit
    def constraint(self, x: np.ndarray) -> np.ndarray:
        # TODO: this function is calling the objective. Figure out a more efficient impplementation
        y = np.atleast_2d(self.objective(x))
        return 1 - y[:, 1] - y[:, 0] ** 2 + sin(2 * pi * (y[:, 0] ** 2 - y[:, 1] + 1))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        idx = (0 < f) & (f < 1 / 2) | (sqrt(1 / 2) < f) & (f < sqrt(3 / 4))
        f = f[~idx]
        return np.c_[f, 1 - f**2]


class CF4(ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, np.zeros(self.n_decision_vars - 1) - 2]
        self.upper_bounds = np.r_[1, np.zeros(self.n_decision_vars - 1) + 2]
        self.n_ieq_constr = 1

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X = np.clip(X, self.lower_bounds, self.upper_bounds)
        N = X.shape[0]
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        Y = X - sin(6 * pi * tile(X[:, 0], (1, D)) + tile(np.arange(1, D + 1), (N, 1)) * pi / D)
        temp = Y[:, 1] < 3 / 2 * (1 - sqrt(1 / 2))
        if temp:
            h = np.atleast_2d(np.concatenate([Y[:, 0] ** 2, abs(Y[temp, 1]), (Y[:, 2:] ** 2).ravel()]))
        else:
            h = np.atleast_2d(
                np.concatenate([Y[:, 0] ** 2, 0.125 + (Y[~temp, 1] - 1) ** 2, (Y[:, 2:] ** 2).ravel()])
            )
        F1 = X[:, 0] + sum(h[:, J1], 1)
        F2 = 1 - X[:, 0] + sum(h[:, J2], 1)
        return np.hstack([F1, F2])

    @timeit
    def constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_decision_vars
        t = X[:, 1] - sin(6 * pi * X[:, 0] + 2 * pi / D) - 0.5 * X[:, 0] + 0.25
        return -t / (1 + exp(4 * abs(t)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = 1 - f1
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = -0.5 * f1[idx1] + 3 / 4
        f2[idx2] = 1 - f1[idx2] + 0.125
        return np.c_[f1, f2]


class CF5(ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, np.zeros(self.n_decision_vars - 1) - 2]
        self.upper_bounds = np.r_[1, np.zeros(self.n_decision_vars - 1) + 2]
        self.n_ieq_constr = 1

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N = X.shape[0]
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        Y1 = X[:, J1] - 0.8 * tile(X[:, 0], (1, len(J1))) * cos(
            6 * pi * tile(X[:, 0], (1, len(J1))) + tile(J1 + 1, (N, 1)) * pi / D
        )
        Y2 = X[:, J2] - 0.8 * tile(X[:, 0], (1, len(J2))) * sin(
            6 * pi * tile(X[:, 0], (1, len(J2))) + tile(J2 + 1, (N, 1)) * pi / D
        )
        h1 = 2 * Y1**2 - cos(4 * pi * Y1) + 1
        h2 = 2 * Y2[:, 1:] ** 2 - cos(4 * pi * Y2[:, 1:]) + 1
        temp = Y2[:, 0] < 3 / 2 * (1 - sqrt(1 / 2))
        if temp:
            h3 = abs(Y2[:, 0])
        else:
            h3 = 0.125 + (Y2[:, 0] - 1) ** 2
        F1 = X[:, 0] + sum(h1, 1)
        F2 = 1 - X[:, 0] + sum(h2, 1) + h3
        return np.hstack([F1, F2])

    @timeit
    def constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_decision_vars
        return -X[:, 1] + 0.8 * X[:, 0] * sin(6 * pi * X[:, 0] + 2 * pi / D) + 0.5 * X[:, 0] - 0.25

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = 1 - f1
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = -0.5 * f1[idx1] + 3 / 4
        f2[idx2] = 1 - f1[idx2] + 0.125
        return np.c_[f1, f2]


class CF6(ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, np.zeros(self.n_decision_vars - 1) - 2]
        self.upper_bounds = np.r_[1, np.zeros(self.n_decision_vars - 1) + 2]
        self.n_ieq_constr = 2

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        Y1 = X[:, J1] - 0.8 * tile(X[:, 0], (1, len(J1))) * cos(
            6 * pi * tile(X[:, 0], (1, len(J1))) + (J1 + 1) * pi / D
        )
        Y2 = X[:, J2] - 0.8 * tile(X[:, 0], (1, len(J2))) * sin(
            6 * pi * tile(X[:, 0], (1, len(J2))) + (J2 + 1) * pi / D
        )
        return np.hstack([X[:, 0] + sum(Y1**2, 1), (1 - X[:, 0]) ** 2 + sum(Y2**2, 1)])

    @timeit
    def constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_decision_vars
        G1 = (
            -X[:, 1]
            + 0.8 * X[:, 0] * sin(6 * pi * X[:, 0] + 2 * pi / D)
            + sign(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2)
            * sqrt(abs(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2))
        )
        G2 = (
            -X[:, 3]
            + 0.8 * X[:, 0] * sin(6 * pi * X[:, 0] + 4 * pi / D)
            + sign(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0]))
            * sqrt(abs(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0])))
        )
        return np.hstack([G1, G2])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = (1 - f1) ** 2
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = 0.5 * (1 - f1[idx1])
        f2[idx2] = 0.25 * sqrt(1 - f1[idx2])
        return np.c_[f1, f2]


class CF7(ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_objectives = 2
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, np.zeros(self.n_decision_vars - 1) - 2]
        self.upper_bounds = np.r_[1, np.zeros(self.n_decision_vars - 1) + 2]
        self.n_ieq_constr = 2

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_decision_vars
        J1 = np.arange(3, self.n_decision_vars, 2) - 1
        J2 = np.arange(2, self.n_decision_vars + 2, 2) - 1
        Y1 = X[:, J1] - cos(6 * pi * tile(X[:, 0], (1, len(J1))) + (J1 + 1) * pi / D)
        Y2 = X[:, J2] - sin(6 * pi * tile(X[:, 0], (1, len(J2))) + (J2 + 1) * pi / D)
        h1 = 2 * Y1**2 - cos(4 * pi * Y1) + 1
        h2 = 2 * Y2[:, 2:] ** 2 - cos(4 * pi * Y2[:, 2:]) + 1
        h3 = Y2[:, :2] ** 2
        return np.hstack([X[:, 0] + sum(h1, 1), (1 - X[:, 0]) ** 2 + sum(h2, 1) + sum(h3, 1)])

    @timeit
    def constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_decision_vars
        G1 = (
            -X[:, 1]
            + sin(6 * pi * X[:, 0] + 2 * pi / D)
            + sign(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2)
            * sqrt(abs(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2))
        )
        G2 = (
            -X[:, 3]
            + sin(6 * pi * X[:, 0] + 4 * pi / D)
            + sign(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0]))
            * sqrt(abs(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0])))
        )
        return np.hstack([G1, G2])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = (1 - f1) ** 2
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = 0.5 * (1 - f1[idx1])
        f2[idx2] = 0.25 * sqrt(1 - f1[idx2])
        return np.c_[f1, f2]


class CF8(UF8, ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__(n_decision_vars)
        self.n_ieq_constr = 1
        self.n_objectives = 3
        self.n_decision_vars = n_decision_vars
        self.lower_bounds = np.r_[0, 0, np.zeros(self.n_decision_vars - 2) - 4]
        self.upper_bounds = np.r_[1, 1, np.zeros(self.n_decision_vars - 2) + 4]

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        np.clip(x, self.lower_bounds, self.upper_bounds)
        return super().objective(x)

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        # TODO: avoid calling the constraint fucntion two times
        F = self.objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + 4 * abs(sin(2 * pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1)))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        N = _np.ceil(N / 5) * 5
        R = _np.zeros((N, 3))
        R[:, 2] = tile(sin(_np.linspace(0, 1, num=N / 5 - 1) * pi / 2).reshape(-1, 1), (5, 1))
        for i in range(0, 5):
            R[i * N / 5 : (i + 1) * N / 5, 0] = sqrt(i / 4 * (1 - R[i * N / 5 : (i + 1) * N / 5, 2] ** 2))
        R[:, 1] = sqrt(max(1 - R[:, 0] ** 2 - R[:, 2] ** 2, 0))
        return R


class CF9(CF8, ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__(n_decision_vars)
        self.lower_bounds = np.r_[0, 0, np.zeros(self.n_decision_vars - 2) - 2]
        self.upper_bounds = np.r_[1, 1, np.zeros(self.n_decision_vars - 2) + 2]

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        return super().objective(x)

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        F = self.objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + 3 * np.sin(2 * np.pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        R = _np.random.rand(N, 3)
        R = R / tile(sqrt(sum(R**2, 1)), (1, 3))
        idx = (1e-5 < R[:, 0]) & (R[:, 0] < sqrt((1 - R[:, 2] ** 2) / 4)) | (
            sqrt((1 - R[:, 2] ** 2) / 2) < R[:, 0]
        ) & (R[:, 0] < sqrt(3 * (1 - R[:, 2] ** 2) / 4))
        return R[~idx]


class CF10(ConstrainedMOOAnalytical):
    def __init__(self, n_decision_vars: int = 10) -> None:
        super().__init__()
        self.n_decision_vars = n_decision_vars
        self.n_ieq_constr = 1
        self.n_objectives = 3
        self.lower_bounds = np.r_[0, 0, np.zeros(self.n_decision_vars - 2) - 2]
        self.upper_bounds = np.r_[1, 1, np.zeros(self.n_decision_vars - 2) + 2]

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N = X.shape[0]
        D = self.n_decision_vars
        J1 = arange(4, D + 1, 3) - 1
        J2 = arange(5, D + 1, 3) - 1
        J3 = arange(3, D + 1, 3) - 1
        Y = X - 2 * tile(X[:, [1]], (1, D)) * sin(
            2 * pi * tile(X[:, [0]], (1, D)) + tile(arange(D) + 1, (N, 1)) * pi / D
        )
        F1 = cos(0.5 * X[:, 0] * pi) * cos(0.5 * X[:, 1] * pi) + 2 * mean(
            4 * Y[:, J1] ** 2 - cos(8 * pi * Y[:, J1]) + 1, 1
        )
        F2 = cos(0.5 * X[:, 0] * pi) * sin(0.5 * X[:, 1] * pi) + 2 * mean(
            4 * Y[:, J2] ** 2 - cos(8 * pi * Y[:, J2]) + 1, 1
        )
        F3 = sin(0.5 * X[:, 0] * pi) + 2 * mean(4 * Y[:, J3] ** 2 - cos(8 * pi * Y[:, J3]) + 1, 1)
        return np.hstack([F1, F2, F3])

    @timeit
    def constraint(self, x: np.ndarray) -> float:
        F = self.objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + sin(2 * pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        R = _np.random.rand(N, 3)
        R = R / tile(sqrt(sum(R**2, 1)), (1, 3))
        idx = (1e-5 < R[:, 0]) & (R[:, 0] < sqrt((1 - R[:, 2] ** 2) / 4)) | (
            sqrt((1 - R[:, 2] ** 2) / 2) < R[:, 0]
        ) & (R[:, 0] < sqrt(3 * (1 - R[:, 2] ** 2) / 4))
        return R[~idx]
