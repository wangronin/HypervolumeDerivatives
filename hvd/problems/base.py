import autograd.numpy as np
from autograd import hessian, jacobian
from autograd.numpy import arange
from pymoo.core.problem import Problem

from ..utils import timeit

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
        self._eq_constraint_jacobian = (
            jacobian(self.eq_constraint) if hasattr(self, "eq_constraint") else None
        )
        self._eq_constraint_hessian = hessian(self.eq_constraint) if hasattr(self, "eq_constraint") else None
        self._ieq_constraint_jacobian = (
            jacobian(self.ieq_constraint) if hasattr(self, "ieq_constraint") else None
        )
        self._ieq_constraint_hessian = (
            hessian(self.ieq_constraint) if hasattr(self, "ieq_constraint") else None
        )

    @timeit
    def eq_constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._eq_constraint_jacobian(x)

    @timeit
    def eq_constraint_hessian(self, x: np.ndarray) -> np.ndarray:
        return self._eq_constraint_hessian(x)

    @timeit
    def ieq_constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._ieq_constraint_jacobian(x)

    @timeit
    def ieq_constraint_hessian(self, x: np.ndarray) -> np.ndarray:
        return self._ieq_constraint_hessian(x)


class PymooProblemWithAD:
    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self.n_obj = self._problem.n_obj
        self.n_var = self._problem.n_var
        self.n_eq_constr = self._problem.n_eq_constr
        # box constraints are converted to inequality constraints
        self.n_ieq_constr = self._problem.n_ieq_constr + 2 * self.n_var
        self.xl = self._problem.xl
        self.xu = self._problem.xu
        self._objective_jacobian = jacobian(self._problem._evaluate)
        self._objective_hessian = hessian(self._problem._evaluate)
        self._ieq_jacobian = jacobian(self.ieq_constraint)
        self.CPU_time: int = 0  # measured in nanoseconds

    def objective(self, x: np.ndarray) -> np.ndarray:
        return self._problem._evaluate(x)

    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([self.xl - x, x - self.xu])

    @timeit
    def objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._objective_jacobian(x)

    @timeit
    def objective_hessian(self, x: np.ndarray) -> np.ndarray:
        return self._objective_hessian(x)

    @timeit
    def ieq_jacobian(self, x) -> np.ndarray:
        return self._ieq_jacobian(x)

    def get_pareto_set(self, *args, **kwargs) -> np.ndarray:
        return self._problem._calc_pareto_set(*args, **kwargs)

    def get_pareto_front(self, *args, **kwargs) -> np.ndarray:
        return self._problem._calc_pareto_front(*args, **kwargs)


class CONV3(MOOAnalytical):
    def __init__(self):
        self.n_obj = 3
        self.n_var = 3
        self.xl = -3 * np.ones(self.n_var)
        self.xu = 3 * np.ones(self.n_var)
        self.a1 = -1 * np.ones(self.n_var)
        self.a2 = np.ones(self.n_var)
        self.a3 = np.r_[-1 * np.ones(self.n_var - 1), 1]
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        func = lambda x, a: np.sum((x - a) ** 2)
        return np.array([func(x, self.a1), func(x, self.a2), func(x, self.a3)])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        w = np.random.rand(N, 3)
        w /= w.sum(axis=1).reshape(-1, 1)
        X = w @ np.vstack([self.a1, self.a2, self.a3])
        return np.array([self.objective(x) for x in X])


class CONV4(MOOAnalytical):
    def __init__(self):
        self.n_obj = 4
        self.n_var = 4
        self.xl = -10 * np.ones(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        a = np.eye(self.n_var)
        deltaa = np.ones(self.n_var)
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
    def __init__(self, n_var: int = 30) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 1]
        self.xu = np.ones(self.n_var)
        self.encoding = np.ones(self.n_var)
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        N = x.shape[0]
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
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
    def __init__(self, n_var: int = 30) -> None:
        self.n_obj = 3
        self.n_var = n_var
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 2]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 2]
        self.encoding = np.ones(self.n_var)
        super().__init__()

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        N = x.shape[0]
        D = self.n_var
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
