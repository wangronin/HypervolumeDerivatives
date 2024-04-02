import os
from functools import partial

# enable double-precision of JAX
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit
from pymoo.core.problem import ElementwiseProblem as PymooElementwiseProblem
from pymoo.core.problem import Problem
from pymoo.core.problem import Problem as PymooProblem

from ..utils import timeit

__author__ = ["Hao Wang"]


def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


def add_boundry_constraints(ieq_func, xl, xu):

    def func(x):
        return (
            jnp.concatenate([ieq_func(x), xl - x, x - xu])
            if ieq_func is not None
            else jnp.concatenate([xl - x, x - xu])
        )

    return func


class MOOAnalytical:
    def __init__(self):
        self._obj_func = jit(partial(self.__class__._objective, self))
        self._objective_jacobian = jit(jacrev(self._obj_func))
        self._objective_hessian = jit(hessian(self._obj_func))
        self.CPU_time: int = 0  # in nanoseconds

    def objective(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._obj_func(x))
        # return self._objective(x)

    @timeit
    def objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._objective_jacobian(x))

    @timeit
    def objective_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._objective_hessian(x))


class ConstrainedMOOAnalytical(MOOAnalytical):
    n_eq_constr = 0
    n_ieq_constr = 0

    def __init__(self, boundry_constraints: bool = False):
        super().__init__()
        self._eq = jit(partial(self.__class__._eq_constraint, self)) if self.n_eq_constr > 0 else None
        self._ieq = partial(self.__class__._ieq_constraint, self) if self.n_ieq_constr > 0 else None

        if boundry_constraints:
            self._ieq = add_boundry_constraints(self._ieq, self.xl, self.xu)
            self.n_ieq_constr += 2 * self.n_var
        if self._ieq is not None:
            self._ieq = jit(self._ieq)

        self._eq_jacobian = jit(jacrev(self._eq)) if self._eq is not None else None
        self._eq_hessian = hessian(self._eq) if self._eq is not None else None
        self._ieq_jacobian = jit(jacrev(self._ieq)) if self._ieq is not None else None
        self._ieq_hessian = hessian(self._ieq) if self._ieq is not None else None

    def eq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._eq(x))

    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq(x))

    @timeit
    def eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._eq_jacobian(x))

    @timeit
    def eq_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._eq_hessian(x))

    @timeit
    def ieq_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_jacobian(x))

    @timeit
    def ieq_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_hessian(x))


# TODO: unify this class with the `ConstrainedMOOAnalytical`
class PymooProblemWithAD:
    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self.n_obj = self._problem.n_obj
        self.n_var = self._problem.n_var
        self.n_eq_constr = self._problem.n_eq_constr
        self.n_ieq_constr = self._problem.n_ieq_constr + 2 * self.n_var
        self.xl = self._problem.xl
        self.xu = self._problem.xu
        self._obj_func = jit(partial(problem.__class__._evaluate, problem))
        ieq_func = jit(partial(PymooProblemWithAD.ieq_constraint, self))
        self._objective_jacobian = jit(jacrev(self._obj_func))
        self._objective_hessian = jit(hessian(self._obj_func))
        self._ieq_jacobian = jit(jacfwd(ieq_func))
        self.CPU_time: int = 0  # measured in nanoseconds

    def objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._obj_func(x)

    def ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        # box constraints are converted to inequality constraints
        return jnp.concatenate([self.xl - x, x - self.xu])

    @timeit
    def objective_jacobian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._objective_jacobian(x)

    @timeit
    def objective_hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._objective_hessian(x)

    @timeit
    def ieq_jacobian(self, x: jnp.ndarray) -> jnp.ndarray:
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
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        func = lambda x, a: jnp.sum((x - a) ** 2)
        return jnp.array([func(x, self.a1), func(x, self.a2), func(x, self.a3)])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        w = np.random.rand(N, 3)
        w /= w.sum(axis=1).reshape(-1, 1)
        X = w @ np.vstack([self.a1, self.a2, self.a3])
        return np.array([self._objective(x) for x in X])


class CONV4(MOOAnalytical):
    pass


class CONV4_2F(ConstrainedMOOAnalytical):
    """Convex Problem 4 with 2 disconnected Pareto fronts"""

    def __init__(self, **kwargs):
        self.n_obj = 4
        self.n_var = 4
        self.xl = -10 * np.ones(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        super().__init__(**kwargs)

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        a = jnp.eye(self.n_var)
        deltaa = jnp.ones(self.n_var)
        fa4 = jnp.array([2, 2, 2, 0])
        fa1 = jnp.array([0, 2, 2, 2])
        deltay = fa4 - fa1
        z = x + deltaa
        y = jax.lax.select(
            jnp.all(x < 0),
            jnp.array([jnp.sum((z - a[i]) ** 2) - 3.5 * deltay[i] for i in range(4)]),
            jnp.array([jnp.sum((x - a[i]) ** 2) for i in range(4)]),
        )
        return y

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        N1 = int(np.ceil(N / 2))
        N2 = max(1, N - N1)
        w1 = np.random.rand(N1, 4)
        w1 /= w1.sum(axis=1).reshape(-1, 1)
        w2 = np.random.rand(N2, 4)
        w2 /= w2.sum(axis=1).reshape(-1, 1)
        X_1 = w1 @ np.eye(self.n_var)  # the positive part
        X_2 = w2 @ (np.eye(self.n_var) - np.ones((self.n_var, self.n_var)))  # the negative part
        return np.array([self._objective(x) for x in np.vstack([X_1, X_2])])


class UF7(MOOAnalytical):
    def __init__(self, n_var: int = 30) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 1]
        self.xu = np.ones(self.n_var)
        self.encoding = np.ones(self.n_var)
        super().__init__()

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        N = x.shape[0]
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        y = x - jnp.sin(
            6 * jnp.pi * jnp.tile(x[:, [0]], (1, D)) + jnp.tile(jnp.arange(D) + 1, (N, 1)) * jnp.pi / D
        )
        return jnp.hstack(
            [
                x[:, 0] ** 0.2 + 2 * jnp.mean(y[:, J1] ** 2, 1),
                1 - x[:, 0] ** 0.2 + 2 * jnp.mean(y[:, J2] ** 2, 1),
            ]
        ).T

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        return np.c_[f, 1 - f]


class UF8(MOOAnalytical):
    def __init__(self, n_var: int = 30, **kwargs) -> None:
        self.n_obj = 3
        self.n_var = n_var
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 2]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 2]
        self.encoding = np.ones(self.n_var)
        super().__init__()

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        N = x.shape[0]
        D = self.n_var
        J1 = jnp.arange(4, D + 1, 3) - 1
        J2 = jnp.arange(5, D + 1, 3) - 1
        J3 = jnp.arange(3, D + 1, 3) - 1
        y = x - 2 * jnp.tile(x[:, [1]], (1, D)) * jnp.sin(
            2 * jnp.pi * jnp.tile(x[:, [0]], (1, D)) + jnp.tile(jnp.arange(D) + 1, (N, 1)) * jnp.pi / D
        )
        return jnp.hstack(
            [
                jnp.cos(0.5 * x[:, 0] * jnp.pi) * jnp.cos(0.5 * x[:, 1] * jnp.pi)
                + 2 * jnp.mean(y[:, J1] ** 2, 1),
                jnp.cos(0.5 * x[:, 0] * jnp.pi) * jnp.sin(0.5 * x[:, 1] * jnp.pi)
                + 2 * jnp.mean(y[:, J2] ** 2, 1),
                jnp.sin(0.5 * x[:, 0] * jnp.pi) + 2 * jnp.mean(y[:, J3] ** 2, 1),
            ]
        ).T


class PymooProblemWrapper(PymooElementwiseProblem):
    """Wrap of the problem I wrote into `Pymoo`'s problem"""

    def __init__(self, problem: MOOAnalytical) -> None:
        self._problem = problem
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            xl=problem.xl,
            xu=problem.xu,
            n_ieq_constr=self._problem.n_ieq_constr if hasattr(self._problem, "n_ieq_constr") else 0,
            n_eq_constr=self._problem.n_eq_constr if hasattr(self._problem, "n_eq_constr") else 0,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        x = np.atleast_2d(x)
        out["F"] = np.array([self._problem.objective(_) for _ in x])  # objective value
        if hasattr(self._problem, "n_eq_constr") and self._problem.n_eq_constr > 0:
            out["H"] = np.array([self._problem.eq_constraint(_) for _ in x])  # equality constraint value
        if hasattr(self._problem, "n_ieq_constr") and self._problem.n_ieq_constr > 0:
            out["G"] = np.array([self._problem.ieq_constraint(_) for _ in x])  # inequality constraint value

    def pareto_front(self, *args, **kwargs) -> np.ndarray:
        return self._problem.get_pareto_front(*args, **kwargs)


class ModifiedObjective(PymooProblem):
    """Modified objective function based on the following paper:

    Ishibuchi, H.; Matsumoto, T.; Masuyama, N.; Nojima, Y.
    Effects of dominance resistant solutions on the performance of evolutionary multi-objective
    and many-objective algorithms. In Proceedings of the Genetic and Evolutionary Computation
    Conference (GECCO '20), Cancún, Mexico, 8-12 July 2020.
    """

    def __init__(self, problem: PymooProblem) -> None:
        self._problem = problem
        self._alpha = 0.02
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            xl=problem.xl,
            xu=problem.xu,
            n_ieq_constr=self._problem.n_ieq_constr if hasattr(self._problem, "n_ieq_constr") else 0,
            n_eq_constr=self._problem.n_eq_constr if hasattr(self._problem, "n_eq_constr") else 0,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        self._problem._evaluate(x, out, *args, **kwargs)
        F = out["F"]
        out["F"] = (1 - self._alpha) * F + self._alpha * np.tile(
            F.sum(axis=1).reshape(-1, 1), (1, self.n_obj)
        ) / self.n_obj

    # def pareto_front(self, *args, **kwargs):
    # return self._problem.pareto_front(*args, **kwargs)
