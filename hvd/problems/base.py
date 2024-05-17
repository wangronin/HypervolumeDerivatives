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

    # TODO: implement it
    # def pareto_front(self, *args, **kwargs):
    # return self._problem.pareto_front(*args, **kwargs)
