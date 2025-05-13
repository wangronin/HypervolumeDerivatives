from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit

from ..utils import timeit


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
    n_eq_constr: int = 0
    n_ieq_constr: int = 0

    def __init__(self, boundry_constraints: bool = False):
        super().__init__()
        self._eq: callable = (
            jit(partial(self.__class__._eq_constraint, self)) if self.n_eq_constr > 0 else None
        )
        self._ieq: callable = (
            jit(partial(self.__class__._ieq_constraint, self)) if self.n_ieq_constr > 0 else None
        )
        if boundry_constraints:
            self._ieq = add_boundry_constraints(self._ieq, self.xl, self.xu)
            self.n_ieq_constr += 2 * self.n_var

        if self._eq:
            self._eq_jacobian = jit(jacrev(self._eq))
            self._eq_hessian = hessian(self._eq)
        if self._ieq:
            self._ieq_jacobian = jit(jacrev(self._ieq))
            self._ieq_hessian = hessian(self._ieq)

    def eq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._eq(x)])

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
