from functools import partial
from typing import Any, Callable
import warnings

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit, vmap
from numpy.typing import ArrayLike

from ..utils import timeit

JaxFunction = Callable[[jnp.ndarray], jnp.ndarray]


def hessian(fun: JaxFunction) -> JaxFunction:
    return jit(jacfwd(jacrev(fun)))


def fixed_n_obj(n_obj: int | None, default: int, problem_name: str) -> int:
    """Return a problem's fixed objective count and warn about explicit overrides."""
    if n_obj is not None:
        warnings.warn(
            f"{problem_name} has a fixed n_obj={default}; the supplied n_obj={n_obj} "
            "is ignored because it cannot be changed for this problem.",
            UserWarning,
            stacklevel=3,
        )
    return default


def add_boundary_constraints(
    ieq_func: JaxFunction | None,
    xl: np.ndarray,
    xu: np.ndarray,
) -> JaxFunction:

    def func(x: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.concatenate([ieq_func(x), xl - x, x - xu])
            if ieq_func is not None
            else jnp.concatenate([xl - x, x - xu])
        )

    return func


class MOP:
    """Base contract for a differentiable multi-objective problem."""

    n_obj: int
    n_var: int
    xl: np.ndarray
    xu: np.ndarray

    def __init__(self, n_var: int, n_obj: int, xl: ArrayLike, xu: ArrayLike) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.asarray(xl)
        self.xu = np.asarray(xu)
        self._validate_problem_definition()
        self._obj_func = jit(partial(self.__class__._objective, self))
        self._objective_jacobian = jit(jacrev(self._obj_func))
        self._objective_hessian = jit(hessian(self._obj_func))
        self._objective_batch = jit(vmap(self._obj_func))
        self.CPU_time: int = 0  # in nanoseconds

    def _validate_problem_definition(self) -> None:
        """Validate and normalize the metadata supplied by a concrete problem."""
        for name in ("n_obj", "n_var", "xl", "xu"):
            if not hasattr(self, name):
                raise TypeError(f"{type(self).__name__} must define `{name}` before calling MOP.__init__().")

        if not isinstance(self.n_obj, (int, np.integer)) or self.n_obj < 1:
            raise ValueError("`n_obj` must be a positive integer.")
        if not isinstance(self.n_var, (int, np.integer)) or self.n_var < 1:
            raise ValueError("`n_var` must be a positive integer.")
        self.n_obj = int(self.n_obj)
        self.n_var = int(self.n_var)

        try:
            self.xl = np.broadcast_to(np.asarray(self.xl, dtype=float), (self.n_var,)).copy()
            self.xu = np.broadcast_to(np.asarray(self.xu, dtype=float), (self.n_var,)).copy()
        except ValueError as error:
            raise ValueError("`xl` and `xu` must be scalar or have shape `(n_var,)`.") from error
        if np.any(self.xl > self.xu):
            raise ValueError("Every lower bound in `xl` must be less than or equal to `xu`.")

    def objective(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._obj_func(x))

    def objective_batch(self, x: np.ndarray) -> np.ndarray:
        """Evaluate a population without a Python-level loop."""
        return np.asarray(self._objective_batch(jnp.asarray(x)))

    def as_pymoo_problem(self):
        """Return a pymoo view of this problem (pymoo is imported lazily)."""
        from .pymoo_wrapper import _PymooProblemAdapter

        return _PymooProblemAdapter(self)

    @classmethod
    def from_pymoo(cls, problem: Any, *, boundary_constraints: bool = True) -> "MOP":
        """Adapt a JAX-traceable pymoo problem to the native MOP API.

        This adapter does not translate NumPy or autograd operations to JAX. For
        pymoo's built-in problems, activate pymoo's ``jax.numpy`` gradient
        toolbox before importing the problem module.
        """
        if isinstance(problem, cls):
            return problem
        from .pymoo_wrapper import _PymooBackedMOP

        return _PymooBackedMOP(problem, boundary_constraints=boundary_constraints)

    @timeit
    def objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._objective_jacobian(x)).reshape(self.n_obj, self.n_var)

    @timeit
    def objective_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._objective_hessian(x)).reshape(self.n_obj, self.n_var, self.n_var)


class ConstrainedMOP(MOP):
    n_eq_constr: int = 0
    n_ieq_constr: int = 0

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        xl: ArrayLike,
        xu: ArrayLike,
        n_eq_constr: int = 0,
        n_ieq_constr: int = 0,
        boundary_constraints: bool = False,
    ) -> None:
        self.n_eq_constr = int(n_eq_constr)
        self.n_ieq_constr = int(n_ieq_constr)
        if self.n_eq_constr < 0 or self.n_ieq_constr < 0:
            raise ValueError("Constraint counts must be non-negative integers.")

        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        self._eq: JaxFunction | None = (
            jit(partial(self.__class__._eq_constraint, self)) if self.n_eq_constr > 0 else None
        )
        self._ieq: JaxFunction | None = (
            jit(partial(self.__class__._ieq_constraint, self)) if self.n_ieq_constr > 0 else None
        )
        if boundary_constraints:
            self._ieq = add_boundary_constraints(self._ieq, self.xl, self.xu)
            self.n_ieq_constr += 2 * self.n_var

        if self._eq:
            self._eq_jacobian = jit(jacrev(self._eq))
            self._eq_hessian = hessian(self._eq)
            self._eq_batch = jit(vmap(self._eq))
        if self._ieq:
            self._ieq_jacobian = jit(jacrev(self._ieq))
            self._ieq_hessian = hessian(self._ieq)
            self._ieq_batch = jit(vmap(self._ieq))

    def eq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._eq(x))

    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq(x))

    def eq_constraint_batch(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._eq_batch(jnp.asarray(x)))

    def ieq_constraint_batch(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._ieq_batch(jnp.asarray(x)))

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
