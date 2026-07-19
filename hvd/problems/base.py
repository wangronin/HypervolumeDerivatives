import warnings
from functools import partial
from typing import Any, Callable, ClassVar

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


def fixed_n_var(n_var: int | None, default: int, problem_name: str) -> int:
    """Return a problem's fixed decision-space dimension and warn about overrides."""
    if n_var is not None:
        warnings.warn(
            f"{problem_name} has a fixed n_var={default}; the supplied n_var={n_var} "
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

    default_n_obj: ClassVar[int]

    def __init__(self, n_var: int, n_obj: int, xl: ArrayLike, xu: ArrayLike) -> None:
        self.n_var: int = self._validate_dimension("n_var", n_var)
        self.n_obj: int = self._validate_dimension("n_obj", n_obj)
        self.xl: np.ndarray = self._broadcast_bound("xl", xl)
        self.xu: np.ndarray = self._broadcast_bound("xu", xu)
        self._validate_bound()
        self._obj_func = jit(partial(self.__class__._objective, self))
        self._objective_jacobian = jit(jacrev(self._obj_func))
        self._objective_hessian = jit(hessian(self._obj_func))
        self._objective_batch = jit(vmap(self._obj_func))
        self._objective_jacobian_batch = jit(vmap(self._objective_jacobian))
        self._objective_hessian_batch = jit(vmap(self._objective_hessian))
        self.CPU_time: int = 0  # in nanoseconds

    @staticmethod
    def _validate_dimension(name: str, value: int) -> int:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"`{name}` must be a positive integer.")
        return int(value)

    def _validate_bound(self) -> None:
        if np.any(self.xl > self.xu):
            raise ValueError("Every lower bound in `xl` must be less than or equal to `xu`.")

    def _broadcast_bound(self, name: str, value: ArrayLike) -> np.ndarray:
        try:
            return np.broadcast_to(np.asarray(value, dtype=float), (self.n_var,)).copy()
        except ValueError as error:
            raise ValueError(f"`{name}` must be scalar or have shape `(n_var,)`.") from error

    def objective(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._obj_func(x))

    def objective_batch(self, x: np.ndarray) -> np.ndarray:
        """Evaluate a population without a Python-level loop."""
        return np.asarray(self._objective_batch(jnp.asarray(x)))

    @timeit
    def objective_jacobian_batch(self, x: np.ndarray) -> np.ndarray:
        """Evaluate one objective Jacobian per row of a population."""
        batch_size = len(x)
        return np.asarray(self._objective_jacobian_batch(jnp.asarray(x))).reshape(
            batch_size, self.n_obj, self.n_var
        )

    @timeit
    def objective_hessian_batch(self, x: np.ndarray) -> np.ndarray:
        """Evaluate one objective Hessian tensor per row of a population."""
        batch_size = len(x)
        return np.asarray(self._objective_hessian_batch(jnp.asarray(x))).reshape(
            batch_size, self.n_obj, self.n_var, self.n_var
        )

    def as_pymoo_problem(self):
        """Return a pymoo view of this problem (pymoo is imported lazily)."""
        from ._pymoo_wrapper import _PymooProblemAdapter

        return _PymooProblemAdapter(self)

    @classmethod
    def from_pymoo(cls, problem: Any, *, boundary_constraints: bool = True) -> "MOP":
        """Adapt a JAX-traceable pymoo problem to the native MOP API.

        This adapter does not translate NumPy or autograd operations to JAX.
        Use ``get_pymoo_problem`` for supported built-in pymoo problems.
        """
        if isinstance(problem, cls):
            return problem
        from ._pymoo_wrapper import _adapt_pymoo_problem

        return _adapt_pymoo_problem(problem, boundary_constraints=boundary_constraints)

    @timeit
    def objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._objective_jacobian(x)).reshape(self.n_obj, self.n_var)

    @timeit
    def objective_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._objective_hessian(x)).reshape(self.n_obj, self.n_var, self.n_var)


class CMOP(MOP):
    """Constrained MOP with class-level constraint defaults and instance-level effective counts."""

    default_n_eq_constr: ClassVar[int] = 0
    default_n_ieq_constr: ClassVar[int] = 0

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
        self.n_eq_constr: int = int(n_eq_constr)
        self.n_ieq_constr: int = int(n_ieq_constr)
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

        if self._eq is not None:
            self._eq_jacobian = jit(jacrev(self._eq))
            self._eq_hessian = hessian(self._eq)
            self._eq_batch = jit(vmap(self._eq))
            self._eq_jacobian_batch = jit(vmap(self._eq_jacobian))
            self._eq_hessian_batch = jit(vmap(self._eq_hessian))
        if self._ieq is not None:
            self._ieq_jacobian = jit(jacrev(self._ieq))
            self._ieq_hessian = hessian(self._ieq)
            self._ieq_batch = jit(vmap(self._ieq))
            self._ieq_jacobian_batch = jit(vmap(self._ieq_jacobian))
            self._ieq_hessian_batch = jit(vmap(self._ieq_hessian))

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
        return np.array(self._eq_jacobian(x)).reshape(self.n_eq_constr, self.n_var)

    @timeit
    def eq_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._eq_hessian(x)).reshape(self.n_eq_constr, self.n_var, self.n_var)

    @timeit
    def eq_jacobian_batch(self, x: np.ndarray) -> np.ndarray:
        batch_size = len(x)
        return np.asarray(self._eq_jacobian_batch(jnp.asarray(x))).reshape(
            batch_size, self.n_eq_constr, self.n_var
        )

    @timeit
    def eq_hessian_batch(self, x: np.ndarray) -> np.ndarray:
        batch_size = len(x)
        return np.asarray(self._eq_hessian_batch(jnp.asarray(x))).reshape(
            batch_size, self.n_eq_constr, self.n_var, self.n_var
        )

    @timeit
    def ieq_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_jacobian(x)).reshape(self.n_ieq_constr, self.n_var)

    @timeit
    def ieq_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_hessian(x)).reshape(self.n_ieq_constr, self.n_var, self.n_var)

    @timeit
    def ieq_jacobian_batch(self, x: np.ndarray) -> np.ndarray:
        batch_size = len(x)
        return np.asarray(self._ieq_jacobian_batch(jnp.asarray(x))).reshape(
            batch_size, self.n_ieq_constr, self.n_var
        )

    @timeit
    def ieq_hessian_batch(self, x: np.ndarray) -> np.ndarray:
        batch_size = len(x)
        return np.asarray(self._ieq_hessian_batch(jnp.asarray(x))).reshape(
            batch_size, self.n_ieq_constr, self.n_var, self.n_var
        )
