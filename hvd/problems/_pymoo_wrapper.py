import jax
import jax.numpy as jnp
import numpy as np
from pymoo.core.problem import Problem as PymooProblem

from .base import CMOP, MOP


class _PymooBackedProblem(CMOP):
    """Adapt any JAX-compatible pymoo problem as a constrained MOP.

    The wrapped ``_evaluate`` implementation must already use JAX-traceable
    operations; wrapping alone cannot make NumPy or autograd code traceable.
    Unconstrained pymoo problems simply have zero equality and inequality
    constraints, which is already a valid :class:`CMOP` configuration.
    """

    _problem: PymooProblem

    def __init__(self, problem: PymooProblem, boundary_constraints: bool) -> None:
        self._validate_problem_type(problem)
        self._problem = problem
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            xl=problem.xl,
            xu=problem.xu,
            n_eq_constr=getattr(problem, "n_eq_constr", 0),
            n_ieq_constr=getattr(problem, "n_ieq_constr", 0),
            boundary_constraints=boundary_constraints,
        )
        self._validate_jax_traceability()

    @staticmethod
    def _validate_problem_type(problem: PymooProblem) -> None:
        if not isinstance(problem, PymooProblem):
            raise TypeError(f"Expected a pymoo Problem, got {type(problem).__name__}.")

    def _validate_jax_traceability(self) -> None:
        """Fail early when pymoo's evaluation cannot be traced by JAX."""
        x = jnp.zeros(self.n_var)
        outputs = [("objective output 'F'", self._objective)]
        ieq = getattr(self, "_ieq", None)
        eq = getattr(self, "_eq", None)
        if ieq is not None:
            outputs.append(("inequality constraints", ieq))
        if eq is not None:
            outputs.append(("equality constraints", eq))

        for label, function in outputs:
            try:
                jax.eval_shape(function, x)
            except Exception as error:
                raise TypeError(
                    f"The pymoo {label} is not JAX-traceable. "
                    "Use only JAX-compatible operations in `_evaluate`, or use "
                    "`get_pymoo_problem` for a supported built-in pymoo problem."
                ) from error

    def _evaluate_key(self, x: jnp.ndarray, key: str) -> jnp.ndarray:
        out = {}
        result = self._problem._evaluate(jnp.atleast_2d(x), out)
        if result is not None and not out and key == "F":
            return result
        return out[key]

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self._evaluate_key(x, "F")).reshape(-1)

    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self._evaluate_key(x, "G")).reshape(-1)

    def _eq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self._evaluate_key(x, "H")).reshape(-1)

    def get_pareto_set(self, *args, **kwargs) -> np.ndarray:
        method = getattr(self._problem, "get_pareto_set", None) or self._problem._calc_pareto_set
        return method(*args, **kwargs)

    def get_pareto_front(self, *args, **kwargs) -> np.ndarray:
        method = getattr(self._problem, "get_pareto_front", None) or self._problem._calc_pareto_front
        return method(*args, **kwargs)


def _adapt_pymoo_problem(problem: PymooProblem, boundary_constraints: bool) -> MOP:
    """Wrap a pymoo problem in the single CMOP-backed adapter."""
    return _PymooBackedProblem(problem, boundary_constraints=boundary_constraints)


class _PymooProblemAdapter(PymooProblem):
    """Wrap of the problem I wrote into `Pymoo`'s problem"""

    def __init__(self, problem: MOP) -> None:
        if not isinstance(problem, MOP):
            raise TypeError(f"Expected an MOP, got {type(problem).__name__}.")

        self._problem = problem
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            xl=problem.xl,
            xu=problem.xu,
            n_ieq_constr=getattr(self._problem, "n_ieq_constr", 0),
            n_eq_constr=getattr(self._problem, "n_eq_constr", 0),
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        x = np.atleast_2d(x)
        out["F"] = self._problem.objective_batch(x)
        if hasattr(self._problem, "n_eq_constr") and self._problem.n_eq_constr > 0:
            out["H"] = self._problem.eq_constraint_batch(x)
        if hasattr(self._problem, "n_ieq_constr") and self._problem.n_ieq_constr > 0:
            out["G"] = self._problem.ieq_constraint_batch(x)

    def pareto_front(self, *args, **kwargs) -> np.ndarray:
        return self._problem.get_pareto_front(*args, **kwargs)
