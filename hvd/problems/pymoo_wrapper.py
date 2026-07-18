import jax
import jax.numpy as jnp
import numpy as np
from pymoo.core.problem import Problem as PymooProblem

from .base import MOP, ConstrainedMOP


class _PymooBackedMOP(ConstrainedMOP):
    """Adapt a JAX-compatible pymoo problem to the native analytical API.

    The wrapped ``_evaluate`` implementation must already use JAX-traceable
    operations; wrapping alone cannot make NumPy or autograd code traceable.
    """

    def __init__(self, problem: PymooProblem, boundary_constraints: bool = True) -> None:
        if not isinstance(problem, PymooProblem):
            raise TypeError(f"Expected a pymoo Problem, got {type(problem).__name__}.")
        self._problem = problem
        self.n_obj = self._problem.n_obj
        self.n_var = self._problem.n_var
        self.n_eq_constr = getattr(self._problem, "n_eq_constr", 0)
        self.n_ieq_constr = getattr(self._problem, "n_ieq_constr", 0)
        self.xl = np.broadcast_to(self._problem.xl, (self.n_var,)).copy()
        self.xu = np.broadcast_to(self._problem.xu, (self.n_var,)).copy()
        self._validate_jax_traceability()
        super().__init__(boundry_constraints=boundary_constraints)

    def _validate_jax_traceability(self) -> None:
        """Fail early when pymoo's evaluation cannot be traced by JAX."""
        x = jnp.zeros(self.n_var)
        outputs = [("objective output 'F'", self._objective)]
        if self.n_ieq_constr > 0:
            outputs.append(("inequality-constraint output 'G'", self._ieq_constraint))
        if self.n_eq_constr > 0:
            outputs.append(("equality-constraint output 'H'", self._eq_constraint))

        for label, function in outputs:
            try:
                jax.eval_shape(function, x)
            except Exception as error:
                raise TypeError(
                    f"The pymoo {label} is not JAX-traceable. "
                    "Use only JAX-compatible operations in `_evaluate`. For pymoo's "
                    "built-in problems, call `pymoo.gradient.activate('jax.numpy')` "
                    "before importing the problem module."
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


# TODO:  decide what to do with it
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
