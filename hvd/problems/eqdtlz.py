import warnings
from typing import ClassVar

import jax.numpy as jnp
import numpy as np

from ..utils import timeit
from .dtlz import _DTLZ, DTLZ1, DTLZ2, DTLZ3, DTLZ4


class InvertedDTLZ(_DTLZ):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x_, x_M = x[: self.n_obj - 1], x[self.n_obj - 1 :]
        g = self.g(x_M)
        x1, x2 = self._transform_x(x_, g)
        return (1 + g) / 2 - self.scale * (1 + g) * jnp.cumprod(jnp.r_[[1], x1])[::-1] * jnp.r_[[1], x2[::-1]]
        # TODO: it seems the implementation in PlatEMO do not have the `1/2` factor, as below.
        # need to figure out which one is the correct definition.
        # the above implementation is from Cuate, Oliver, Lourdes Uribe, Adriana Lara, and Oliver Schütze.
        # "A benchmark for equality constrained multi-objective optimization."
        # Swarm and Evolutionary Computation 52 (2020): 100619.
        # ```
        # return (1 + g) - self.scale * (1 + g) * jnp.cumprod(jnp.r_[[1], x1])[::-1] * jnp.r_[[1], x2[::-1]]
        # ```

    def get_pareto_front(self, **kwargs) -> np.ndarray:
        ref_dirs = super().get_pareto_front(**kwargs)
        # TODO: it seems the implementation in PlatEMO do not have the `1/2` factor, as below.
        # need to figure out which one is the correct definition.
        # ```return 1 - ref_dirs```
        return 0.5 - ref_dirs


class ConstrainedDTLZ(_DTLZ):
    default_n_eq_constr = 1
    eq_radius: ClassVar[float] = 0.4

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - self.eq_radius**2)

    def get_pareto_set(self, N: int = 1000) -> np.ndarray:
        """Intersect the parent Pareto set with both surfaces where h(x) = 0."""
        M = self.n_obj - 1
        if M == 1:
            # bi-objective case: the feasible efficient set contains only two points
            if N > 2:
                warnings.warn(
                    f"The supplied N={N} is ignored because in the bi-objective case, "
                    f"{type(self).__name__}'s feasible efficient set contains only two points.",
                    UserWarning,
                    stacklevel=3,
                )
            directions = np.array([[-1.0], [1.0]])
        elif M == 2:
            # evenly-spaced points on S^2
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            # Haar-random points on the S^M
            directions = np.random.default_rng(0).normal(size=(N, M))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        positions = 0.5 + self.eq_radius * directions
        return np.column_stack((positions, np.full((len(positions), self.k), 0.5)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        """Evaluate the constrained Pareto set using the concrete objective family."""
        return self.objective(self.get_pareto_set(N))


class IDTLZ1(InvertedDTLZ, DTLZ1):
    pass


class IDTLZ2(InvertedDTLZ, DTLZ2):
    pass


class IDTLZ3(InvertedDTLZ, DTLZ3):
    pass


class IDTLZ4(InvertedDTLZ, DTLZ4):
    pass


class Eq1DTLZ1(ConstrainedDTLZ, DTLZ1):
    pass


class Eq1IDTLZ1(ConstrainedDTLZ, IDTLZ1):
    pass


class Eq1DTLZ2(ConstrainedDTLZ, DTLZ2):
    pass


class Eq1IDTLZ2(ConstrainedDTLZ, IDTLZ2):
    pass


class Eq1DTLZ3(ConstrainedDTLZ, DTLZ3):
    pass


class Eq1IDTLZ3(ConstrainedDTLZ, IDTLZ3):
    pass


class Eq1DTLZ4(ConstrainedDTLZ, DTLZ4):
    pass


class Eq1IDTLZ4(ConstrainedDTLZ, IDTLZ4):
    pass
