import jax.numpy as jnp
import numpy as np

from ..utils import timeit
from .dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, BaseDTLZ

__authors__ = ["Hao Wang"]


class InvertedDTLZ(BaseDTLZ):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x_, x_M = x[: self.n_obj - 1], x[self.n_obj - 1 :]
        g = self.g(x_M)
        x1, x2 = self._transform_x(x_, g)
        return (1 + g) / 2 - self.scale * (1 + g) * jnp.cumprod(jnp.r_[[1], x1])[::-1] * jnp.r_[[1], x2[::-1]]

    def get_pareto_front(self, **kwargs) -> np.ndarray:
        ref_dirs = super().get_pareto_front(**kwargs)
        return 1 / 2 - ref_dirs


class ConstrainedDTLZ(BaseDTLZ):
    n_eq_constr = 1

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4


class IDTLZ1(InvertedDTLZ, DTLZ1):
    pass


class IDTLZ2(InvertedDTLZ, DTLZ2):
    pass


class IDTLZ3(InvertedDTLZ, DTLZ3):
    pass


class IDTLZ4(InvertedDTLZ, DTLZ4):
    pass


class Eq1DTLZ1(DTLZ1, ConstrainedDTLZ):
    pass


class Eq1IDTLZ1(IDTLZ1, ConstrainedDTLZ):
    pass


class Eq1DTLZ2(DTLZ2, ConstrainedDTLZ):
    pass


class Eq1IDTLZ2(IDTLZ2, ConstrainedDTLZ):
    pass


class Eq1DTLZ3(DTLZ3, ConstrainedDTLZ):
    pass


class Eq1IDTLZ3(IDTLZ3, ConstrainedDTLZ):
    pass


class Eq1DTLZ4(DTLZ4, ConstrainedDTLZ):
    pass


class Eq1IDTLZ4(IDTLZ4, ConstrainedDTLZ):
    pass
