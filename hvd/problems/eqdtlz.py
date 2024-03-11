import jax.numpy as jnp
import numpy as np

from ..utils import timeit
from .base import ConstrainedMOOAnalytical, MOOAnalytical

# TODO: unify the DTLZ classes here with the ones in `dtlz.py`

class _DTLZ(MOOAnalytical):
    def __init__(self, n_obj: int = 3, n_var: int = 11, boundry_constraints: bool = False):
        self.n_obj = n_obj
        # the default decision space of 11-dimensional
        self.n_var = n_var if n_var is not None else self.n_obj + 8
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        super().__init__(boundry_constraints=boundry_constraints)

    def get_pareto_set(self, N: int = 1000, kind="uniform") -> np.ndarray:
        M = self.n_obj
        theta = (
            np.sort(np.random.rand(N) * 2 * np.pi)
            if kind == "uniform"
            else np.r_[np.linspace(0, 2 * np.pi, N - 1), 0]
        )
        x = np.c_[np.cos(theta), np.sin(theta)] * 0.4
        return np.c_[x + 0.5, np.tile(0.5, (N, self.n_var - M + 1))]

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        x = self.get_pareto_set(N)
        y = np.array([self._objective(xx) for xx in x])
        return y


class DTLZ1(_DTLZ, ConstrainedMOOAnalytical):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        D = len(x)
        M = self.n_obj
        g = 100 * (D - M + 1 + jnp.sum((x[M - 1 :] - 0.5) ** 2 - jnp.cos(20.0 * jnp.pi * (x[M - 1 :] - 0.5))))
        return (
            0.5 * (1 + g) * jnp.cumprod(jnp.r_[[1], x[0 : M - 1]])[::-1] * jnp.r_[[1], 1 - x[0 : M - 1][::-1]]
        )


class IDTLZ1(_DTLZ, ConstrainedMOOAnalytical):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        D = len(x)
        M = self.n_obj
        g = 100 * (D - M + 1 + jnp.sum((x[M - 1 :] - 0.5) ** 2 - jnp.cos(20.0 * jnp.pi * (x[M - 1 :] - 0.5))))
        return (1 + g) / 2 - 0.5 * (1 + g) * jnp.cumprod(jnp.r_[[1], x[0 : M - 1]])[::-1] * jnp.r_[
            [1], 1 - x[0 : M - 1][::-1]
        ]
    # @timeit
    # def _objective(self, x: np.ndarray) -> np.ndarray:
    #     D = len(x)
    #     M = self.n_obj
    #     g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
    #     return (1 + g) / 2 - 0.5 * (1 + g) * np.cumprod(np.r_[[1], x[0 : M - 1]])[::-1] * np.r_[
    #         [1], 1 - x[0 : M - 1][::-1]
    #     ]

class DTLZ2(_DTLZ, ConstrainedMOOAnalytical):

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        M = self.n_obj
        g = jnp.sum((x[M - 1 :] - 0.5) ** 2)
        return (
            (1 + g)
            * jnp.cumprod(jnp.r_[[1], jnp.cos(x[0 : M - 1] * jnp.pi / 2)])[::-1]
            * jnp.r_[[1], jnp.sin(x[0 : M - 1][::-1] * jnp.pi / 2)]
        )


class IDTLZ2(_DTLZ, ConstrainedMOOAnalytical):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        M = self.n_obj
        g = jnp.sum((x[M - 1 :] - 0.5) ** 2)
        return (1 + g) / 2 - (
            (1 + g)
            * jnp.cumprod(jnp.r_[[1], jnp.cos(x[0 : M - 1] * jnp.pi / 2)])[::-1]
            * jnp.r_[[1], jnp.sin(x[0 : M - 1][::-1] * jnp.pi / 2)]
        )

    # @timeit
    # def _objective(self, x: np.ndarray) -> np.ndarray:
    #     M = self.n_obj
    #     g = np.sum((x[M - 1 :] - 0.5) ** 2)
    #     return (1 + g) / 2 - (
    #         (1 + g)
    #         * np.cumprod(np.r_[[1], np.cos(x[0 : M - 1] * np.pi / 2)])[::-1]
    #         * np.r_[[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)]
    #     )


class DTLZ3(_DTLZ, ConstrainedMOOAnalytical):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        M = self.n_obj
        D = len(x)
        g = 100 * (D - M + 1 + jnp.sum((x[M - 1 :] - 0.5) ** 2 - jnp.cos(20.0 * jnp.pi * (x[M - 1 :] - 0.5))))
        return (
            (1 + g)
            * jnp.cumprod(jnp.r_[[1], jnp.cos(x[0 : M - 1] * jnp.pi / 2)])[::-1]
            * jnp.r_[[1], jnp.sin(x[0 : M - 1][::-1] * jnp.pi / 2)]
        )


class IDTLZ3(_DTLZ, ConstrainedMOOAnalytical):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        M = self.n_obj
        D = len(x)
        g = 100 * (D - M + 1 + jnp.sum((x[M - 1 :] - 0.5) ** 2 - jnp.cos(20.0 * jnp.pi * (x[M - 1 :] - 0.5))))
        return (1 + g) / 2 - (1 + g) * jnp.cumprod(jnp.r_[[1], jnp.cos(x[0 : M - 1] * jnp.pi / 2)])[
            ::-1
        ] * jnp.r_[[1], jnp.sin(x[0 : M - 1][::-1] * jnp.pi / 2)]
    
    # @timeit
    # def _objective(self, x: np.ndarray) -> np.ndarray:
    #     M = self.n_obj
    #     D = len(x)
    #     g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
    #     return (1 + g) / 2 - (1 + g) * np.cumprod(np.r_[[1], np.cos(x[0 : M - 1] * np.pi / 2)])[
    #         ::-1
    #     ] * np.r_[[1], np.sin(x[0 : M - 1][::-1] * np.pi / 2)]


class DTLZ4(_DTLZ, ConstrainedMOOAnalytical):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        M = self.n_obj
        x_ = x[0 : M - 1] ** 100
        g = jnp.sum((x[M - 1 :] - 0.5) ** 2)
        return (
            (1 + g)
            * jnp.cumprod(jnp.r_[[1], jnp.cos(x_ * jnp.pi / 2)])[::-1]
            * jnp.r_[[1], jnp.sin(x_[::-1] * jnp.pi / 2)]
        )


class IDTLZ4(_DTLZ, ConstrainedMOOAnalytical):
    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        M = self.n_obj
        x_ = x[0 : M - 1] ** 100
        g = jnp.sum((x[M - 1 :] - 0.5) ** 2)
        return (1 + g) / 2 - (1 + g) * jnp.cumprod(jnp.r_[[1], jnp.cos(x_ * jnp.pi / 2)])[::-1] * jnp.r_[
            [1], jnp.sin(x_[::-1] * jnp.pi / 2)
        ]

    # @timeit
    # def _objective(self, x: np.ndarray) -> np.ndarray:
    #     M = self.n_obj
    #     x_ = x[0 : M - 1] ** 100
    #     g = np.sum((x[M - 1 :] - 0.5) ** 2)
    #     return (1 + g) / 2 - (1 + g) * np.cumprod(np.r_[[1], np.cos(x_ * np.pi / 2)])[::-1] * np.r_[
    #         [1], np.sin(x_[::-1] * np.pi / 2)
    #     ]

# TODO: somehow simplify the way of defining the following classes since the constraints are the same
class Eq1DTLZ1(DTLZ1):
    n_eq_constr = 1

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4


class Eq1IDTLZ1(IDTLZ1):
    n_eq_constr = 1

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4
    

class Eq1DTLZ2(DTLZ2):
    n_eq_constr = 1

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4


class Eq1IDTLZ2(IDTLZ2):
    n_eq_constr = 1 

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4
    

class Eq1DTLZ3(DTLZ3):
    n_eq_constr = 1

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4


class Eq1IDTLZ3(IDTLZ3):
    n_eq_constr = 1

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4


class Eq1DTLZ4(DTLZ4):
    n_eq_constr = 1

    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4
    

class Eq1IDTLZ4(IDTLZ4):
    n_eq_constr = 1
    
    @timeit
    def _eq_constraint(self, x: jnp.ndarray) -> float:
        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return jnp.abs(jnp.sum(xx**2) - r**2) - 1e-4







