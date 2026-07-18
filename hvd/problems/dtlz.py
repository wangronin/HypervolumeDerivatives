from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from pymoo.util.remote import Remote

from ..utils import timeit
from .base import CMOP
from .reference import generic_sphere, get_ref_dirs

jax.config.update("jax_enable_x64", True)
# NOTE: `eps` is to cap the decision variables below for DTLZ6 since it is not differentiable at x = 0
eps = 1e-30


class _DTLZ(CMOP):
    def __init__(
        self,
        n_var: int | None = 11,
        n_obj: int = 3,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
        boundary_constraints: bool = False,
        scale: float = 1,
        alpha: float = 1,
    ):
        n_var = n_var if n_var is not None else n_obj + 8
        self.k: int = n_var - n_obj + 1
        self.scale: float = scale
        self.alpha: float = alpha
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=0.0 if xl is None else xl,
            xu=1.0 if xu is None else xu,
            n_eq_constr=self.default_n_eq_constr,
            n_ieq_constr=self.default_n_ieq_constr,
            boundary_constraints=boundary_constraints,
        )

    def _transform_x(self, x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.cos(jnp.power(x, self.alpha) * jnp.pi / 2), jnp.sin(jnp.power(x, self.alpha) * jnp.pi / 2)

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x_, x_M = x[: self.n_obj - 1], x[self.n_obj - 1 :]
        g = self.g(x_M)
        x1, x2 = self._transform_x(x_, g)
        return self.scale * (1 + g) * jnp.cumprod(jnp.r_[[1], x1])[::-1] * jnp.r_[[1], x2[::-1]]

    def g(self, x_M: jnp.ndarray) -> float:
        return jnp.sum(jnp.square(x_M - 0.5))

    def get_pareto_set(self, N: int = 1000) -> np.ndarray:
        x = np.random.rand(N, self.n_var - self.k)
        return np.c_[x, np.tile(0.5, (N, self.k))]

    def get_pareto_front(self, **kwargs) -> np.ndarray:
        return NotImplemented


class DTLZ1(_DTLZ):
    def __init__(
        self,
        n_var: int | None = 11,
        n_obj: int = 3,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
        boundary_constraints: bool = False,
    ) -> None:
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            scale=0.5,
            boundary_constraints=boundary_constraints,
        )

    def _transform_x(self, x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return x, 1 - x

    def g(self, x_M: jnp.ndarray) -> float:
        return 100 * (self.k + jnp.sum(jnp.square(x_M - 0.5) - jnp.cos(20 * jnp.pi * (x_M - 0.5))))

    def get_pareto_front(self, **kwargs) -> np.ndarray:
        ref_dirs = get_ref_dirs(self.n_obj)
        return 0.5 * ref_dirs


class DTLZ2(_DTLZ):
    def get_pareto_front(self, **kwargs) -> np.ndarray:
        ref_dirs = get_ref_dirs(self.n_obj)
        return generic_sphere(ref_dirs)


class DTLZ3(DTLZ2):
    def g(self, x_M: jnp.ndarray) -> float:
        return 100 * (self.k + jnp.sum(jnp.square(x_M - 0.5) - jnp.cos(20 * jnp.pi * (x_M - 0.5))))


class DTLZ4(DTLZ2):
    def __init__(
        self,
        n_var: int | None = 11,
        n_obj: int = 3,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
        boundary_constraints: bool = False,
    ) -> None:
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            alpha=100,
            boundary_constraints=boundary_constraints,
        )


class DTLZ5(_DTLZ):
    def _transform_x(self, x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        theta = (1 + 2 * g * x) / (2 * (1 + g))
        theta = jnp.r_[x[0], theta[1:]]
        return jnp.cos(theta * jnp.pi / 2), jnp.sin(theta * jnp.pi / 2)

    def get_pareto_front(self):
        if self.n_obj == 3:
            # TODO: get rid of this remote call
            return Remote.get_instance().load("pymoo", "pf", "dtlz5-3d.pf")
        else:
            raise Exception("Not implemented yet.")


class DTLZ6(DTLZ5):
    def g(self, x_M: jnp.ndarray) -> float:
        x_M_ = jnp.clip(jnp.abs(x_M), eps)
        return jnp.sum(jnp.power(x_M_, 0.1))

    def get_pareto_set(self, N: int = 1000) -> np.ndarray:
        x = np.random.rand(N, self.n_var - self.k)
        return np.c_[x, np.tile(0, (N, self.k))]

    def get_pareto_front(self):
        if self.n_obj == 3:
            return Remote.get_instance().load("pymoo", "pf", "dtlz6-3d.pf")
        else:
            raise Exception("Not implemented yet.")


class DTLZ7(_DTLZ):
    def g(self, x_M: jnp.ndarray) -> float:
        return 1 + 9 / self.k * jnp.sum(x_M[-self.k :])

    def get_pareto_set(self, N: int = 1000) -> np.ndarray:
        x = np.random.rand(N, self.n_var - self.k)
        return np.c_[x, np.tile(0, (N, self.k))]

    def get_pareto_front(self):
        if self.n_obj == 3:
            return Remote.get_instance().load("pymoo", "pf", "dtlz7-3d.pf")
        else:
            raise Exception("Not implemented yet.")

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        f, x_M = x[: self.n_obj - 1], x[self.n_obj - 1 :]
        g = self.g(x_M)
        h = self.n_obj - jnp.sum(f / (1 + g) * (1 + jnp.sin(3 * jnp.pi * f)))
        return jnp.r_[f, (1 + g) * h]
