"""JAX-native continuous ZDT benchmark problems."""

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .base import MOP, fixed_n_obj

_EPS = 1e-12


class ZDT(MOP):
    default_n_obj = 2

    def __init__(
        self,
        n_var: int = 30,
        n_obj: int | None = None,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
    ) -> None:
        super().__init__(
            n_var=n_var,
            n_obj=fixed_n_obj(n_obj, self.default_n_obj, type(self).__name__),
            xl=0.0 if xl is None else xl,
            xu=1.0 if xu is None else xu,
        )

    def get_pareto_set(self, n_pareto_points: int = 100, kind: str = "linear") -> np.ndarray:
        x = np.linspace(0, 1, n_pareto_points) if kind == "linear" else np.random.random(n_pareto_points)
        return np.c_[x, np.zeros((n_pareto_points, self.n_var - 1))]


class ZDT1(ZDT):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        g = 1 + 9 * jnp.mean(x[1:])
        return jnp.array([x[0], g * (1 - jnp.sqrt(jnp.maximum(x[0] / g, _EPS)))])

    def get_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        f = np.linspace(0, 1, n_pareto_points)
        return np.c_[f, 1 - np.sqrt(f)]


class ZDT2(ZDT):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        g = 1 + 9 * jnp.mean(x[1:])
        return jnp.array([x[0], g * (1 - (x[0] / g) ** 2)])

    def get_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        f = np.linspace(0, 1, n_pareto_points)
        return np.c_[f, 1 - f**2]


class ZDT3(ZDT1):
    regions = np.array(
        [
            [0, 0.0830015349],
            [0.182228780, 0.2577623634],
            [0.4093136748, 0.4538821041],
            [0.6183967944, 0.6525117038],
            [0.8233317983, 0.8518328654],
        ]
    )

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        g = 1 + 9 * jnp.mean(x[1:])
        r = x[0] / g
        return jnp.array([x[0], g * (1 - jnp.sqrt(jnp.maximum(r, _EPS)) - r * jnp.sin(10 * jnp.pi * x[0]))])

    def get_pareto_front(self, n_points: int = 100, flatten: bool = True) -> np.ndarray:
        blocks = []
        for lo, hi in self.regions:
            f = np.linspace(lo, hi, max(1, n_points // len(self.regions)))
            blocks.append(np.c_[f, 1 - np.sqrt(f) - f * np.sin(10 * np.pi * f)])
        return np.row_stack(blocks) if flatten else np.stack(blocks)


class ZDT4(ZDT1):
    def __init__(
        self,
        n_var: int = 10,
        n_obj: int | None = None,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
    ) -> None:
        super().__init__(
            n_var,
            n_obj=n_obj,
            xl=np.r_[0.0, np.full(n_var - 1, -5.0)] if xl is None else xl,
            xu=np.r_[1.0, np.full(n_var - 1, 5.0)] if xu is None else xu,
        )

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        g = 1 + 10 * (self.n_var - 1) + jnp.sum(x[1:] ** 2 - 10 * jnp.cos(4 * jnp.pi * x[1:]))
        return jnp.array([x[0], g * (1 - jnp.sqrt(jnp.maximum(x[0] / g, _EPS)))])


class ZDT6(ZDT):
    def __init__(
        self,
        n_var: int = 10,
        n_obj: int | None = None,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
    ) -> None:
        super().__init__(n_var, n_obj=n_obj, xl=xl, xu=xu)

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        f1 = 1 - jnp.exp(-4 * x[0]) * jnp.sin(6 * jnp.pi * x[0]) ** 6
        g = 1 + 9 * jnp.maximum(jnp.mean(x[1:]), _EPS) ** 0.25
        return jnp.array([f1, g * (1 - (f1 / g) ** 2)])

    def get_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        f = np.linspace(0.2807753191, 1, n_pareto_points)
        return np.c_[f, 1 - f**2]
