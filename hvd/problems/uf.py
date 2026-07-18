"""CEC 2009 unconstrained UF test problems, implemented with JAX."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .base import CMOP, fixed_n_obj
from .reference import generic_sphere, get_ref_dirs


class _UF2D(CMOP):
    default_n_obj = 2
    _default_lower = -1.0
    _default_upper = 1.0

    def __init__(
        self,
        n_var: int = 30,
        n_obj: int | None = None,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
        boundary_constraints: bool = False,
    ) -> None:
        if n_var < 3:
            raise ValueError("UF1--UF7 require n_var >= 3")
        if xl is None:
            xl = np.r_[0.0, np.full(n_var - 1, self._default_lower)]
        if xu is None:
            xu = np.r_[1.0, np.full(n_var - 1, self._default_upper)]
        self._indices = jnp.arange(1, n_var + 1)
        self._odd = jnp.arange(2, n_var, 2)  # MATLAB 3,5,...
        self._even = jnp.arange(1, n_var, 2)  # MATLAB 2,4,...
        super().__init__(
            n_var=n_var,
            n_obj=fixed_n_obj(n_obj, self.default_n_obj, type(self).__name__),
            xl=xl,
            xu=xu,
            boundary_constraints=boundary_constraints,
        )

    def _sine_y(self, x: jnp.ndarray) -> jnp.ndarray:
        return x - jnp.sin(6 * jnp.pi * x[0] + self._indices * jnp.pi / self.n_var)

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        return np.c_[f, self._front(f)]


class UF1(_UF2D):
    @staticmethod
    def _front(f: np.ndarray) -> np.ndarray:
        return 1 - np.sqrt(f)

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._sine_y(x)
        return jnp.array(
            [x[0] + 2 * jnp.mean(y[self._odd] ** 2), 1 - jnp.sqrt(x[0]) + 2 * jnp.mean(y[self._even] ** 2)]
        )


class UF2(UF1):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        def residual(idx: jnp.ndarray, trig: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            phase = 6 * jnp.pi * x[0] + (idx + 1) * jnp.pi / self.n_var
            envelope = (
                0.3 * x[0] ** 2 * jnp.cos(24 * jnp.pi * x[0] + 4 * (idx + 1) * jnp.pi / self.n_var)
                + 0.6 * x[0]
            )
            return x[idx] - envelope * trig(phase)

        y1, y2 = residual(self._odd, jnp.cos), residual(self._even, jnp.sin)
        return jnp.array([x[0] + 2 * jnp.mean(y1**2), 1 - jnp.sqrt(x[0]) + 2 * jnp.mean(y2**2)])


class UF3(UF1):
    _default_lower = 0.0

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        exponent = 0.5 * (1 + 3 * (self._indices - 2) / (self.n_var - 2))
        y = x - x[0] ** exponent

        def penalty(idx: jnp.ndarray) -> jnp.ndarray:
            z = y[idx]
            return (
                2
                / idx.size
                * (4 * jnp.sum(z**2) - 2 * jnp.prod(jnp.cos(20 * jnp.pi * z / jnp.sqrt(idx + 1))) + 2)
            )

        return jnp.array([x[0] + penalty(self._odd), 1 - jnp.sqrt(x[0]) + penalty(self._even)])


class UF4(_UF2D):
    _default_lower = -2.0
    _default_upper = 2.0

    @staticmethod
    def _front(f: np.ndarray) -> np.ndarray:
        return 1 - f**2

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        h = jnp.abs(self._sine_y(x)) / (1 + jnp.exp(2 * jnp.abs(self._sine_y(x))))
        return jnp.array([x[0] + 2 * jnp.mean(h[self._odd]), 1 - x[0] ** 2 + 2 * jnp.mean(h[self._even])])


class UF5(_UF2D):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._sine_y(x)
        h = 2 * y**2 - jnp.cos(4 * jnp.pi * y) + 1
        common = 0.15 * jnp.abs(jnp.sin(20 * jnp.pi * x[0]))
        return jnp.array(
            [x[0] + common + 2 * jnp.mean(h[self._odd]), 1 - x[0] + common + 2 * jnp.mean(h[self._even])]
        )

    def get_pareto_front(self, N: int = 21) -> np.ndarray:
        f = np.arange(21) / 20
        return np.c_[f, 1 - f]


class UF6(_UF2D):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._sine_y(x)

        def penalty(idx: jnp.ndarray) -> jnp.ndarray:
            z = y[idx]
            return (
                2
                / idx.size
                * (4 * jnp.sum(z**2) - 2 * jnp.prod(jnp.cos(20 * jnp.pi * z / jnp.sqrt(idx + 1))) + 2)
            )

        common = jnp.maximum(0.0, 0.7 * jnp.sin(4 * jnp.pi * x[0]))
        return jnp.array([x[0] + common + penalty(self._odd), 1 - x[0] + common + penalty(self._even)])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        keep = ~(((f > 0) & (f < 0.25)) | ((f > 0.5) & (f < 0.75)))
        return np.c_[f[keep], 1 - f[keep]]


class UF7(_UF2D):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._sine_y(x)
        a = x[0] ** 0.2
        return jnp.array([a + 2 * jnp.mean(y[self._odd] ** 2), 1 - a + 2 * jnp.mean(y[self._even] ** 2)])

    @staticmethod
    def _front(f: np.ndarray) -> np.ndarray:
        return 1 - f


class _UF3D(CMOP):
    default_n_obj = 3

    def __init__(
        self,
        n_var: int = 30,
        n_obj: int | None = None,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
        boundary_constraints: bool = False,
    ) -> None:
        if n_var < 5:
            raise ValueError("UF8--UF10 require n_var >= 5")
        if xl is None:
            xl = np.r_[0.0, 0.0, np.full(n_var - 2, -2.0)]
        if xu is None:
            xu = np.r_[1.0, 1.0, np.full(n_var - 2, 2.0)]
        self._indices = jnp.arange(1, n_var + 1)
        self._groups = (jnp.arange(3, n_var, 3), jnp.arange(4, n_var, 3), jnp.arange(2, n_var, 3))
        super().__init__(
            n_var=n_var,
            n_obj=fixed_n_obj(n_obj, self.default_n_obj, type(self).__name__),
            xl=xl,
            xu=xu,
            n_eq_constr=self.default_n_eq_constr,
            n_ieq_constr=self.default_n_ieq_constr,
            boundary_constraints=boundary_constraints,
        )

    def _y(self, x: jnp.ndarray) -> jnp.ndarray:
        return x - 2 * x[1] * jnp.sin(2 * jnp.pi * x[0] + self._indices * jnp.pi / self.n_var)

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        return generic_sphere(get_ref_dirs(3, n_points=N))


class UF8(_UF3D):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._y(x)
        p = [2 * jnp.mean(y[g] ** 2) for g in self._groups]
        a, b = 0.5 * jnp.pi * x[0], 0.5 * jnp.pi * x[1]
        return jnp.array([jnp.cos(a) * jnp.cos(b) + p[0], jnp.cos(a) * jnp.sin(b) + p[1], jnp.sin(a) + p[2]])


class UF9(_UF3D):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._y(x)
        p = [2 * jnp.mean(y[g] ** 2) for g in self._groups]
        a = jnp.maximum(0.0, 1.1 * (1 - 4 * (2 * x[0] - 1) ** 2))
        return jnp.array(
            [0.5 * (a + 2 * x[0]) * x[1] + p[0], 0.5 * (a - 2 * x[0] + 2) * x[1] + p[1], 1 - x[1] + p[2]]
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        r = get_ref_dirs(3, n_points=N)
        keep = ~((r[:, 0] > (1 - r[:, 2]) / 4) & (r[:, 0] < (1 - r[:, 2]) * 3 / 4))
        return r[keep]


class UF10(UF8):
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._y(x)
        h = 4 * y**2 - jnp.cos(8 * jnp.pi * y) + 1
        p = [2 * jnp.mean(h[g]) for g in self._groups]
        a, b = 0.5 * jnp.pi * x[0], 0.5 * jnp.pi * x[1]
        return jnp.array([jnp.cos(a) * jnp.cos(b) + p[0], jnp.cos(a) * jnp.sin(b) + p[1], jnp.sin(a) + p[2]])
