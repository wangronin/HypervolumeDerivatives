import os

# enable double-precision of JAX
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import timeit
from .base import ConstrainedMOOAnalytical, MOOAnalytical


class DENT(MOOAnalytical):
    def __init__(self, **kwargs):
        self.n_obj = 2
        self.n_var = 2
        self.xl = -2 * np.ones(self.n_var)
        self.xu = 2 * np.ones(self.n_var)
        super().__init__(**kwargs)

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        term1 = jnp.sqrt(1 + (x[0] + x[1]) ** 2) + jnp.sqrt(1 + (x[0] - x[1]) ** 2)
        term2 = 0.85 * jnp.exp(-((x[0] - x[1]) ** 2))
        y1 = 0.5 * (term1 + x[0] - x[1]) + term2
        y2 = 0.5 * (term1 - x[0] + x[1]) + term2
        return jnp.array([y1, y2])

    def get_pareto_set(self, n: int = 100) -> np.ndarray:
        x = np.linspace(-2, 2, n)
        return np.c_[x, -x]

    def get_pareto_front(self, n: int = 100) -> np.ndarray:
        X = self.get_pareto_set(n)
        return np.array([self.objective(x) for x in X])


class CONV3(ConstrainedMOOAnalytical):
    def __init__(self, **kwargs):
        self.n_obj = 3
        self.n_var = 3
        self.xl = -3 * np.ones(self.n_var)
        self.xu = 3 * np.ones(self.n_var)
        self.a1 = -1 * np.ones(self.n_var)
        self.a2 = np.ones(self.n_var)
        self.a3 = np.r_[-1 * np.ones(self.n_var - 1), 1]
        super().__init__(**kwargs)

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        func = lambda x, a: jnp.sum((x - a) ** 2)
        return jnp.array([func(x, self.a1), func(x, self.a2), func(x, self.a3)])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        w = np.random.rand(N, 3)
        w /= w.sum(axis=1).reshape(-1, 1)
        X = w @ np.vstack([self.a1, self.a2, self.a3])
        return np.array([self._objective(x) for x in X])


class CONV4(ConstrainedMOOAnalytical):
    """Convex Problem 4"""

    def __init__(self, **kwargs):
        self.n_obj = 4
        self.n_var = 4
        self.xl = -10 * np.ones(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        self.centers = np.eye(self.n_var)
        super().__init__(**kwargs)

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([jnp.sum((x - self.centers[i]) ** 2) for i in range(self.n_var)])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        w = np.random.rand(N, 4)
        w /= w.sum(axis=1).reshape(-1, 1)
        X = w @ np.eye(self.n_var)  # the positive part
        return np.array([self._objective(x) for x in X])


class CONV4_2F(ConstrainedMOOAnalytical):
    """Convex Problem 4 with 2 disconnected Pareto fronts"""

    def __init__(self, **kwargs):
        self.n_obj = 4
        self.n_var = 4
        self.xl = -10 * np.ones(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        super().__init__(**kwargs)

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        a = jnp.eye(self.n_var)
        deltaa = jnp.ones(self.n_var)
        fa4 = jnp.array([2, 2, 2, 0])
        fa1 = jnp.array([0, 2, 2, 2])
        deltay = fa4 - fa1
        z = x + deltaa
        y = jax.lax.select(
            jnp.all(x < 0),
            jnp.array([jnp.sum((z - a[i]) ** 2) - 3.5 * deltay[i] for i in range(4)]),
            jnp.array([jnp.sum((x - a[i]) ** 2) for i in range(4)]),
        )
        return y

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        N1 = int(np.ceil(N / 2))
        N2 = max(1, N - N1)
        w1 = np.random.rand(N1, 4)
        w1 /= w1.sum(axis=1).reshape(-1, 1)
        w2 = np.random.rand(N2, 4)
        w2 /= w2.sum(axis=1).reshape(-1, 1)
        X_1 = w1 @ np.eye(self.n_var)  # the positive part
        X_2 = w2 @ (np.eye(self.n_var) - np.ones((self.n_var, self.n_var)))  # the negative part
        return np.array([self._objective(x) for x in np.vstack([X_1, X_2])])


class UF7(MOOAnalytical):
    def __init__(self, n_var: int = 30) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 1]
        self.xu = np.ones(self.n_var)
        self.encoding = np.ones(self.n_var)
        super().__init__()

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        N = x.shape[0]
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        y = x - jnp.sin(
            6 * jnp.pi * jnp.tile(x[:, [0]], (1, D)) + jnp.tile(jnp.arange(D) + 1, (N, 1)) * jnp.pi / D
        )
        return jnp.hstack(
            [
                x[:, 0] ** 0.2 + 2 * jnp.mean(y[:, J1] ** 2, 1),
                1 - x[:, 0] ** 0.2 + 2 * jnp.mean(y[:, J2] ** 2, 1),
            ]
        ).T

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        return np.c_[f, 1 - f]


class UF8(MOOAnalytical):
    def __init__(self, n_var: int = 30, **kwargs) -> None:
        self.n_obj = 3
        self.n_var = n_var
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 2]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 2]
        self.encoding = np.ones(self.n_var)
        super().__init__()

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        N = x.shape[0]
        D = self.n_var
        J1 = jnp.arange(4, D + 1, 3) - 1
        J2 = jnp.arange(5, D + 1, 3) - 1
        J3 = jnp.arange(3, D + 1, 3) - 1
        y = x - 2 * jnp.tile(x[:, [1]], (1, D)) * jnp.sin(
            2 * jnp.pi * jnp.tile(x[:, [0]], (1, D)) + jnp.tile(jnp.arange(D) + 1, (N, 1)) * jnp.pi / D
        )
        return jnp.hstack(
            [
                jnp.cos(0.5 * x[:, 0] * jnp.pi) * jnp.cos(0.5 * x[:, 1] * jnp.pi)
                + 2 * jnp.mean(y[:, J1] ** 2, 1),
                jnp.cos(0.5 * x[:, 0] * jnp.pi) * jnp.sin(0.5 * x[:, 1] * jnp.pi)
                + 2 * jnp.mean(y[:, J2] ** 2, 1),
                jnp.sin(0.5 * x[:, 0] * jnp.pi) + 2 * jnp.mean(y[:, J3] ** 2, 1),
            ]
        ).T


class DISCONNECTED(ConstrainedMOOAnalytical):
    def __init__(self, **kwargs):
        self.n_obj = 2
        self.n_var = 2
        self.n_ieq_constr = 2
        self.n_eq_constr = 1
        self.xl = np.array([0, -8])
        self.xu = np.array([1, 1])
        super().__init__(**kwargs)

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        g1 = jax.numpy.where(x[0] <= 0.5, 0.1 - x[0], 0.6 + 1e-2 - x[0])
        g2 = jax.numpy.where(x[0] <= 0.5, x[0] - 0.4, x[0] - 0.8)
        return jnp.array([g1, g2])

    def _eq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.numpy.where(
            x[0] <= 0.6,
            (2.5 * x[0]) ** 2 - 5 * x[0] + 1 - x[1],
            1 - (2.5 * x[0]) ** 2 - 5 - x[1],
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        pass
