import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy import pi

from ..utils import timeit
from .base import UF8, ConstrainedMOOAnalytical


class CF1(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        self.n_ieq_constr = 1
        super().__init__()

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        x = jnp.clip(x, self.xl, self.xu)
        N = x.shape[0]
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        term1 = jnp.tile(x[:, [0]], (1, len(J1))) ** (
            0.5 * (1 + 3 * (jnp.tile(J1 + 1, (N, 1)) - 2) / (D - 2))
        )
        term2 = jnp.tile(x[:, [0]], (1, len(J2))) ** (
            0.5 * (1 + 3 * (jnp.tile(J2 + 1, (N, 1)) - 2) / (D - 2))
        )
        F1 = x[:, 0] + 2 * jnp.mean((x[:, J1] - term1) ** 2, 1)
        F2 = 1 - x[:, 0] + 2 * jnp.mean((x[:, J2] - term2) ** 2, 1)
        return jnp.hstack([F1, F2])

    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.atleast_2d(self._objective(x))
        return 1 - y[:, 0] - y[:, 1] + jnp.abs(jnp.sin(10 * jnp.pi * (y[:, 0] - y[:, 1] + 1)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        return np.c_[f, 1 - f]


class CF2(ConstrainedMOOAnalytical):
    """Constrained Problem 2 in CEC 09 benchmark

    Reference:
        Zhang, Qingfu, Aimin Zhou, Shizheng Zhao, Ponnuthurai Nagaratnam Suganthan, Wudong Liu,
        and Santosh Tiwari. "Multiobjective optimization test instances for the CEC 2009 special
        session and competition." (2008): 1-30.
    """

    def __init__(self, n_var: int = 10) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 1]
        self.xu = np.ones(self.n_var)
        self.n_ieq_constr = 1
        super().__init__()

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        x = jnp.clip(x, self.xl, self.xu)
        N = x.shape[0]
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        term1 = (
            x[:, J1] - jnp.sin(6 * pi * jnp.tile(x[:, [0]], (1, len(J1))) + jnp.tile(J1 + 1, (N, 1)) * pi / D)
        ) ** 2
        term2 = (
            x[:, J2] - jnp.cos(6 * pi * jnp.tile(x[:, [0]], (1, len(J2))) + jnp.tile(J2 + 1, (N, 1)) * pi / D)
        ) ** 2
        return jnp.hstack(
            [
                x[:, 0] + 2 * jnp.mean(term1, 1),
                1 - jnp.sqrt(x[:, 0]) + 2 * jnp.mean(term2, 1),
            ]
        )

    @timeit
    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        # TODO: this function is calling the _objective. Figure out a more efficient impplementation
        y = jnp.atleast_2d(self._objective(x))
        t = y[:, 1] + jnp.sqrt(y[:, 0]) - jnp.sin(2 * jnp.pi * (jnp.sqrt(y[:, 0]) - y[:, 1] + 1)) - 1
        return -t / (1 + jnp.exp(4 * jnp.abs(t)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        idx = (1 / 16 <= f) & (f <= 4 / 16) | (9 / 16 <= f) & (f <= 1)
        f = f[idx]
        return np.c_[f, 1 - jnp.sqrt(f)]


class CF3(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 1
        super().__init__()

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        x = jnp.clip(x, self.xl, self.xu)
        N = x.shape[0]
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        Y = x - jnp.sin(6 * pi * jnp.tile(x[:, 0], (1, D)) + jnp.tile(jnp.arange(1, D + 1), (N, 1)) * pi / D)
        term1 = jnp.prod(jnp.cos(20 * Y[:, J1] * pi / jnp.sqrt(jnp.tile(J1 + 1, (N, 1)))), 1)
        term2 = jnp.prod(jnp.cos(20 * Y[:, J2] * pi / jnp.sqrt(jnp.tile(J2 + 1, (N, 1)))), 1)
        F1 = x[:, 0] + 2 / len(J1) * (4 * jnp.sum(Y[:, J1] ** 2, 1) - 2 * term1 + 2)
        F2 = 1 - x[:, 0] ** 2 + 2 / len(J2) * (4 * jnp.sum(Y[:, J2] ** 2, 1) - 2 * term2 + 2)
        return jnp.hstack([F1, F2])

    @timeit
    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        # TODO: this function is calling `_objective`. Figure out a more efficient impplementation
        y = jnp.atleast_2d(self._objective(x))
        return 1 - y[:, 1] - y[:, 0] ** 2 + jnp.sin(2 * pi * (y[:, 0] ** 2 - y[:, 1] + 1))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        idx = (0 < f) & (f < 1 / 2) | (np.sqrt(1 / 2) < f) & (f < np.sqrt(3 / 4))
        f = f[~idx]
        return np.c_[f, 1 - f**2]


class CF4(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 1
        super().__init__()

    @timeit
    def _objective(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        X = jnp.clip(X, self.xl, self.xu)
        N = X.shape[0]
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        Y = X - jnp.sin(6 * pi * jnp.tile(X[:, 0], (1, D)) + jnp.tile(jnp.arange(1, D + 1), (N, 1)) * pi / D)
        temp = Y[:, 1] < 3 / 2 * (1 - jnp.sqrt(1 / 2))
        h = jax.numpy.where(
            temp[0],
            jnp.atleast_2d(jnp.concatenate([Y[:, 0] ** 2, abs(Y[:, 1]), (Y[:, 2:] ** 2).ravel()])),
            jnp.atleast_2d(jnp.concatenate([Y[:, 0] ** 2, jnp.array([0.125]), (Y[:, 2:] ** 2).ravel()])),
        )
        F1 = X[:, 0] + jnp.sum(h[:, J1], 1)
        F2 = 1 - X[:, 0] + jnp.sum(h[:, J2], 1)
        return jnp.hstack([F1, F2])

    @timeit
    def _ieq_constraint(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        D = self.n_var
        t = X[:, 1] - jnp.sin(6 * jnp.pi * X[:, 0] + 2 * jnp.pi / D) - 0.5 * X[:, 0] + 0.25
        return -t / (1 + jnp.exp(4 * abs(t)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = 1 - f1
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = -0.5 * f1[idx1] + 3 / 4
        f2[idx2] = 1 - f1[idx2] + 0.125
        return np.c_[f1, f2]


class CF5(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 1
        super().__init__()

    @timeit
    def _objective(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        N = X.shape[0]
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        Y1 = X[:, J1] - 0.8 * jnp.tile(X[:, 0], (1, len(J1))) * jnp.cos(
            6 * pi * jnp.tile(X[:, 0], (1, len(J1))) + jnp.tile(J1 + 1, (N, 1)) * pi / D
        )
        Y2 = X[:, J2] - 0.8 * jnp.tile(X[:, 0], (1, len(J2))) * jnp.sin(
            6 * pi * jnp.tile(X[:, 0], (1, len(J2))) + jnp.tile(J2 + 1, (N, 1)) * pi / D
        )
        h1 = 2 * Y1**2 - jnp.cos(4 * pi * Y1) + 1
        h2 = 2 * Y2[:, 1:] ** 2 - jnp.cos(4 * pi * Y2[:, 1:]) + 1
        temp = Y2[:, 0] < 3 / 2 * (1 - jnp.sqrt(1 / 2))
        h3 = jax.numpy.where(
            temp[0],
            jnp.abs(Y2[:, 0]),
            0.125 + (Y2[:, 0] - 1) ** 2,
        )
        F1 = X[:, 0] + jnp.sum(h1, 1)
        F2 = 1 - X[:, 0] + jnp.sum(h2, 1) + h3
        return jnp.hstack([F1, F2])

    @timeit
    def _ieq_constraint(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        D = self.n_var
        return -X[:, 1] + 0.8 * X[:, 0] * jnp.sin(6 * pi * X[:, 0] + 2 * pi / D) + 0.5 * X[:, 0] - 0.25

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = 1 - f1
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = -0.5 * f1[idx1] + 3 / 4
        f2[idx2] = 1 - f1[idx2] + 0.125
        return np.c_[f1, f2]


class CF6(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 2
        super().__init__()

    @timeit
    def _objective(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        Y1 = X[:, J1] - 0.8 * jnp.tile(X[:, 0], (1, len(J1))) * jnp.cos(
            6 * pi * jnp.tile(X[:, 0], (1, len(J1))) + (J1 + 1) * pi / D
        )
        Y2 = X[:, J2] - 0.8 * jnp.tile(X[:, 0], (1, len(J2))) * jnp.sin(
            6 * pi * jnp.tile(X[:, 0], (1, len(J2))) + (J2 + 1) * pi / D
        )
        return jnp.hstack([X[:, 0] + jnp.sum(Y1**2, 1), (1 - X[:, 0]) ** 2 + jnp.sum(Y2**2, 1)])

    @timeit
    def _ieq_constraint(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        D = self.n_var
        G1 = (
            -X[:, 1]
            + 0.8 * X[:, 0] * jnp.sin(6 * pi * X[:, 0] + 2 * pi / D)
            + jnp.sign(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2)
            * jnp.sqrt(abs(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2))
        )
        G2 = (
            -X[:, 3]
            + 0.8 * X[:, 0] * jnp.sin(6 * pi * X[:, 0] + 4 * pi / D)
            + jnp.sign(0.25 * jnp.sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0]))
            * jnp.sqrt(abs(0.25 * jnp.sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0])))
        )
        return jnp.hstack([G1, G2])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = (1 - f1) ** 2
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = 0.5 * (1 - f1[idx1])
        f2[idx2] = 0.25 * np.sqrt(1 - f1[idx2])
        return np.c_[f1, f2]


class CF7(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 2
        super().__init__()

    @timeit
    def _objective(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        D = self.n_var
        J1 = jnp.arange(3, self.n_var, 2) - 1
        J2 = jnp.arange(2, self.n_var + 2, 2) - 1
        Y1 = X[:, J1] - jnp.cos(6 * pi * jnp.tile(X[:, 0], (1, len(J1))) + (J1 + 1) * pi / D)
        Y2 = X[:, J2] - jnp.sin(6 * pi * jnp.tile(X[:, 0], (1, len(J2))) + (J2 + 1) * pi / D)
        h1 = 2 * Y1**2 - jnp.cos(4 * pi * Y1) + 1
        h2 = 2 * Y2[:, 2:] ** 2 - jnp.cos(4 * pi * Y2[:, 2:]) + 1
        h3 = Y2[:, :2] ** 2
        return jnp.hstack([X[:, 0] + jnp.sum(h1, 1), (1 - X[:, 0]) ** 2 + jnp.sum(h2, 1) + jnp.sum(h3, 1)])

    @timeit
    def _ieq_constraint(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        D = self.n_var
        G1 = (
            -X[:, 1]
            + jnp.sin(6 * pi * X[:, 0] + 2 * pi / D)
            + jnp.sign(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2)
            * jnp.sqrt(abs(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2))
        )
        G2 = (
            -X[:, 3]
            + jnp.sin(6 * pi * X[:, 0] + 4 * pi / D)
            + jnp.sign(0.25 * jnp.sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0]))
            * jnp.sqrt(abs(0.25 * jnp.sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0])))
        )
        return jnp.hstack([G1, G2])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = (1 - f1) ** 2
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = 0.5 * (1 - f1[idx1])
        f2[idx2] = 0.25 * np.sqrt(1 - f1[idx2])
        return np.c_[f1, f2]


class CF8(UF8, ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_ieq_constr = 1
        self.n_obj = 3
        self.n_var = n_var
        super().__init__(n_var)
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 4]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 4]

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        jnp.clip(x, self.xl, self.xu)
        return super()._objective(x)

    @timeit
    def _ieq_constraint(self, x: jnp.ndarray) -> float:
        # TODO: avoid calling the constraint fucntion two times
        F = self._objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + 4 * abs(jnp.sin(2 * pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1)))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        N = int(np.ceil(N / 5) * 5)
        R = np.zeros((N, 3))
        R[:, 2] = np.tile(np.sin(np.linspace(0, 1, num=int(N / 5)) * pi / 2), (5,))
        for i in range(0, 5):
            R[int(i * N / 5) : int((i + 1) * N / 5), 0] = np.sqrt(
                i / 4 * (1 - R[int(i * N / 5) : int((i + 1) * N / 5), 2] ** 2)
            )
        R[:, 1] = np.sqrt(np.max(1 - R[:, 0] ** 2 - R[:, 2] ** 2, 0))
        return R


class CF9(CF8, ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_var = n_var
        super().__init__(n_var)
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 2]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 2]

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return super()._objective(x)

    @timeit
    def _ieq_constraint(self, x: jnp.ndarray) -> float:
        F = self._objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + 3 * jnp.sin(2 * jnp.pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        R = np.random.rand(N, 3)
        R = R / np.sqrt(np.sum(R**2, axis=1)).reshape(-1, 1)
        idx = (1e-5 < R[:, 0]) & (R[:, 0] < np.sqrt((1 - R[:, 2] ** 2) / 4)) | (
            np.sqrt((1 - R[:, 2] ** 2) / 2) < R[:, 0]
        ) & (R[:, 0] < np.sqrt(3 * (1 - R[:, 2] ** 2) / 4))
        return R[~idx]


class CF10(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        self.n_var = n_var
        self.n_ieq_constr = 1
        self.n_obj = 3
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 2]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 2]
        super().__init__()

    @timeit
    def _objective(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        N = X.shape[0]
        D = self.n_var
        J1 = jnp.arange(4, D + 1, 3) - 1
        J2 = jnp.arange(5, D + 1, 3) - 1
        J3 = jnp.arange(3, D + 1, 3) - 1
        Y = X - 2 * jnp.tile(X[:, [1]], (1, D)) * jnp.sin(
            2 * pi * jnp.tile(X[:, [0]], (1, D)) + jnp.tile(jnp.arange(D) + 1, (N, 1)) * pi / D
        )
        F1 = jnp.cos(0.5 * X[:, 0] * pi) * jnp.cos(0.5 * X[:, 1] * pi) + 2 * jnp.mean(
            4 * Y[:, J1] ** 2 - jnp.cos(8 * pi * Y[:, J1]) + 1, 1
        )
        F2 = jnp.cos(0.5 * X[:, 0] * pi) * jnp.sin(0.5 * X[:, 1] * pi) + 2 * jnp.mean(
            4 * Y[:, J2] ** 2 - jnp.cos(8 * pi * Y[:, J2]) + 1, 1
        )
        F3 = jnp.sin(0.5 * X[:, 0] * pi) + 2 * jnp.mean(4 * Y[:, J3] ** 2 - jnp.cos(8 * pi * Y[:, J3]) + 1, 1)
        return jnp.hstack([F1, F2, F3])

    @timeit
    def _ieq_constraint(self, x: jnp.ndarray) -> float:
        F = self._objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + jnp.sin(2 * pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        R = np.random.rand(N, 3)
        R = R / np.sqrt(np.sum(R**2, axis=1)).reshape(-1, 1)
        idx = (1e-5 < R[:, 0]) & (R[:, 0] < np.sqrt((1 - R[:, 2] ** 2) / 4)) | (
            np.sqrt((1 - R[:, 2] ** 2) / 2) < R[:, 0]
        ) & (R[:, 0] < np.sqrt(3 * (1 - R[:, 2] ** 2) / 4))
        return R[~idx]
