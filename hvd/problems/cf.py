import autograd.numpy as np
import numpy as _np
from autograd.numpy import abs, arange, cos, exp, mean, pi, prod, sign, sin, sqrt, sum, tile

from ..utils import timeit
from .base import UF8, ConstrainedMOOAnalytical


class CF1(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__()
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        self.n_ieq_constr = 1

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.clip(x, self.xl, self.xu)
        N = x.shape[0]
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
        term1 = np.tile(x[:, [0]], (1, len(J1))) ** (0.5 * (1 + 3 * (np.tile(J1 + 1, (N, 1)) - 2) / (D - 2)))
        term2 = np.tile(x[:, [0]], (1, len(J2))) ** (0.5 * (1 + 3 * (np.tile(J2 + 1, (N, 1)) - 2) / (D - 2)))
        F1 = x[:, 0] + 2 * np.mean((x[:, J1] - term1) ** 2, 1)
        F2 = 1 - x[:, 0] + 2 * np.mean((x[:, J2] - term2) ** 2, 1)
        return np.hstack([F1, F2])

    @timeit
    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        # TODO: this function is calling the objective. Figure out a more efficient impplementation
        y = np.atleast_2d(self.objective(x))
        return 1 - y[:, 0] - y[:, 1] + np.abs(np.sin(10 * np.pi * (y[:, 0] - y[:, 1] + 1)))

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
        super().__init__()
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 1]
        self.xu = np.ones(self.n_var)
        self.n_ieq_constr = 1

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.clip(x, self.xl, self.xu)
        N = x.shape[0]
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
        term1 = (x[:, J1] - sin(6 * pi * tile(x[:, [0]], (1, len(J1))) + tile(J1 + 1, (N, 1)) * pi / D)) ** 2
        term2 = (x[:, J2] - cos(6 * pi * tile(x[:, [0]], (1, len(J2))) + tile(J2 + 1, (N, 1)) * pi / D)) ** 2
        return np.hstack(
            [
                x[:, 0] + 2 * np.mean(term1, 1),
                1 - np.sqrt(x[:, 0]) + 2 * np.mean(term2, 1),
            ]
        )

    @timeit
    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        # TODO: this function is calling the objective. Figure out a more efficient impplementation
        y = np.atleast_2d(self.objective(x))
        t = y[:, 1] + np.sqrt(y[:, 0]) - np.sin(2 * np.pi * (np.sqrt(y[:, 0]) - y[:, 1] + 1)) - 1
        return -t / (1 + np.exp(4 * abs(t)))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        idx = (1 / 16 <= f) & (f <= 4 / 16) | (9 / 16 <= f) & (f <= 1)
        f = f[idx]
        return np.c_[f, 1 - np.sqrt(f)]


class CF3(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__()
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 1

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.clip(x, self.xl, self.xu)
        N = x.shape[0]
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
        Y = x - sin(6 * pi * tile(x[:, 0], (1, D)) + tile(np.arange(1, D + 1), (N, 1)) * pi / D)
        term1 = prod(cos(20 * Y[:, J1] * pi / sqrt(tile(J1 + 1, (N, 1)))), 1)
        term2 = prod(cos(20 * Y[:, J2] * pi / sqrt(tile(J2 + 1, (N, 1)))), 1)
        F1 = x[:, 0] + 2 / len(J1) * (4 * sum(Y[:, J1] ** 2, 1) - 2 * term1 + 2)
        F2 = 1 - x[:, 0] ** 2 + 2 / len(J2) * (4 * sum(Y[:, J2] ** 2, 1) - 2 * term2 + 2)
        return np.hstack([F1, F2])

    @timeit
    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        # TODO: this function is calling the objective. Figure out a more efficient impplementation
        y = np.atleast_2d(self.objective(x))
        return 1 - y[:, 1] - y[:, 0] ** 2 + sin(2 * pi * (y[:, 0] ** 2 - y[:, 1] + 1))

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f = np.linspace(0, 1, N)
        idx = (0 < f) & (f < 1 / 2) | (sqrt(1 / 2) < f) & (f < sqrt(3 / 4))
        f = f[~idx]
        return np.c_[f, 1 - f**2]


class CF4(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__()
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 1

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X = np.clip(X, self.xl, self.xu)
        N = X.shape[0]
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
        Y = X - sin(6 * pi * tile(X[:, 0], (1, D)) + tile(np.arange(1, D + 1), (N, 1)) * pi / D)
        temp = Y[:, 1] < 3 / 2 * (1 - sqrt(1 / 2))
        if temp:
            h = np.atleast_2d(np.concatenate([Y[:, 0] ** 2, abs(Y[temp, 1]), (Y[:, 2:] ** 2).ravel()]))
        else:
            h = np.atleast_2d(
                np.concatenate([Y[:, 0] ** 2, 0.125 + (Y[~temp, 1] - 1) ** 2, (Y[:, 2:] ** 2).ravel()])
            )
        F1 = X[:, 0] + sum(h[:, J1], 1)
        F2 = 1 - X[:, 0] + sum(h[:, J2], 1)
        return np.hstack([F1, F2])

    @timeit
    def ieq_constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_var
        t = X[:, 1] - sin(6 * pi * X[:, 0] + 2 * pi / D) - 0.5 * X[:, 0] + 0.25
        return -t / (1 + exp(4 * abs(t)))

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
        super().__init__()
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 1

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N = X.shape[0]
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
        Y1 = X[:, J1] - 0.8 * tile(X[:, 0], (1, len(J1))) * cos(
            6 * pi * tile(X[:, 0], (1, len(J1))) + tile(J1 + 1, (N, 1)) * pi / D
        )
        Y2 = X[:, J2] - 0.8 * tile(X[:, 0], (1, len(J2))) * sin(
            6 * pi * tile(X[:, 0], (1, len(J2))) + tile(J2 + 1, (N, 1)) * pi / D
        )
        h1 = 2 * Y1**2 - cos(4 * pi * Y1) + 1
        h2 = 2 * Y2[:, 1:] ** 2 - cos(4 * pi * Y2[:, 1:]) + 1
        temp = Y2[:, 0] < 3 / 2 * (1 - sqrt(1 / 2))
        if temp:
            h3 = abs(Y2[:, 0])
        else:
            h3 = 0.125 + (Y2[:, 0] - 1) ** 2
        F1 = X[:, 0] + sum(h1, 1)
        F2 = 1 - X[:, 0] + sum(h2, 1) + h3
        return np.hstack([F1, F2])

    @timeit
    def ieq_constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_var
        return -X[:, 1] + 0.8 * X[:, 0] * sin(6 * pi * X[:, 0] + 2 * pi / D) + 0.5 * X[:, 0] - 0.25

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
        super().__init__()
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 2

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
        Y1 = X[:, J1] - 0.8 * tile(X[:, 0], (1, len(J1))) * cos(
            6 * pi * tile(X[:, 0], (1, len(J1))) + (J1 + 1) * pi / D
        )
        Y2 = X[:, J2] - 0.8 * tile(X[:, 0], (1, len(J2))) * sin(
            6 * pi * tile(X[:, 0], (1, len(J2))) + (J2 + 1) * pi / D
        )
        return np.hstack([X[:, 0] + sum(Y1**2, 1), (1 - X[:, 0]) ** 2 + sum(Y2**2, 1)])

    @timeit
    def ieq_constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_var
        G1 = (
            -X[:, 1]
            + 0.8 * X[:, 0] * sin(6 * pi * X[:, 0] + 2 * pi / D)
            + sign(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2)
            * sqrt(abs(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2))
        )
        G2 = (
            -X[:, 3]
            + 0.8 * X[:, 0] * sin(6 * pi * X[:, 0] + 4 * pi / D)
            + sign(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0]))
            * sqrt(abs(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0])))
        )
        return np.hstack([G1, G2])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = (1 - f1) ** 2
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = 0.5 * (1 - f1[idx1])
        f2[idx2] = 0.25 * sqrt(1 - f1[idx2])
        return np.c_[f1, f2]


class CF7(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__()
        self.n_obj = 2
        self.n_var = n_var
        self.xl = np.r_[0, np.zeros(self.n_var - 1) - 2]
        self.xu = np.r_[1, np.zeros(self.n_var - 1) + 2]
        self.n_ieq_constr = 2

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_var
        J1 = np.arange(3, self.n_var, 2) - 1
        J2 = np.arange(2, self.n_var + 2, 2) - 1
        Y1 = X[:, J1] - cos(6 * pi * tile(X[:, 0], (1, len(J1))) + (J1 + 1) * pi / D)
        Y2 = X[:, J2] - sin(6 * pi * tile(X[:, 0], (1, len(J2))) + (J2 + 1) * pi / D)
        h1 = 2 * Y1**2 - cos(4 * pi * Y1) + 1
        h2 = 2 * Y2[:, 2:] ** 2 - cos(4 * pi * Y2[:, 2:]) + 1
        h3 = Y2[:, :2] ** 2
        return np.hstack([X[:, 0] + sum(h1, 1), (1 - X[:, 0]) ** 2 + sum(h2, 1) + sum(h3, 1)])

    @timeit
    def ieq_constraint(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = self.n_var
        G1 = (
            -X[:, 1]
            + sin(6 * pi * X[:, 0] + 2 * pi / D)
            + sign(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2)
            * sqrt(abs(0.5 * (1 - X[:, 0]) - (1 - X[:, 0]) ** 2))
        )
        G2 = (
            -X[:, 3]
            + sin(6 * pi * X[:, 0] + 4 * pi / D)
            + sign(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0]))
            * sqrt(abs(0.25 * sqrt(1 - X[:, 0]) - 0.5 * (1 - X[:, 0])))
        )
        return np.hstack([G1, G2])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, N)
        f2 = (1 - f1) ** 2
        idx1 = (0.5 < f1) & (f1 <= 0.75)
        idx2 = 0.75 < f1
        f2[idx1] = 0.5 * (1 - f1[idx1])
        f2[idx2] = 0.25 * sqrt(1 - f1[idx2])
        return np.c_[f1, f2]


class CF8(UF8, ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var)
        self.n_ieq_constr = 1
        self.n_obj = 3
        self.n_var = n_var
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 4]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 4]

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        np.clip(x, self.xl, self.xu)
        return super().objective(x)

    @timeit
    def ieq_constraint(self, x: np.ndarray) -> float:
        # TODO: avoid calling the constraint fucntion two times
        F = self.objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + 4 * abs(sin(2 * pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1)))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        N = _np.ceil(N / 5) * 5
        R = _np.zeros((N, 3))
        R[:, 2] = tile(sin(_np.linspace(0, 1, num=N / 5 - 1) * pi / 2).reshape(-1, 1), (5, 1))
        for i in range(0, 5):
            R[i * N / 5 : (i + 1) * N / 5, 0] = sqrt(i / 4 * (1 - R[i * N / 5 : (i + 1) * N / 5, 2] ** 2))
        R[:, 1] = sqrt(max(1 - R[:, 0] ** 2 - R[:, 2] ** 2, 0))
        return R


class CF9(CF8, ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var)
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 2]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 2]

    @timeit
    def objective(self, x: np.ndarray) -> np.ndarray:
        return super().objective(x)

    @timeit
    def ieq_constraint(self, x: np.ndarray) -> float:
        F = self.objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + 3 * np.sin(2 * np.pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        R = _np.random.rand(N, 3)
        R = R / tile(sqrt(sum(R**2, 1)), (1, 3))
        idx = (1e-5 < R[:, 0]) & (R[:, 0] < sqrt((1 - R[:, 2] ** 2) / 4)) | (
            sqrt((1 - R[:, 2] ** 2) / 2) < R[:, 0]
        ) & (R[:, 0] < sqrt(3 * (1 - R[:, 2] ** 2) / 4))
        return R[~idx]


class CF10(ConstrainedMOOAnalytical):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__()
        self.n_var = n_var
        self.n_ieq_constr = 1
        self.n_obj = 3
        self.xl = np.r_[0, 0, np.zeros(self.n_var - 2) - 2]
        self.xu = np.r_[1, 1, np.zeros(self.n_var - 2) + 2]

    @timeit
    def objective(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N = X.shape[0]
        D = self.n_var
        J1 = arange(4, D + 1, 3) - 1
        J2 = arange(5, D + 1, 3) - 1
        J3 = arange(3, D + 1, 3) - 1
        Y = X - 2 * tile(X[:, [1]], (1, D)) * sin(
            2 * pi * tile(X[:, [0]], (1, D)) + tile(arange(D) + 1, (N, 1)) * pi / D
        )
        F1 = cos(0.5 * X[:, 0] * pi) * cos(0.5 * X[:, 1] * pi) + 2 * mean(
            4 * Y[:, J1] ** 2 - cos(8 * pi * Y[:, J1]) + 1, 1
        )
        F2 = cos(0.5 * X[:, 0] * pi) * sin(0.5 * X[:, 1] * pi) + 2 * mean(
            4 * Y[:, J2] ** 2 - cos(8 * pi * Y[:, J2]) + 1, 1
        )
        F3 = sin(0.5 * X[:, 0] * pi) + 2 * mean(4 * Y[:, J3] ** 2 - cos(8 * pi * Y[:, J3]) + 1, 1)
        return np.hstack([F1, F2, F3])

    @timeit
    def ieq_constraint(self, x: np.ndarray) -> float:
        F = self.objective(x).reshape(1, -1)
        return (
            1
            - (F[:, 0] ** 2 + F[:, 1] ** 2) / (1 - F[:, 2] ** 2)
            + sin(2 * pi * ((F[:, 0] ** 2 - F[:, 1] ** 2) / (1 - F[:, 2] ** 2) + 1))
        )

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        R = _np.random.rand(N, 3)
        R = R / tile(sqrt(sum(R**2, 1)), (1, 3))
        idx = (1e-5 < R[:, 0]) & (R[:, 0] < sqrt((1 - R[:, 2] ** 2) / 4)) | (
            sqrt((1 - R[:, 2] ** 2) / 2) < R[:, 0]
        ) & (R[:, 0] < sqrt(3 * (1 - R[:, 2] ** 2) / 4))
        return R[~idx]
