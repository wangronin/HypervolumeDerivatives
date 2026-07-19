import os

# enable double-precision of JAX
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from pymoo.core.problem import Problem as PymooProblem

from ..utils import timeit
from .base import CMOP, MOP, fixed_n_obj, fixed_n_var


class DENT(MOP):
    def __init__(
        self,
        n_var: int | None = None,
        n_obj: int | None = None,
        xl: ArrayLike = -2.0,
        xu: ArrayLike = 2.0,
    ) -> None:
        n_var = fixed_n_var(n_var, 2, type(self).__name__)
        n_obj = fixed_n_obj(n_obj, 2, type(self).__name__)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
        )

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


class CONV3(CMOP):
    def __init__(
        self,
        n_var: int | None = None,
        n_obj: int | None = None,
        xl: ArrayLike = -3.0,
        xu: ArrayLike = 3.0,
        boundary_constraints: bool = False,
    ) -> None:
        n_var = fixed_n_var(n_var, 3, type(self).__name__)
        n_obj = fixed_n_obj(n_obj, 3, type(self).__name__)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            boundary_constraints=boundary_constraints,
        )
        self.a1 = -1 * np.ones(self.n_var)
        self.a2 = np.ones(self.n_var)
        self.a3 = np.r_[-1 * np.ones(self.n_var - 1), 1]

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        func = lambda x, a: jnp.sum((x - a) ** 2)
        return jnp.array([func(x, self.a1), func(x, self.a2), func(x, self.a3)])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        w = np.random.rand(N, 3)
        w /= w.sum(axis=1).reshape(-1, 1)
        X = w @ np.vstack([self.a1, self.a2, self.a3])
        return np.array([self._objective(x) for x in X])


class CONV4(CMOP):
    """Convex Problem 4"""

    def __init__(
        self,
        n_var: int | None = None,
        n_obj: int | None = None,
        xl: ArrayLike = -10.0,
        xu: ArrayLike = 10.0,
        boundary_constraints: bool = False,
    ) -> None:
        n_var = fixed_n_var(n_var, 4, type(self).__name__)
        n_obj = fixed_n_obj(n_obj, 4, type(self).__name__)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            boundary_constraints=boundary_constraints,
        )
        self.centers = np.eye(self.n_var)

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([jnp.sum((x - self.centers[i]) ** 2) for i in range(self.n_var)])

    def get_pareto_front(self, N: int = 1000) -> np.ndarray:
        w = np.random.rand(N, 4)
        w /= w.sum(axis=1).reshape(-1, 1)
        X = w @ np.eye(self.n_var)  # the positive part
        return np.array([self._objective(x) for x in X])


class CONV4_2F(CMOP):
    """Convex Problem 4 with 2 disconnected Pareto fronts"""

    def __init__(
        self,
        n_var: int | None = None,
        n_obj: int | None = None,
        xl: ArrayLike = -10.0,
        xu: ArrayLike = 10.0,
        boundary_constraints: bool = False,
    ) -> None:
        n_var = fixed_n_var(n_var, 4, type(self).__name__)
        n_obj = fixed_n_obj(n_obj, 4, type(self).__name__)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            boundary_constraints=boundary_constraints,
        )

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


class DisConnected(CMOP):
    def __init__(
        self,
        n_var: int | None = None,
        n_obj: int | None = None,
        xl: ArrayLike | None = None,
        xu: ArrayLike | None = None,
        boundary_constraints: bool = False,
    ) -> None:
        n_var = fixed_n_var(n_var, 2, type(self).__name__)
        n_obj = fixed_n_obj(n_obj, 2, type(self).__name__)
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=np.array([0, -8]) if xl is None else xl,
            xu=np.array([1, 1]) if xu is None else xu,
            n_eq_constr=1,
            n_ieq_constr=2,
            boundary_constraints=boundary_constraints,
        )

    @timeit
    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        g1 = jax.numpy.where(x[0] <= 0.5, 0.1 - x[0], 0.6 - x[0])
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
