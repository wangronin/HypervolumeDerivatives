import sys
from typing import List, Union

sys.path.insert(0, "./")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from hvd.newton import HVN
from hvd.problems.base import MOOAnalytical

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 15
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1


class FLatConvex(MOOAnalytical):
    def __init__(self, **kwargs):
        self.n_obj = 2
        self.n_var = 1
        self.xl = 0 * np.ones(self.n_var)
        self.xu = 1 * np.ones(self.n_var)
        super().__init__(**kwargs)

    def _objective(self, t: float) -> jnp.ndarray:
        t = t * 0.9 + 0.05  # rescale `t` to the interval [0.05, 0.95]
        a = 1.0 + 10.0 * (t - t**2)
        f1 = jnp.power(t, a)
        f2 = jnp.power(1.0 - t, a)
        return jnp.array([f1, f2])


class Linear(MOOAnalytical):
    def __init__(self, **kwargs):
        self.n_obj = 2
        self.n_var = 1
        self.xl = 0 * np.ones(self.n_var)
        self.xu = 1 * np.ones(self.n_var)
        super().__init__(**kwargs)

    def _objective(self, t: float) -> jnp.ndarray:
        t *= 0.3
        f1 = 0.92713359 + t
        f2 = -0.5 - t
        return jnp.array([f1, f2])


class PiecewiseAnalytical:
    def __init__(self, functions: List[MOOAnalytical]) -> None:
        self._functions: List[MOOAnalytical] = functions
        self.N: int = len(self._functions)
        self.intervals: np.ndarray = np.linspace(0, 1, self.N + 1)
        self.size: float = 1 / self.N

    def get_index(self, t: float) -> int:
        """get the index of the B-spline to which the input parameter `t` belongs

        Args:
            t (float): the parameter value

        Returns:
            int: the index of the B-spline that `t` belongs to
        """
        if isinstance(t, np.ndarray):
            t = t[0]
        k = np.floor(t / (self.size))
        if t == 1.0:
            k = self.N - 1
        return int(k)

    def __func__(self, t: Union[np.ndarray, float], type: str):
        if isinstance(t, np.ndarray):
            t = t[0]
        k = self.get_index(t)
        return getattr(self._functions[k], type)(self.N * (t - self.intervals[k]))

    def objective(self, t: Union[np.ndarray, float]) -> np.ndarray:
        return self.__func__(t, "objective")

    def jacobian(self, t: Union[np.ndarray, float]) -> np.ndarray:
        return self.__func__(t, "objective_jacobian")

    def hessian(self, t: Union[np.ndarray, float]) -> np.ndarray:
        return self.__func__(t, "objective_hessian")


max_iters = 30
problem = PiecewiseAnalytical([FLatConvex(), Linear()])

ref = np.array([2, 1])
X0 = np.r_[np.linspace(0, 0.49, 10), np.linspace(0.51, 1, 30)]
# the number of points of each connected component of the Pareto front initially
labels0, counts0 = np.unique([problem.get_index(t) for t in X0], return_counts=True)
Y0 = np.array([problem.objective(x) for x in X0])
N = len(X0)
pareto_front1 = np.array([problem.objective(_) for _ in np.linspace(0, 0.4999, 1000)])
pareto_front2 = np.array([problem.objective(_) for _ in np.linspace(0.5, 1, 1000)])

opt = HVN(
    n_var=1,
    n_obj=2,
    ref=ref,
    func=problem.objective,
    jac=problem.jacobian,
    hessian=problem.hessian,
    N=N,
    X0=X0,
    xl=0,
    xu=1,
    max_iters=max_iters,
    verbose=False,
    preconditioning=False,
)
Y = opt.run()[1]
# the number of points of each connected component of the Pareto front after optimization
labels1, counts1 = np.unique([problem.get_index(t) for t in opt.state.X], return_counts=True)
best_so_far_HV = np.maximum.accumulate(np.array(opt.history_indicator_value))
best_so_far_R_norm = np.minimum.accumulate(np.array(opt.history_R_norm))
HV0, HV1 = best_so_far_HV[0], best_so_far_HV[-1]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 12))
plt.subplots_adjust(right=0.93, left=0.05)
ax0.plot(Y0[:, 0], Y0[:, 1], "r+", ms=8)
ax0.plot(pareto_front1[:, 0], pareto_front1[:, 1], "k--", alpha=0.7)
ax0.plot(pareto_front2[:, 0], pareto_front2[:, 1], "k--", alpha=0.7)
ax0.set_title(f"Initial HV: {HV0}")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.set_aspect("equal")
ax0.text(0.38, 0.6, f"{counts0[0]} points in component 1", transform=ax0.transAxes, ha="center")
ax0.text(0.4, 0.1, f"{counts0[1]} points in component 2", transform=ax0.transAxes, ha="center")
ax0.legend([r"$Y_0$", "Approximated Pareto front"])

ax1.plot(Y[:, 0], Y[:, 1], "r+", ms=8)
ax1.plot(pareto_front1[:, 0], pareto_front1[:, 1], "k--", alpha=0.7)
ax1.plot(pareto_front2[:, 0], pareto_front2[:, 1], "k--", alpha=0.7)
ax1.set_title(f"Final HV: {HV1}")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax1.set_aspect("equal")
ax1.text(0.38, 0.6, f"{counts1[0]} points in component 1", transform=ax1.transAxes, ha="center")
ax1.text(0.4, 0.1, f"{counts1[1]} points in component 2", transform=ax1.transAxes, ha="center")
ax1.legend([r"$Y_{\text{final}}$", "Approximated Pareto front"])
fig.suptitle(f"reference point = {ref}")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
