import sys
from typing import List

sys.path.insert(0, "./")
import random
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.cluster import KMeans

from hvd.newton import HVN
from hvd.problems import DTLZ1, DTLZ2
from hvd.problems.base import ConstrainedMOOAnalytical

jax.config.update("jax_enable_x64", True)
random.seed(42)
np.random.seed(42)

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

path = "./data/HVSS/"


class ParetoFrontConvex(ConstrainedMOOAnalytical):
    n_eq_constr: int = 1
    n_ieq_constr: int = 1

    def __init__(self) -> None:
        self.n_var: int = 3
        self.n_obj: int = 3
        self.xl: np.ndarray = np.array([-1] * self.n_var)
        self.xu: np.ndarray = np.array([0] * self.n_var)
        super().__init__(boundry_constraints=True)

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def _eq_constraint(self, x: jnp.ndarray) -> float:
        return jnp.array([jnp.sum(x**2) - 1])

    def _ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return jnp.array([jnp.sum(x[0:2] ** 2) - 1])

    def get_pareto_front(self) -> np.ndarray:
        return DTLZ2(n_var=3).get_pareto_front()


class ParetoFrontConcave(ParetoFrontConvex, ConstrainedMOOAnalytical):

    def __init__(self) -> None:
        self.n_var: int = 3
        self.n_obj: int = 3
        self.xl: np.ndarray = np.array([0] * self.n_var)
        self.xu: np.ndarray = np.array([1] * self.n_var)
        ConstrainedMOOAnalytical.__init__(self, boundry_constraints=True)


class ParetoFrontLinear(ConstrainedMOOAnalytical):
    n_eq_constr: int = 1
    n_ieq_constr: int = 1

    def __init__(self) -> None:
        self.n_var: int = 3
        self.n_obj: int = 3
        self.xl: np.ndarray = np.array([0] * self.n_var)
        self.xu: np.ndarray = np.array([1] * self.n_var)
        super().__init__(boundry_constraints=True)

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def _eq_constraint(self, x: jnp.ndarray) -> float:
        return jnp.array([jnp.sum(x) - 0.5])

    def _ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return jnp.array([jnp.sum(x[0:2]) - 0.5])

    def get_pareto_front(self) -> np.ndarray:
        return DTLZ1(n_var=3).get_pareto_front()


def select_initial_points(X0: np.ndarray, N: int) -> List[int]:
    kmeans = KMeans(n_clusters=2 * N, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X0)
    centers = kmeans.cluster_centers_
    # pick representative: nearest to each center
    idx = []
    for i in range(2 * N):
        mask = labels == i
        pts = X0[mask]
        if pts.shape[0] == 0:
            continue
        # get distances to centroid
        dists = np.linalg.norm(pts - centers[i], axis=1)
        # pick index of nearest
        chosen = np.where(mask)[0][np.argmin(dists)]
        idx.append(chosen)
    idx = random.sample(idx, N)
    return idx


def HV_subset_selection(
    N: int = 150,
    max_iters: int = 40,
    ref_point: List[float] = [0.1, 0.1, 0.1],
    problem: str = "convex",
) -> None:
    t0 = time.time()
    if problem == "convex":
        func = ParetoFrontConvex()
        X0 = Y0 = pd.read_csv(f"{path}/DTLZ2_Ay.csv", header=None, index_col=False).values
        X0 = -1.0 * X0
    if problem == "concave":
        func = ParetoFrontConcave()
        X0 = Y0 = pd.read_csv(f"{path}/DTLZ2_Ay.csv", header=None, index_col=False).values
    elif problem == "linear":
        func = ParetoFrontLinear()
        X0 = Y0 = func.get_pareto_front()

    idx = select_initial_points(X0, N)
    pareto_front = func.get_pareto_front()
    # calling HVN
    opt = HVN(
        n_var=3,
        n_obj=3,
        ref=ref_point,
        func=func.objective,
        jac=func.objective_jacobian,
        hessian=func.objective_hessian,
        h=func.eq_constraint,
        h_jac=func.eq_jacobian,
        h_hessian=func.eq_hessian,
        g=func.ieq_constraint,
        g_jac=func.ieq_jacobian,
        g_hessian=func.ieq_hessian,
        N=N,
        X0=X0[idx],
        xl=func.xl,
        xu=func.xu,
        max_iters=max_iters,
        verbose=True,
        preconditioning=False,
    )
    Y = opt.run()[1]
    best_so_far_HV = np.maximum.accumulate(np.array(opt.history_indicator_value))
    # best_so_far_R_norm = np.minimum.accumulate(np.array(opt.history_R_norm))
    HV0, HV1 = best_so_far_HV[0], best_so_far_HV[-1]
    wall_clock_time = time.time() - t0
    print(f"wall clock time: {wall_clock_time}")
    print(f"initial HV: {HV0}")
    print(f"final HV: {HV1}")
    print(f"{len(Y)} final points")

    fig = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(bottom=0.08, top=0.9, right=0.93, left=0.05)
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)
    ax0.plot(Y0[idx, 0], Y0[idx, 1], Y0[idx, 2], "g+", ms=9, alpha=1)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.6)
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.set_zlabel(r"$f_3$")
    ax0.set_title(f"initial HV: {HV0}")

    if problem == "convex":
        Y = -1.0 * Y  # remember to invert the Y values
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(45, 45)
    ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r*", ms=8, alpha=0.6)
    ax1.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.6)
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.set_zlabel(r"$f_3$")
    ax1.set_title(f"final HV: {HV1}")

    ax2 = fig.add_subplot(1, 3, 3)
    ax22 = ax2.twinx()
    ax2.set_box_aspect(1)
    ax22.set_box_aspect(1)
    ax2.plot(range(1, len(best_so_far_HV) + 1), best_so_far_HV, "b-")
    ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
    ax2.set_ylabel("HV", color="b")
    ax22.set_ylabel(r"$\lVert R(\mathbf{X}) \rVert$", color="g")
    ax2.set_title("Performance")
    ax2.set_xlabel("iteration")
    plt.tight_layout()
    plt.savefig(f"3D-HVSS-{problem}-{N}.pdf", dpi=150)


params = [
    dict(problem="convex", N=30, ref_point=[0.002, 0.002, 0.002], max_iters=25),
    dict(problem="convex", N=50, ref_point=[0.05, 0.05, 0.05], max_iters=30),
    dict(problem="convex", N=100, ref_point=[0.04, 0.04, 0.04], max_iters=40),
    dict(problem="convex", N=150, ref_point=[0.1, 0.1, 0.1], max_iters=40),
    dict(problem="linear", N=100, ref_point=[1, 1, 1], max_iters=20),
    dict(problem="concave", N=100, ref_point=[1, 1, 1], max_iters=30),
]
HV_subset_selection(**params[5])
