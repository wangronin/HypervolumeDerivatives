import sys
from typing import List

sys.path.insert(0, "./")
import random
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import HVN
from hvd.problems import DTLZ2
from hvd.problems.base import ConstrainedMOOAnalytical

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
problem = DTLZ2(n_var=7, boundry_constraints=True)
pareto_front = problem.get_pareto_front()


class ParetoFront(ConstrainedMOOAnalytical):
    n_eq_constr: int = 1
    n_ieq_constr: int = 1

    def __init__(self) -> None:
        self.n_var: int = 3
        self.n_obj: int = 3
        self.xl: np.ndarray = np.array([0] * self.n_var)
        self.xu: np.ndarray = np.array([1] * self.n_var)
        super().__init__()

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def _eq_constraint(self, x: jnp.ndarray) -> float:
        return jnp.sum(x**2) - 1

    def _ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return jnp.sum(x[0:2] ** 2) - 1


def HV_subset_selection(
    N: int = 50,
    max_iters: int = 20,
    ref_point: List[float] = [1.2, 1.2, 1.2],
) -> None:
    t0 = time.time()
    X0 = Y0 = pd.read_csv(f"{path}/DTLZ2_Ay.csv", header=None, index_col=False).values
    idx = random.sample(range(0, len(Y0) + 1), N)
    # from sklearn.cluster import KMeans

    # kmeans = KMeans(n_clusters=2 * N, random_state=42, n_init=10)
    # labels = kmeans.fit_predict(X0)
    # centers = kmeans.cluster_centers_

    # # pick representative: nearest to each center
    # idx = []
    # for i in range(2 * N):
    #     mask = labels == i
    #     pts = X0[mask]
    #     if pts.shape[0] == 0:
    #         continue
    #     # get distances to centroid
    #     dists = np.linalg.norm(pts - centers[i], axis=1)
    #     # pick index of nearest
    #     chosen = np.where(mask)[0][np.argmin(dists)]
    #     idx.append(chosen)
    # idx = random.sample(idx, N)

    func = ParetoFront()
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
        preconditioning=True,
    )
    Y = opt.run()[1]
    best_so_far_HV = np.maximum.accumulate(np.array(opt.history_indicator_value))
    best_so_far_R_norm = np.minimum.accumulate(np.array(opt.history_R_norm))
    HV0, HV1 = best_so_far_HV[0], best_so_far_HV[-1]
    # select points from the archive
    wall_clock_time = time.time() - t0
    print(f"wall clock time: {wall_clock_time}")
    print(f"initial HV: {HV0}")
    print(f"final HV: {HV1}")

    # plot the approximation surface
    # fig = plt.figure(figsize=plt.figaspect(1 / 1.0))
    # plt.subplots_adjust(bottom=0.08, top=0.9, right=0.93, left=0.05)
    # ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    # ax0.set_box_aspect((1, 1, 1))
    # ax0.view_init(45, 45)
    # x, y = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
    # f = lambda x: func._pareto_approximator(torch.from_numpy(x).float()).detach().cpu().numpy()
    # z = np.array([f(p) for p in np.c_[x.ravel(), y.ravel()]])
    # z = z.reshape(x.shape[0], -1)
    # ax0.plot_surface(x, y, z, cmap="viridis", edgecolor="none", antialiased=True)
    # ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.6)
    # ax0.set_xlabel(r"$f_1$")
    # ax0.set_ylabel(r"$f_2$")
    # ax0.set_zlabel(r"$f_3$")

    fig = plt.figure(figsize=plt.figaspect(1 / 1.0))
    plt.subplots_adjust(bottom=0.08, top=0.9, right=0.93, left=0.05)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)
    ax0.plot(Y0[idx, 0], Y0[idx, 1], Y0[idx, 2], "g+", ms=9, alpha=1)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.6)
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.set_zlabel(r"$f_3$")
    ax0.set_title(f"initial HV: {HV0}")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(45, 45)
    ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r*", ms=8, alpha=0.6)
    ax1.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.6)
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.set_zlabel(r"$f_3$")
    ax1.set_title(f"final HV: {HV1}")
    plt.show()


HV_subset_selection()
