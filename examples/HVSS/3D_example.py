import sys
from typing import List

sys.path.insert(0, "./")
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from base import ParetoApproximator
from matplotlib import rcParams

from hvd.newton import HVN
from hvd.problems import DTLZ2

random.seed(422)
np.random.seed(422)

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


def HV_subset_selection(
    N: int = 100,
    max_iters: int = 25,
    ref_point: List[float] = [1.5, 1.5, 1.5],
) -> None:
    t0 = time.time()
    X0 = Y0 = pd.read_csv(f"{path}/DTLZ2_Ay.csv", header=None, index_col=False).values
    idx = random.sample(range(0, len(Y0) + 1), N)
    centers = pd.read_csv(f"{path}/DTLZ2_box_centers_f1f2.csv", header=None, index_col=False).values
    radius = pd.read_csv(f"{path}/DTLZ2_box_radius_f1f2.csv", header=None, index_col=False).values
    func = ParetoApproximator(data=Y0, box_centers=centers, radii=radius)

    fig = plt.figure(figsize=plt.figaspect(1 / 1.0))
    plt.subplots_adjust(bottom=0.08, top=0.9, right=0.93, left=0.05)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)
    x, y = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
    f = lambda x: func._pareto_approximator(torch.from_numpy(x).float()).detach().cpu().numpy()
    z = np.array([f(p) for p in np.c_[x.ravel(), y.ravel()]])
    z = z.reshape(x.shape[0], -1)
    ax0.plot_surface(x, y, z, cmap="viridis", edgecolor="none", antialiased=True)
    line = ax0.plot(Y0[:, 0], Y0[:, 1], Y0[:, 2], "k.", mec="none", ms=5, alpha=0.6)
    ax0.legend(line, ["training points"])
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.set_zlabel(r"$f_3$")

    # calling HVN
    opt = HVN(
        n_var=3,
        n_obj=3,
        ref=ref_point,
        func=func.objective,
        jac=func.jacobian,
        hessian=func.hessian,
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
    ax0.set_title(f"initial HV: {HV0}")

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

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(45, 45)
    func.plot_domain(ax=ax1, facecolor="none")
    lns = []
    lns += ax1.plot(Y0[idx, 0], Y0[idx, 1], Y0[idx, 2], "g+", ms=9, alpha=1)
    lns += ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r*", ms=8, alpha=0.6)
    lns += ax1.plot(
        pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.6
    )
    ax1.legend(lns, ["initial points", "final points", "approximated PF"])
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.set_zlabel(r"$f_3$")
    ax1.set_title(f"final HV: {HV1}")
    plt.show()


HV_subset_selection()
