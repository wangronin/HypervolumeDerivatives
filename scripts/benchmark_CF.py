import sys

sys.path.insert(0, "./")
import random
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import rcParams

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10
from hvd.utils import get_non_dominated

plt.style.use("ggplot")
rcParams["font.size"] = 17
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
# rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 17
rcParams["ytick.labelsize"] = 17
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

max_iters = 6
n_jobs = 30
problem_name = sys.argv[1]
print(problem_name)
f = locals()[problem_name]()
problem = f
pareto_front = problem.get_pareto_front(1000)

path = "./data-reference/CF/"
emoa = "SMS-EMOA"
gen = 300

# TODO: move those plotting functions to a utils.py


def plot_2d(y0, Y, ref, hist_Y, history_medoids, hist_IGD, hist_R_norm, fig_name):
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[17], colors[19]]
    plt.style.use("ggplot")
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))
    plt.subplots_adjust(right=0.93, left=0.05)

    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.4)
    ax0.plot(y0[:, 0], y0[:, 1], "k+", ms=12, alpha=1)
    ax0.plot(ref[:, 0], ref[:, 1], "b.", mec="none", ms=5, alpha=0.3)
    ax0.set_title("Objective space (Initialization)")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    lgnd = ax0.legend(["Pareto front", r"$Y_0$", "reference set", "matched points"])
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    N = len(y0)
    if 1 < 2:
        trajectory = np.array([y0] + hist_Y)
        for i in range(N):
            x, y = trajectory[:, i, 0], trajectory[:, i, 1]
            ax1.quiver(
                x[:-1],
                y[:-1],
                x[1:] - x[:-1],
                y[1:] - y[:-1],
                scale_units="xy",
                angles="xy",
                scale=1,
                color="k",
                width=0.003,
                alpha=0.5,
                headlength=4.5,
                headwidth=2.5,
            )

    lines = []
    lines += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.3)
    shifts = []
    for i, M in enumerate(history_medoids):
        c = colors[len(M) - 1]
        for j, x in enumerate(M):
            line = ax1.plot(x[0], x[1], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
            if j == len(shifts):
                shifts.append(line)

    lines += shifts
    lines += ax1.plot(Y[:, 0], Y[:, 1], "k*", mec="none", ms=8, alpha=0.9)
    counts = np.unique([len(m) for m in history_medoids], return_counts=True)[1]
    lgnd = ax1.legend(
        lines,
        ["Pareto front"]
        + [f"{i + 1} shift(s): {k} points" for i, k in enumerate(counts)]
        + [r"$Y_{\mathrm{final}}$"],
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(12)

    ax1.set_title("Objective space")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")

    ax22 = ax2.twinx()
    ax2.semilogy(range(1, len(hist_IGD) + 1), hist_IGD, "r-", label="IGD")
    ax22.semilogy(range(1, len(hist_R_norm) + 1), hist_R_norm, "g--")
    ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    ax22.set_ylabel(r"R norm", color="g")
    ax2.set_title("Performance")
    ax2.set_xlabel("iteration")
    ax2.set_xticks(range(1, max_iters + 1))
    ax2.legend()
    # plt.tight_layout()
    plt.savefig(fig_name, dpi=1000)
    plt.close(fig)


def plot_3d(y0, Y, ref, hist_Y, history_medoids, hist_IGD, hist_R_norm, fig_name):
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[15], colors[19]]
    medoids0 = np.array([h[0] for h in history_medoids])

    fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
    plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)
    ax0.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k.", ms=12, alpha=0.3)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.4)
    # ax0.plot(ref[:, 0], ref[:, 1], ref[:, 2], "b.", mec="none", ms=5, alpha=0.2)
    ax0.plot(
        medoids0[:, 0],
        medoids0[:, 1],
        medoids0[:, 2],
        color=colors[0],
        ls="none",
        marker="^",
        mec="none",
        ms=7,
        alpha=0.8,
    )
    ax0.set_title("Objective space (Initialization)")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.set_ylabel(r"$f_3$")
    lgnd = ax0.legend(
        [r"$Y_0$", "Pareto front", "medoids"],
        # [r"$Y_0$", "medoids"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fancybox=True,
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(45, 45)
    if 11 < 2:
        trajectory = np.array([y0] + hist_Y)
        for i in range(len(y0)):
            x, y, z = trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2]
            ax1.quiver(
                x[:-1],
                y[:-1],
                z[:-1],
                x[1:] - x[:-1],
                y[1:] - y[:-1],
                z[1:] - z[:-1],
                color="k",
                alpha=0.5,
                arrow_length_ratio=0.05,
            )

    lines = []
    # lines += ax1.plot(
    #     pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.2
    # )
    shifts = []
    for i, M in enumerate(history_medoids):
        c = colors[len(M) - 1]
        for j, x in enumerate(M):
            line = ax1.plot(x[0], x[1], x[2], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
            if j == len(shifts):
                shifts.append(line)

    lines += shifts
    lines += ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "k*", mec="none", ms=8, alpha=0.9)
    counts = np.unique([len(m) for m in history_medoids], return_counts=True)[1]
    lgnd = ax1.legend(
        handles=lines,
        labels=
        # ["Pareto front"]
        [f"{i + 1} shift(s): {k} medoids" for i, k in enumerate(counts)] + [r"$Y_{\mathrm{final}}$"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fancybox=True,
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    ax1.set_title("Objective space")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.set_ylabel(r"$f_3$")

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    ax2.set_box_aspect((1, 1, 1))
    ax2.view_init(45, 45)
    ax2.set_title("Before/After")
    ax2.set_xlabel(r"$f_1$")
    ax2.set_ylabel(r"$f_2$")
    ax2.set_ylabel(r"$f_3$")
    ax2.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k.", ms=12, alpha=0.3)
    ax2.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g+", ms=8, alpha=0.8)
    lgnd = ax2.legend(
        [r"$Y_0$", r"$Y_{\mathrm{final}}$"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fancybox=True,
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    # ax2 = fig.add_subplot(1, 3, 3)
    # ax2.set_aspect("equal")
    # ax22 = ax2.twinx()
    # ax2.semilogy(range(1, len(hist_IGD) + 1), hist_IGD, "r-", label="IGD")
    # ax22.semilogy(range(1, len(hist_R_norm) + 1), hist_R_norm, "g--")
    # ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    # ax2.set_title("Performance")
    # ax2.set_xlabel("iteration")
    # ax2.set_xticks(range(1, max_iters + 1))
    # ax2.legend()
    # plt.tight_layout()
    plt.savefig(fig_name, dpi=1000)
    plt.close(fig)


def execute(run: int):
    ref_label = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_component_id_gen{gen}.csv", header=None
    ).values[0]
    n_cluster = len(np.unique(ref_label))
    ref = dict()
    eta = dict()
    # load the reference set
    for i in range(n_cluster):
        try:
            r = pd.read_csv(
                f"{path}/{problem_name}_{emoa}_run_{run}_filling_comp{i+1}_gen{gen}.csv", header=None
            ).values
        except:
            continue
        # downsample the reference; otherwise, the initial clustering takes forever
        ref[i] = np.array(random.sample(r.tolist(), 500)) if len(r) >= 500 else r
        try:
            eta[i] = pd.read_csv(
                f"{path}/{problem_name}_{emoa}_run_{run}_eta_{i+1}_gen{gen}.csv", header=None
            ).values.ravel()
        except:
            if i > 0 and eta[i - 1] is not None:  # copy the shift direction from the last cluster
                eta[i] = eta[i - 1]
            else:
                eta = None

    # sometimes the precomputed `eta` value can be `nan`
    if np.any([np.any(np.isnan(_eta)) for _eta in eta.values()]):
        eta = None

    # the load the final population from an EMOA
    x0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_x_gen{gen}.csv", header=None).values
    y0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_y_gen{gen}.csv", header=None).values
    Y_label = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_labels_gen{gen}.csv", header=None
    ).values.ravel()
    Y_label = Y_label - 1
    # removing the outliers in `Y`
    idx = (Y_label != -2) & (Y_label != -1)
    x0 = x0[idx]
    y0 = y0[idx]
    Y_label = Y_label[idx]
    all_ref = np.concatenate([v for v in ref.values()], axis=0)

    # TODO: this is an ad-hoc solution. Maybe fix this special case in the `ReferenceSet` class
    # if the minimal number of points in the `ref` clusters is smaller than
    # the maximal number of points in `y0` clusters, then we merge all clusters
    min_point_ref_cluster = np.min([len(r) for r in ref.values()])
    max_point_y_cluster = np.max(np.unique(Y_label, return_counts=True)[1])
    # if the number of clusters of `Y` is more than that of the reference set
    if (len(np.unique(Y_label)) > len(ref)) or (max_point_y_cluster > min_point_ref_cluster):
        ref = np.vstack([r for r in ref.values()])
        Y_label = np.zeros(len(y0))
        eta = None

    # create the algorithm
    opt = DpN(
        dim=problem.n_var,
        n_obj=problem.n_obj,
        ref=ref,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        N=len(x0),
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=max_iters,
        type="igd",
        verbose=True,
        pareto_front=pareto_front,
        eta=eta,
        Y_label=Y_label,
    )
    X, Y, _ = opt.run()
    # remove the dominated solution in Y
    Y = get_non_dominated(Y)
    fig_name = f"./plots/{problem_name}_DpN_{emoa}_run{run}_{gen}.pdf"
    # plot_2d(y0, Y, all_ref, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    plot_3d(y0, Y, all_ref, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    return np.array([igd_value, gd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]
if problem_name == "CF2" and emoa == "NSGA-II":
    run_id = list(set(run_id) - set([15, 16]))
if problem_name == "CF3" and emoa == "NSGA-II":
    run_id = list(set(run_id) - set([8]))
if problem_name == "CF5" and emoa == "NSGA-II":
    run_id = list(set(run_id) - set([26, 8, 18]))
if problem_name == "CF6" and emoa == "NSGA-II":
    run_id = list(set(run_id) - set([30, 1, 11, 16, 20, 24, 27, 29]))
if problem_name == "CF7" and emoa == "NSGA-II":
    run_id = list(set(run_id) - set([18, 17, 24, 26, 27]))
if problem_name == "CF5" and emoa == "NSGA-III":
    run_id = list(set(run_id) - set([4]))

if 11 < 2:
    data = []
    for i in run_id:
        print(i)
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
# df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV"])
df.to_csv(f"results/{problem_name}-DpN-{emoa}-{gen}.csv", index=False)
