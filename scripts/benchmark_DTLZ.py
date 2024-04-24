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
from pymoo.core.problem import Problem as PymooProblem

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.newton import DpN
from hvd.problems import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    IDTLZ1,
    IDTLZ2,
    IDTLZ3,
    IDTLZ4,
    PymooProblemWithAD,
)
from hvd.utils import get_non_dominated

plt.style.use("ggplot")
rcParams["font.size"] = 12
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
# rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

max_iters = 5
n_jobs = 30
problem_name = sys.argv[1]
gen = 300
emoa = "NSGA-III"
print(problem_name)
if problem_name.startswith("IDTLZ"):
    problem = locals()[problem_name](boundry_constraints=True)
    path = "./data-reference/IDTLZ/"
elif problem_name.startswith("DTLZ"):
    problem = locals()[problem_name](boundry_constraints=True)
    path = "./data-reference/DTLZ/"

problem = PymooProblemWithAD(problem) if isinstance(problem, PymooProblem) else problem
pareto_front = problem.get_pareto_front()
reference_point = {
    "DTLZ[1-6]": np.array([1, 1, 1]),
    "DTLZ7": np.array([1, 1, 6]),
    "IDTLZ1[1-4]": np.array([1, 1, 1]),
}


def plot_3d(y0, Y, ref, hist_Y, history_medoids, hist_IGD, hist_R_norm, fig_name):
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[15], colors[19]]
    medoids0 = np.array([h[0] for h in history_medoids])

    # fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
    # plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
    # ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    fig = plt.figure(figsize=plt.figaspect(1 / 1.0))
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    ax0 = fig.add_subplot(1, 1, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)
    ax0.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k.", ms=8, alpha=0.6)
    # ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.4)
    # ax0.plot(ref[:, 0], ref[:, 1], ref[:, 2], "b.", mec="none", ms=5, alpha=0.2)
    ax0.plot(
        medoids0[:, 0],
        medoids0[:, 1],
        medoids0[:, 2],
        color=colors[0],
        ls="none",
        marker="^",
        mec="none",
        ms=5,
        alpha=0.8,
    )

    ax0.set_title("Initialization")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.set_zlabel(r"$f_3$")
    lgnd = ax0.legend(
        # [r"$Y_0$", "Pareto front", "medoids"],
        [r"$Y_0$", "reference set"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
        fancybox=True,
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)
    plt.savefig(fig_name + "_1.pdf", dpi=1000)

    # for i in range(len(y0)):
    # ax0.plot((medoids0[i, 0], y0[i, 0]), (medoids0[i, 1], y0[i, 1]), (medoids0[i, 2], y0[i, 2]), "k-")

    fig = plt.figure(figsize=plt.figaspect(1 / 1.0))
    # plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")

    # ax1 = fig.add_subplot(1, 3, 2, projection="3d")
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
        # labels=[f"{i + 1} shift(s): {k} medoids" for i, k in enumerate(counts)]  # ["Pareto front"]
        labels=["reference set"] + [r"$Y_{\mathrm{final}}$"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
        fancybox=True,
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    ax1.set_title("Final population")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.set_ylabel(r"$f_3$")
    plt.savefig(fig_name + "_2.pdf", dpi=1000)

    # ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    # ax2.set_box_aspect((1, 1, 1))
    # ax2.view_init(45, 45)
    # ax2.set_title("Before/After")
    # ax2.set_xlabel(r"$f_1$")
    # ax2.set_ylabel(r"$f_2$")
    # ax2.set_ylabel(r"$f_3$")
    # ax2.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k.", ms=12, alpha=0.3)
    # ax2.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g+", ms=8, alpha=0.8)
    # lgnd = ax2.legend(
    #     [r"$Y_0$", r"$Y_{\mathrm{final}}$"],
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.25),
    #     ncol=2,
    #     fancybox=True,
    # )
    # for handle in lgnd.legend_handles:
    #     handle.set_markersize(10)

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6.5))
    plt.subplots_adjust(right=0.85, left=0.2)

    ax22 = ax2.twinx()
    ax2.semilogy(range(1, len(hist_IGD) + 1), hist_IGD, "r-", label="IGD")
    ax22.semilogy(range(1, len(hist_R_norm) + 1), hist_R_norm, "g--")
    ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    ax2.set_ylabel("IGD", color="r")
    # ax22.set_ylabel(r"R norm", color="g")
    ax2.set_title("Performance")
    ax2.set_xlabel("iteration")
    ax2.set_xticks(range(1, max_iters + 1))
    ax2.legend()
    # plt.tight_layout()
    plt.savefig(fig_name + "_3.pdf", dpi=1000)

    # plt.savefig(fig_name, dpi=1000)
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
        if problem_name in ["DTLZ6", "DTLZ7"]:
            # for DTLZ7 we need to load the dense fillings of the reference set
            r = pd.read_csv(
                f"{path}/{problem_name}_{emoa}_run_{run}_filling_comp{i+1}_gen{gen}.csv", header=None
            ).values
        else:
            if n_cluster == 1:
                ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_gen{gen}.csv"
            else:
                ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_{i+1}_gen{gen}.csv"
            try:
                r = pd.read_csv(ref_file, header=None).values
            except:
                continue
        # downsample the reference; otherwise, initial clustering take too long
        ref[i] = np.array(random.sample(r.tolist(), 3000)) if len(r) >= 3000 else r
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
    if (eta is not None) and (np.any([np.any(np.isnan(_eta)) for _eta in eta.values()])):
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
    all_ref = np.vstack([r for r in ref.values()])

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
        # ensure the number of approximation points is less than the number of reference points
        if len(ref) < len(y0):
            n = len(ref)
            x0 = x0[:n]
            y0 = y0[:n]
            Y_label = Y_label[:n]

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
        x0=x0,
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
    if 1 < 2:  # plotting the final approximation set
        fig_name = f"./plots/{problem_name}_DpN_{emoa}_run{run}_{gen}"
        plot_3d(y0, Y, all_ref, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    if 1 < 2:  # save the final approximation set
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y.csv", index=False)
        df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
        df_y0.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y0.csv", index=False)

    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    # hv_value = hypervolume(Y, ref)
    return np.array([igd_value, gd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]
if problem_name == "DTLZ2" and emoa == "SMS-EMOA":
    run_id = list(set(run_id) - set([7]))
if problem_name == "DTLZ4" and emoa == "MOEAD":
    run_id = list(set(run_id) - set([3]))
if problem_name == "IDTLZ4" and emoa == "NSGA-III":
    run_id = list(set(run_id) - set([12]))

if 1 < 2:
    data = []
    for i in [19]:
        print(i)
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
# df.to_csv(f"results/{problem_name}-Dp/N-{emoa}-{gen}.csv", index=False)
