import sys

sys.path.insert(0, "./")
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import rcParams

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, PymooProblemWithAD

rcParams["font.size"] = 17
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 17
rcParams["ytick.labelsize"] = 17
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

max_iters = 5
n_jobs = 30
problem_name = sys.argv[1]
print(problem_name)
f = locals()[problem_name]()
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front(1000)

path = "./data-reference/ZDT/"
emoa = "SMS-EMOA"
gen = 300


def plot_2d(y0, Y, ref, hist_Y, history_medoids, hist_IGD, hist_R_norm, fig_name):
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[17], colors[19]]
    plt.style.use("ggplot")
    # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))
    # plt.subplots_adjust(right=0.93, left=0.05)

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6.5))
    plt.subplots_adjust(right=0.9, left=0.1)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.4)
    ax0.plot(y0[:, 0], y0[:, 1], "k+", ms=12, alpha=1)
    ax0.plot(ref[:, 0], ref[:, 1], "b.", mec="none", ms=5, alpha=0.3)
    ax0.set_title("Objective space (Initialization)")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    lgnd = ax0.legend(["Pareto front", r"$Y_0$", "reference set", "matched points"])
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    plt.savefig(fig_name + "_1.pdf", dpi=1000)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6.5))
    plt.subplots_adjust(right=0.9, left=0.1)

    N = len(y0)
    if 11 < 2:
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
    plt.savefig(fig_name + "_2.pdf", dpi=1000)

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6.5))
    plt.subplots_adjust(right=0.85, left=0.2)

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
        # ref[i] = pd.read_csv(
        #     f"{path}/{problem_name}_{emoa}_run_{run}_ref_{i+1}_gen{gen}.csv", header=None
        # ).values
        ref[i] = pd.read_csv(
            f"{path}/{problem_name}_{emoa}_run_{run}_filling_comp{i+1}_gen{gen}.csv", header=None
        ).values
        eta[i] = pd.read_csv(
            f"{path}/{problem_name}_{emoa}_run_{run}_eta_{i+1}_gen{gen}.csv", header=None
        ).values.ravel()

    all_ref = np.concatenate([v for v in ref.values()], axis=0)
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
    idx = Y_label != -2  # outliers
    x0 = x0[idx]
    y0 = y0[idx]
    Y_label = Y_label[idx]
    y0 = np.array([problem.objective(_) for _ in x0])
    # if the number of clusters of `Y` is more than that of the reference set
    if len(np.unique(Y_label)) > len(ref):
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
    fig_name = f"./plots/{problem_name}_DpN_{emoa}_run{run}_{gen}"
    plot_2d(y0, Y, all_ref, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    # save the data
    df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
    df.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y.csv", index=False)
    df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
    df_y0.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y0.csv", index=False)
    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    return np.array([igd_value, gd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]
if problem_name == "ZDT2" and emoa == "NSGA-III":
    run_id = list(set(run_id) - set([24]))

if 11 < 2:
    for i in [12]:
        execute(i)
        # breakpoint()
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)
    df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
    # df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV"])
    # df.to_csv(f"results/{problem_name}-DpN-{emoa}-{gen}.csv", index=False)
