import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import rcParams
from scipy.io import loadmat

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.newton import DpN
from hvd.zdt import ZDT3, PymooProblemWithAD

plt.style.use("ggplot")
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

max_iters = 8
f = ZDT3()
n_jobs = 30
ref_point = np.array([11, 11])
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front(1000)
path = "./ZDT/ZDT3/"


def plot(y0, Y, ref, hist_Y, history_medroids, hist_IGD, hist_R_norm, fig_name):
    all_ref = np.concatenate([v for v in ref.values()], axis=0)
    medroids0 = np.vstack([m[0] for m in history_medroids])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))
    plt.subplots_adjust(right=0.93, left=0.05)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.4)
    ax0.plot(y0[:, 0], y0[:, 1], "k+", ms=12, alpha=1)
    ax0.plot(all_ref[:, 0], all_ref[:, 1], "b.", mec="none", ms=5, alpha=0.3)
    ax0.plot(medroids0[:, 0], medroids0[:, 1], "r^", mec="none", ms=7, alpha=0.8)
    ax0.set_title("Objective space (Initialization)")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    lgnd = ax0.legend(["Pareto front", r"$Y_0$", "reference set", "matched points"])
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    N = len(y0)
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

    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13]]
    shifts = []
    for i, M in enumerate(history_medroids):
        c = colors[len(M) - 1]
        for j, x in enumerate(M):
            line = ax1.plot(x[0], x[1], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
            if j == len(shifts):
                shifts.append(line)

    lines += shifts
    lines += ax1.plot(Y[:, 0], Y[:, 1], "k*", mec="none", ms=8, alpha=0.9)
    counts = np.unique([len(m) for m in history_medroids], return_counts=True)[1]
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
    ax2.set_title("Performance")
    ax2.set_xlabel("iteration")
    ax2.set_xticks(range(1, max_iters + 1))
    ax2.legend()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=1000)
    plt.close(fig)


def execute(run: int):
    ref_label = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_component_id.csv", header=None).values[0]
    n_cluster = len(np.unique(ref_label))
    # load the reference set
    ref = dict()
    eta = dict()
    for i in range(n_cluster):
        ref[i] = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_filling_comp{i+1}.csv", header=None).values
        eta[i] = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_eta_{i+1}.csv", header=None).values

    # the load the final population from an EMOA
    x0 = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_lastpopu_x.csv", header=None).values[0:50]
    y0 = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_lastpopu_y.csv", header=None).values[0:50]
    Y_label = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_lastpopu_labels.csv", header=None).values.ravel()
    Y_label = Y_label[0:50] - 1
    opt = DpN(
        dim=problem.n_var,
        n_objective=problem.n_obj,
        ref=ref,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        mu=len(x0),
        x0=x0,
        lower_bounds=problem.xl,
        upper_bounds=problem.xu,
        max_iters=max_iters,
        type="igd",
        verbose=False,
        pareto_front=pareto_front,
        eta=eta,
        Y_label=Y_label,
    )
    opt.run()
    Y = opt.Y
    fig_name = f"./figure/{f.__class__.__name__}-run{run}.pdf"
    plot(y0, Y, ref, opt.hist_Y, opt.history_medroids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    gd_func = GenerationalDistance(pareto_front)
    igd_func = InvertedGenerationalDistance(pareto_front)
    return np.array([igd_func.compute(Y=Y), gd_func.compute(Y=Y), hypervolume(Y, ref_point)])


data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in range(1, 31))
df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV"])
df.to_csv(f"{f.__class__.__name__}.csv", index=False)
