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
from hvd.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, PymooProblemWithAD

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

max_iters = 10
f = ZDT1()
n_jobs = 30
ref_point = np.array([11, 11])
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front(1000)

data = loadmat("./data/ZDT/ZDT1_NSGA-II.mat")
columns = (
    ["run", "iteration"]
    + [f"x{i}" for i in range(1, problem.n_var + 1)]
    + [f"f{i}" for i in range(1, problem.n_obj + 1)]
)
data = pd.DataFrame(data["data"], columns=columns)


def plot(y0, Y, ref, hist_Y, history_medroids, hist_IGD, hist_R_norm, fig_name):
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
    # lines += ax1.plot(y0[:, 0], y0[:, 1], "k+", ms=12, alpha=0.9)

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
        ["Pareto front"]  # , r"$Y_0$"]
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
    fig.close()


def execute(run: int):
    # load the reference set
    ref = pd.read_csv(f"./data-reference-set/ZDT/ZDT1_NSGA-II_run_{run}_ref.csv", header=None).values
    # the load the final population from an EMOA
    df = data[(data.run == run) & (data.iteration == 999)]
    x0 = df.loc[:, "x1":f"x{problem.n_var}"].iloc[:50, :].values
    y0 = df.loc[:, "f1":f"f{problem.n_obj}"].iloc[:50, :].values

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
        pareto_front=problem.get_pareto_front(500),
    )
    opt.run()
    Y = opt.Y
    fig_name = f"./figure/{f.__class__.__name__}-run{run}.pdf"
    plot(y0, Y, ref, opt.hist_Y, opt.history_medroids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    gd_func = GenerationalDistance(pareto_front)
    igd_func = InvertedGenerationalDistance(pareto_front)
    return np.array([igd_func.compute(Y), gd_func.compute(Y), hypervolume(Y, ref_point)])


data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in range(1, 31))
df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV"])
df.to_csv(f"{f.__class__.__name__}.csv", index=False)
