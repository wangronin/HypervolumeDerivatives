import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import rcParams
from scipy.io import loadmat

from hvd.delta_p import InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9

np.random.seed(66)

n_reps = 30
n_jobs = n_reps
max_iters = 5
problem = CF1()
pareto_front = problem.get_pareto_front(1000)
igd_func = InvertedGenerationalDistance(pareto_front)


def plot_trajectory(y0, y, ref, medriods, opt, run_id):
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

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))
    plt.subplots_adjust(right=0.93, left=0.05)

    ax0.plot(y0[:, 0], y0[:, 1], "r.", ms=7, alpha=0.5)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=4, alpha=0.2)
    ax0.plot(ref[:, 0], ref[:, 1], "b.", ms=4, mec="none", alpha=0.3)
    # ax1.plot(y0[:, 0], y0[:, 1], "r.", ms=7, alpha=0.5)
    ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=4, alpha=0.2)
    ax1.plot(ref[:, 0], ref[:, 1], "b.", ms=4, mec="none", alpha=0.3)
    ax1.plot(medriods[:, 0], medriods[:, 1], "r^", ms=7, alpha=0.5)
    ax1.plot(y[:, 0], y[:, 1], "r.", ms=7, alpha=0.5)

    # if 1 < 2:
    #     trajectory = np.array([y0] + opt.hist_Y)
    #     for i in range():
    #         x, y = trajectory[:, i, 0], trajectory[:, i, 1]
    #         ax1.quiver(
    #             x[:-1],
    #             y[:-1],
    #             x[1:] - x[:-1],
    #             y[1:] - y[:-1],
    #             scale_units="xy",
    #             angles="xy",
    #             scale=1,
    #             color="k",
    #             width=0.003,
    #             alpha=0.5,
    #             headlength=4.5,
    #             headwidth=2.5,
    #         )

    ax1.set_title("Objective space")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")

    ax22 = ax2.twinx()
    ax2.semilogy(range(1, len(opt.hist_GD) + 1), opt.hist_GD, "b-", label="GD")
    ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
    ax22.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
    ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    ax2.set_title("Performance")
    ax2.set_xlabel("iteration")
    ax2.set_xticks(range(1, max_iters + 1))
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{problem.__class__.__name__}-run{run_id}.pdf", dpi=1000)
    plt.close(fig)


def run(id: int, verbose: bool = True):
    # load the reference set
    ref = pd.read_csv(f"./data-reference-set/CF/CF4_GDE3_run_{id}_ref.csv", header=0).values
    delta = 0.1
    # the load the final population from an EMOA
    data = loadmat("./data/CF4_GDE3.mat")
    columns = (
        ["run", "iteration"]
        + [f"x{i}" for i in range(1, problem.n_decision_vars + 1)]
        + [f"f{i}" for i in range(1, problem.n_objectives + 1)]
        + [f"h{i}" for i in range(1, problem.n_eq_constr + 1)]
        + [f"g{i}" for i in range(1, problem.n_ieq_constr + 1)]
    )
    df = pd.DataFrame(data["data"], columns=columns)
    max_iter = df.iteration.max()
    df = df[(df.run == id) & (df.iteration == max_iter)]
    x0 = df.loc[:, "x1":f"x{problem.n_decision_vars}"].iloc[:, :].values
    y0 = df.loc[:, "f1":f"f{problem.n_objectives}"].iloc[:, :].values
    # measure the performance of the input approximation set
    igd_init = igd_func.compute(Y=y0)
    cstr_init = np.mean([problem.constraint(x) for x in x0])

    N = len(x0)
    opt = DpN(
        dim=problem.n_decision_vars,
        n_objective=problem.n_objectives,
        ref=ref,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        g=problem.constraint,
        g_jac=problem.constraint_jacobian,
        mu=N,
        x0=x0,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        max_iters=max_iters,
        type="igd",
        verbose=verbose,
    )

    while not opt.terminate():
        ref -= delta
        delta *= 0.5  # exponential decay of the shift
        opt.reference_set = ref
        opt.newton_iteration()
        opt.log()

    # plot the result
    plot_trajectory(y0, opt.Y, ref, opt.active_indicator._medroids, opt, id)
    # measure the performance after optimization
    X = opt._get_primal_dual(opt.X)[0]
    Y = opt.Y
    igd_final = igd_func.compute(Y=Y)
    cstr_final = np.mean([problem.constraint(x) for x in X])
    return np.array([igd_init, igd_final, cstr_init, cstr_final])


data = Parallel(n_jobs=n_jobs)(delayed(run)(id=i, verbose=False) for i in range(1, n_reps + 1))
df = pd.DataFrame(np.array(data), columns=["initial IGD", "final IGD", "initial cstr", "final cstr"])
df.to_csv(f"{problem.__class__.__name__}.csv", index=False)
