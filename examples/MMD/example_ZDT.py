import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd_newton import MMDNewton
from hvd.newton import DpN
from hvd.problems import ZDT1, PymooProblemWithAD
from hvd.reference_set import ReferenceSet
from hvd.utils import read_reference_set_data

plt.style.use("ggplot")
rcParams["font.size"] = 15
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

max_iters = 15
problem_name = "ZDT1"
print(problem_name)
f = locals()[problem_name](n_var=3)
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front(1000)

if 1 < 2:
    ref_ = pd.read_csv("./ZDT1/ZDT1_REF_Filling.csv", header=None).values
    medoids = pd.read_csv("./ZDT1/ZDT1_REF_Match_30points.csv", header=None).values
    # the load the final population from an EMOA
    X0 = pd.read_csv("./ZDT1/ZDT1_Pop_x.csv", header=None).values
    Y0 = pd.read_csv("./ZDT1/ZDT1_Pop_y.csv", header=None).values
    eta = {0: pd.read_csv("./ZDT1/ZDT1_eta.csv", header=None).values.ravel()}
    Y_idx = None
    # path = "./data-reference/ZDT/"
    # emoa = "SMS-EMOA"
    # gen = 300
    # run = 1
    # ref_, eta, X0, Y0, Y_idx = read_reference_set_data(path, problem_name, emoa, run, gen)
else:
    ref_ = problem.get_pareto_front(15)
    X0 = problem.get_pareto_set(15, kind="linear")
    X0[:, 1] += 0.02
    Y0 = np.array([problem.objective(x) for x in X0])
    eta, Y_idx = {0: np.array([-0.70710678, -0.70710678])}, None

N = len(X0)
ref = ReferenceSet(ref=ref_, eta=eta, Y_idx=Y_idx)
metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
opt = MMDNewton(
    n_var=problem.n_var,
    n_obj=problem.n_obj,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    N=N,
    X0=X0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    verbose=True,
    metrics=metrics,
    preconditioning=True,
)
X, Y, _ = opt.run()

opt2 = DpN(
    dim=problem.n_var,
    n_obj=problem.n_obj,
    ref=ref_,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    N=N,
    x0=X0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    verbose=True,
    type="igd",
    eta=eta,
    Y_label=None,
    pareto_front=pareto_front,
)
X_DpN, Y_DpN, _ = opt2.run()

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 12))
plt.subplots_adjust(right=0.93, left=0.03, wspace=0.07, hspace=0.03)

# ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.4)
# ax0.plot(Y0[:, 0], Y0[:, 1], "k+", ms=12, alpha=1)
# ax0.plot(ref_[:, 0], ref_[:, 1], "b.", mec="none", ms=5, alpha=0.3)
# ax0.plot(medoids[:, 0], medoids[:, 1], "r^", mec="none", ms=7, alpha=0.8)
# ax0.set_title("Objective space (Initialization)")
# ax0.set_xlabel(r"$f_1$")
# ax0.set_ylabel(r"$f_2$")
# lgnd = ax0.legend(["Pareto front", r"$Y_0$", "reference set", "matched points"])
# for handle in lgnd.legend_handles:
#     handle.set_markersize(10)

if 1 < 2:
    trajectory = np.array([Y0] + opt.history_Y)
    for i in range(N):
        x, y = trajectory[:, i, 0], trajectory[:, i, 1]
        ax0.quiver(
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

    trajectory = np.array([Y0] + opt2.history_Y)
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

medoids = opt.ref.reference_set
for i, m in enumerate(medoids):
    ax0.plot((m[0], Y[i, 0]), (m[1], Y[i, 1]), "k--", alpha=0.5)

lines = []
lines += ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.3)
lines += ax0.plot(Y0[:, 0], Y0[:, 1], "k+", ms=12, alpha=0.9)

colors = plt.get_cmap("tab20").colors
colors = [colors[2], colors[12], colors[13]]
shifts = []
for i, M in opt.history_medoids.items():
    c = colors[len(M) - 1]
    for j, x in enumerate(M):
        line = ax0.plot(x[0], x[1], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
        if j == len(shifts):
            shifts.append(line)
lines += shifts
lines += ax0.plot(Y[:, 0], Y[:, 1], "k*", mec="none", ms=8, alpha=0.9)
counts = np.unique([len(m) for m in opt.history_medoids.values()], return_counts=True)[1]
lgnd = ax0.legend(
    lines,
    ["Pareto front", r"$Y_0$"]
    + [f"{i + 1} shift(s): {k} points" for i, k in enumerate(counts)]
    + [r"$Y_{\mathrm{MMD}}$"],
)
for handle in lgnd.legend_handles:
    handle.set_markersize(12)

ax0.set_title("MMD-Newton")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")

medoids = opt2._igd._medoids
for i, m in enumerate(medoids):
    ax1.plot((m[0], Y_DpN[i, 0]), (m[1], Y_DpN[i, 1]), "k--", alpha=0.5)

lines = []
lines += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.3)
lines += ax1.plot(Y0[:, 0], Y0[:, 1], "k+", ms=12, alpha=0.9)

colors = plt.get_cmap("tab20").colors
colors = [colors[2], colors[12], colors[13]]
shifts = []
for i, M in enumerate(opt2.history_medoids):
    c = colors[len(M) - 1]
    for j, x in enumerate(M):
        line = ax1.plot(x[0], x[1], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
        if j == len(shifts):
            shifts.append(line)
lines += shifts
lines += ax1.plot(Y_DpN[:, 0], Y_DpN[:, 1], "k*", mec="none", ms=8, alpha=0.9)
counts = np.unique([len(m) for m in opt2.history_medoids], return_counts=True)[1]
lgnd = ax1.legend(
    lines,
    ["Pareto front", r"$Y_0$"]
    + [f"{i + 1} shift(s): {k} points" for i, k in enumerate(counts)]
    + [r"$Y_{\mathrm{DpN}}$"],
)
for handle in lgnd.legend_handles:
    handle.set_markersize(12)

ax1.set_title("DpN")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

xticks = range(1, len(opt.history_indicator_value) + 1)
ax2.semilogy(xticks, opt.history_R_norm, "k--", label=r"$||R(\mathbf{X})||_{\mathrm{MMD}}$")
ax2.semilogy(xticks, opt2.history_R_norm, "r--", label=r"$||R(\mathbf{X})||_{\mathrm{DpN}}$")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
ax2.legend()

# ax32 = ax3.twinx()
# ax3.semilogy(xticks, opt.history_indicator_value, "r-", label="MMD-matching")
# for i, (name, values) in enumerate(opt.history_metrics.items()):
# ax3.semilogy(xticks, values, color=colors[i], ls="solid", label=name)
ax3.semilogy(xticks, opt.history_metrics["IGD"], color=colors[0], ls="solid", label="IGD-MMD")
# ax32.semilogy(xticks, opt.history_metrics["IGD"], color=colors[1], ls="solid", label="IGD-MMD")

ax3.semilogy(xticks, opt2.hist_IGD, color=colors[0], ls="dashed", label="IGD-DpN")
# ax32.semilogy(xticks, opt2.hist_IGD, color=colors[1], ls="dashed", label="IGD-DpN")
ax3.set_title("Performance")
# ax3.set_ylabel("GD")
# ax32.set_ylabel("IGD")
ax3.legend()
# ax32.legend()

plt.tight_layout()
plt.savefig(f"MMD-{f.__class__.__name__}.pdf", dpi=1000)
