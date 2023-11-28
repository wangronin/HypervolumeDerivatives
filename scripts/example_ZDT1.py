import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

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

max_iters = 13
f = ZDT1(n_var=3)
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front(1000)

# load the reference set
ref = pd.read_csv("./ZDT/ZDT1/ZDT1_REF_Filling.csv", header=None).values
medroids = pd.read_csv("./ZDT/ZDT1/ZDT1_REF_Match_30points.csv", header=None).values
# the load the final population from an EMOA
x0 = pd.read_csv("./ZDT/ZDT1/ZDT1_Pop_x.csv", header=None).values
y0 = pd.read_csv("./ZDT/ZDT1/ZDT1_Pop_y.csv", header=None).values
N = len(x0)

opt = DpN(
    dim=problem.n_var,
    n_objective=problem.n_obj,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    mu=N,
    x0=x0,
    lower_bounds=problem.xl,
    upper_bounds=problem.xu,
    max_iters=max_iters,
    type="igd",
    verbose=True,
    pareto_front=pareto_front,
)
opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)

ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.4)
ax0.plot(y0[:, 0], y0[:, 1], "k+", ms=12, alpha=1)
ax0.plot(ref[:, 0], ref[:, 1], "b.", mec="none", ms=5, alpha=0.3)
ax0.plot(medroids[:, 0], medroids[:, 1], "r^", mec="none", ms=7, alpha=0.8)
ax0.set_title("Objective space (Initialization)")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
lgnd = ax0.legend(["Pareto front", r"$Y_0$", "reference set", "matched points"])
for handle in lgnd.legend_handles:
    handle.set_markersize(10)

X = opt._get_primal_dual(opt.X)[0]
Y = opt.Y
if 1 < 2:
    trajectory = np.array([y0] + opt.hist_Y)
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
lines += ax1.plot(y0[:, 0], y0[:, 1], "k+", ms=12, alpha=0.9)

colors = plt.get_cmap("tab20").colors
colors = [colors[2], colors[12], colors[13]]
shifts = []
for i, M in enumerate(opt.history_medroids):
    c = colors[len(M) - 1]
    for j, x in enumerate(M):
        line = ax1.plot(x[0], x[1], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
        if j == len(shifts):
            shifts.append(line)
lines += shifts
lines += ax1.plot(Y[:, 0], Y[:, 1], "k*", mec="none", ms=8, alpha=0.9)
counts = np.unique([len(m) for m in opt.history_medroids], return_counts=True)[1]
lgnd = ax1.legend(
    lines,
    ["Pareto front", r"$Y_0$"]
    + [f"{i + 1} shift(s): {k} points" for i, k in enumerate(counts)]
    + [r"$Y_{\mathrm{final}}$"],
)
for handle in lgnd.legend_handles:
    handle.set_markersize(12)

ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

# ax22 = ax2.twinx()
# ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax2.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
ax2.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
ax2.legend()
plt.tight_layout()
plt.savefig(f"{f.__class__.__name__}.pdf", dpi=1000)

# data = [np.c_[[i + 1] * 28, y] for i, y in enumerate(opt.hist_Y)]
# df = pd.DataFrame(np.concatenate(data, axis=0), columns=["iteration", "f1", "f2"])
# df.to_csv("ZDT1_example.csv")
