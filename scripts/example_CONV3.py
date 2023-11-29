import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.spatial.distance import cdist

from hvd.newton import DpN
from hvd.problems.problems import CONV3
from hvd.utils import non_domin_sort

plt.style.use("ggplot")
rcParams["font.size"] = 17
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["legend.fontsize"] = 10
rcParams["xtick.labelsize"] = 17
rcParams["ytick.labelsize"] = 17
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

max_iters = 10
problem = CONV3()
pareto_front = problem.get_pareto_front(5000)

# load the reference set
path = "./CONV3_example/"
ref = pd.read_csv(path + f"CONV3_NSGA-II_run_1_ref.csv", header=None).values
idx = non_domin_sort(ref, only_front_indices=True)[0]
ref = ref[idx]
idx = cdist(pareto_front, ref).min(axis=0).argsort()
idx = idx[0:-10]
ref = ref[idx]
# the load the final population from an EMOA
x0 = pd.read_csv(path + f"CONV3_NSGA-II_run_1_lastpopu_x.csv", header=None).values[:100]
y0 = pd.read_csv(path + f"CONV3_NSGA-II_run_1_lastpopu_y.csv", header=None).values[:100]
N = len(x0)

opt = DpN(
    dim=problem.n_var,
    n_objective=problem.n_obj,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    # g=problem.ieq_constraint,
    # g_jac=problem.ieq_jacobian,
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
medroids0 = np.vstack([m[0] for m in opt.history_medroids])
X = opt._get_primal_dual(opt.X)[0]
Y = opt.Y

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
ax0 = fig.add_subplot(1, 3, 1, projection="3d")
ax0.set_box_aspect((1, 1, 1))
ax0.view_init(70, -20)

ax0.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k.", ms=12, alpha=1)
ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.3)
ax0.plot(ref[:, 0], ref[:, 1], ref[:, 2], "b.", mec="none", ms=5, alpha=0.2)
ax0.plot(medroids0[:, 0], medroids0[:, 1], medroids0[:, 2], "r^", mec="none", ms=7, alpha=0.8)

ax0.set_title("Objective space (Initialization)")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.set_ylabel(r"$f_3$")

lgnd = ax0.legend(
    [r"$Y_0$", "Pareto front", "reference set", "matched points"],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=2,
    fancybox=True,
)
for handle in lgnd.legend_handles:
    handle.set_markersize(10)

ax1 = fig.add_subplot(1, 3, 2, projection="3d")
ax1.set_box_aspect((1, 1, 1))
ax1.view_init(50, -20)

if 11 < 2:
    trajectory = np.array([y0] + opt.hist_Y)
    for i in range(N):
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
lines += ax1.plot(
    pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.2
)

colors = plt.get_cmap("tab20").colors
colors = [colors[2], colors[12], colors[13]]
shifts = []
for i, M in enumerate(opt.history_medroids):
    c = colors[len(M) - 1]
    for j, x in enumerate(M):
        line = ax1.plot(x[0], x[1], x[2], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
        if j == len(shifts):
            shifts.append(line)
lines += shifts
lines += ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "k*", mec="none", ms=8, alpha=0.9)
counts = np.unique([len(m) for m in opt.history_medroids], return_counts=True)[1]
# lgnd = ax1.legend(
#     handles=lines,
#     labels=["Pareto front"]
#     + [f"{i + 1} shift(s): {k} points" for i, k in enumerate(counts)]
#     + [r"$Y_{\mathrm{final}}$"],
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.14),
#     ncol=2,
#     fancybox=True,
# )
# for handle in lgnd.legend_handles:
#     handle.set_markersize(10)

ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax0.set_ylabel(r"$f_3$")

ax2 = fig.add_subplot(1, 3, 3)
ax2.set_aspect("equal")
ax22 = ax2.twinx()
ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax22.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
ax2.legend()
plt.tight_layout()
plt.show()
plt.savefig(f"{problem.__class__.__name__}.pdf", dpi=1000)

breakpoint()
data = np.concatenate([np.c_[[0] * N, y0], np.c_[[max_iters] * N, opt.hist_Y[-1]]], axis=0)
df = pd.DataFrame(data, columns=["iteration", "f1", "f2", "f3"])
df.to_csv("CONV3_example.csv")
