import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd_newton import MMDNewton
from hvd.problems import ZDT1, PymooProblemWithAD
from hvd.reference_set import ClusteredReferenceSet

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

max_iters = 5
# problem_name = sys.argv[1]
problem_name = "ZDT1"
print(problem_name)
f = locals()[problem_name](n_var=30)
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front(1000)

path = "./data-reference/ZDT/"
emoa = "SMS-EMOA"
gen = 300
run = 1

# TODO: create a helper function for this snippet
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
X0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_x_gen{gen}.csv", header=None).values
Y0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_y_gen{gen}.csv", header=None).values
Y_label = pd.read_csv(
    f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_labels_gen{gen}.csv", header=None
).values.ravel()
Y_label = Y_label - 1
idx = Y_label != -2  # outliers
X0 = X0[idx]
Y0 = Y0[idx]
Y_label = Y_label[idx]
Y0 = np.array([problem.objective(_) for _ in X0])
# if the number of clusters of `Y` is more than that of the reference set
if len(np.unique(Y_label)) > len(ref):
    ref = np.vstack([r for r in ref.values()])
    Y_label = np.zeros(len(Y0))
    eta = None

n_cluster = len(np.unique(Y_label))
Y_idx = [np.nonzero(Y_label == i)[0] for i in range(n_cluster)]
ref = ClusteredReferenceSet(ref=ref, eta=eta, Y_idx=Y_idx)
metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
opt = MMDNewton(
    n_var=problem.n_var,
    n_obj=problem.n_obj,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    N=len(X0),
    X0=X0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    verbose=True,
    metrics=metrics,
)
X, Y, _ = opt.run()

# fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))
# plt.subplots_adjust(right=0.93, left=0.05)

# ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.4)
# ax0.plot(Y0[:, 0], Y0[:, 1], "k+", ms=12, alpha=1)
# ax0.plot(ref[:, 0], ref[:, 1], "b.", mec="none", ms=5, alpha=0.3)
# ax0.plot(medoids[:, 0], medoids[:, 1], "r^", mec="none", ms=7, alpha=0.8)
# ax0.set_title("Objective space (Initialization)")
# ax0.set_xlabel(r"$f_1$")
# ax0.set_ylabel(r"$f_2$")
# lgnd = ax0.legend(["Pareto front", r"$Y_0$", "reference set", "matched points"])
# for handle in lgnd.legend_handles:
#     handle.set_markersize(10)

# if 1 < 2:
#     trajectory = np.array([Y0] + opt.hist_Y)
#     for i in range(N):
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

# lines = []
# lines += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.3)
# lines += ax1.plot(Y0[:, 0], Y0[:, 1], "k+", ms=12, alpha=0.9)

# colors = plt.get_cmap("tab20").colors
# colors = [colors[2], colors[12], colors[13]]
# shifts = []
# for i, M in enumerate(opt.history_medoids):
#     c = colors[len(M) - 1]
#     for j, x in enumerate(M):
#         line = ax1.plot(x[0], x[1], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
#         if j == len(shifts):
#             shifts.append(line)
# lines += shifts
# lines += ax1.plot(Y[:, 0], Y[:, 1], "k*", mec="none", ms=8, alpha=0.9)
# counts = np.unique([len(m) for m in opt.history_medoids], return_counts=True)[1]
# lgnd = ax1.legend(
#     lines,
#     ["Pareto front", r"$Y_0$"]
#     + [f"{i + 1} shift(s): {k} points" for i, k in enumerate(counts)]
#     + [r"$Y_{\mathrm{final}}$"],
# )
# for handle in lgnd.legend_handles:
#     handle.set_markersize(12)

# ax1.set_title("Objective space")
# ax1.set_xlabel(r"$f_1$")
# ax1.set_ylabel(r"$f_2$")

# # ax22 = ax2.twinx()
# # ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
# ax2.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
# ax2.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
# ax2.set_title("Performance")
# ax2.set_xlabel("iteration")
# ax2.set_xticks(range(1, max_iters + 1))
# # ax2.legend()
# plt.tight_layout()
# plt.savefig(f"{f.__class__.__name__}.pdf", dpi=1000)

# data = [np.c_[[i + 1] * 28, y] for i, y in enumerate(opt.hist_Y)]
# df = pd.DataFrame(np.concatenate(data, axis=0), columns=["iteration", "f1", "f2"])
# df.to_csv("ZDT1_example.csv")
