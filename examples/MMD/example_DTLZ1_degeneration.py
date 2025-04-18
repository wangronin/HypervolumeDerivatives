import sys

sys.path.insert(0, "./")

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn_extra.cluster import KMedoids

from hvd.bootstrap import bootstrap_reference_set
from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd import MMD
from hvd.mmd_newton import MMDNewton
from hvd.newton import DpN
from hvd.problems import DTLZ1
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated

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

random.seed(66)
np.random.seed(66)
interval = 3
max_iters = 7 * interval + 1

problem = DTLZ1(n_var=7, boundry_constraints=True)
pareto_front = problem.get_pareto_front()
X = pd.read_csv("./data/DTLZ1/DTLZ1_degenerate.csv", header=None).values
Y = np.array([problem.objective(x) for x in X])
km = KMedoids(n_clusters=100, method="alternate", random_state=0, init="k-medoids++").fit(Y)
X0 = X[km.medoid_indices_]
Y0 = Y[km.medoid_indices_]
ref = Y0.copy()
eta = -1 * np.array([1 / np.sqrt(3)] * 3)

N = len(X0)
metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
igd = InvertedGenerationalDistance(pareto_front)
mmd = MMD(n_var=problem.n_var, n_obj=problem.n_obj, ref=pareto_front, theta=1.0 / N)
opt = MMDNewton(
    n_var=problem.n_var,
    n_obj=problem.n_obj,
    ref=ReferenceSet(ref=ref, eta=eta, Y_idx=None),
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
    preconditioning=False,
)
opt.indicator.beta = 0.3  # start with a large spreading effect
X, Y, _, __ = bootstrap_reference_set(
    opt, problem, interval=interval, plot=True, save_reference_set=True, save_population=True
)
ref_new = opt.ref.reference_set - 0.05 * opt.ref.eta[0]

# Several DpN runs for the convergence
metrics = dict(GD=GenerationalDistance(ref=pareto_front), IGD=InvertedGenerationalDistance(ref=pareto_front))
opt = DpN(
    dim=problem.n_var,
    n_obj=problem.n_obj,
    ref=ReferenceSet(ref=ref_new, eta=eta),
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    N=N,
    x0=X,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    verbose=True,
    type="igd",
    metrics=metrics,
)
X, Y, _ = opt.run()
Y = get_non_dominated(Y)
igd_mmd = igd.compute(Y=Y)
mmd_mmd = mmd.compute(Y=Y)

opt_dpn = DpN(
    dim=problem.n_var,
    n_obj=problem.n_obj,
    ref=ReferenceSet(ref=ref, eta=eta),
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
    metrics=metrics,
)
X_DpN, Y_DpN, _ = opt_dpn.run()
Y_DpN = get_non_dominated(Y_DpN)
igd_dpn = igd.compute(Y=Y_DpN)
mmd_dpn = mmd.compute(Y=Y_DpN)

colors = plt.get_cmap("tab20").colors
colors = [colors[2], colors[12], colors[13], colors[15], colors[19]]

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
plt.subplots_adjust(bottom=0.08, top=0.9, right=0.93, left=0.05)
ax0 = fig.add_subplot(1, 3, 1, projection="3d")
ax0.set_box_aspect((1, 1, 1))
ax0.view_init(45, 45)
ax0.plot(Y0[:, 0], Y0[:, 1], Y0[:, 2], "r+", ms=8, alpha=0.6)
ax0.plot(ref[:, 0], ref[:, 1], ref[:, 2], "g.", ms=6, alpha=0.6)
ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.4)

ax0.set_title("Initialization")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.set_zlabel(r"$f_3$")
lgnd = ax0.legend(
    [r"$Y$", "Pareto front"],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.1),
    ncol=2,
    fancybox=True,
)
for handle in lgnd.legend_handles:
    handle.set_markersize(10)

ax1 = fig.add_subplot(1, 3, 2, projection="3d")
ax1.set_box_aspect((1, 1, 1))
ax1.view_init(45, 45)
ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r*", ms=8, alpha=0.6)
ax1.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.4)
ax1.set_title(f"MMD: {mmd_mmd}\n IGD:{igd_mmd}", fontsize=10)
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax1.set_ylabel(r"$f_3$")

ax2 = fig.add_subplot(1, 3, 3, projection="3d")
ax2.set_box_aspect((1, 1, 1))
ax2.view_init(45, 45)
ax2.plot(Y_DpN[:, 0], Y_DpN[:, 1], Y_DpN[:, 2], "r*", ms=8, alpha=0.6)
ax2.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.4)
ax2.set_title(f"MMD: {mmd_dpn}\n IGD:{igd_dpn}", fontsize=10)
ax2.set_xlabel(r"$f_1$")
ax2.set_ylabel(r"$f_2$")
ax2.set_ylabel(r"$f_3$")

plt.tight_layout()
plt.savefig(f"MMD-{problem.__class__.__name__}_degeneration.pdf", dpi=1000)
