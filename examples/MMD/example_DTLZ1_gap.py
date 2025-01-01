import sys

sys.path.insert(0, "./")

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.bootstrap import bootstrap_reference_set
from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd_newton import MMDNewton
from hvd.newton import DpN
from hvd.problems import DTLZ1
from hvd.reference_set import ClusteredReferenceSet
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
max_iters = 11

problem = DTLZ1(n_var=7, boundry_constraints=True)
pareto_front = problem.get_pareto_front()

ref_ = pd.read_csv("./data/DTLZ1/DTLZ1_RANDOM_HOLE_run_1_ref_1_gen0.csv", header=None).values
X0 = pd.read_csv("./data/DTLZ1/DTLZ1_RANDOM_HOLE_run_1_lastpopu_x_gen0.csv", header=None).values
Y0 = pd.read_csv("./data/DTLZ1/DTLZ1_RANDOM_HOLE_run_1_lastpopu_y_gen0.csv", header=None).values
# retrieve two components manually
idx = Y0[:, 2] >= 0.307
X_component1 = X0[idx]
Y_component1 = Y0[idx]
ref1_ = pareto_front[pareto_front[:, 2] >= 0.307]
# retrieve two components manually
idx = Y0[:, 2] <= 0.2
X_component2 = X0[idx]
Y_component2 = Y0[idx]
ref2_ = pareto_front[pareto_front[:, 2] <= 0.2]

# NOTE: for bootstrapping, the spread looks better
# if we used a clustered reference set to represent the gap
ref = {0: ref1_, 1: ref2_}
eta = {0: -1 * np.array([1 / np.sqrt(3)] * 3), 1: -1 * np.array([1 / np.sqrt(3)] * 3)}
Y_idx = [list(range(0, len(Y_component1))), list(range(len(Y_component1), len(Y0)))]
Y_label = np.array([0] * len(X_component1) + [1] * len(X_component2))

N = len(X_component2)
metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
igd = InvertedGenerationalDistance(pareto_front)
opt = MMDNewton(
    n_var=problem.n_var,
    n_obj=problem.n_obj,
    ref=ClusteredReferenceSet(ref=ref2_, eta=None, Y_idx=None),
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    N=N,
    X0=X_component2,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    verbose=True,
    metrics=metrics,
    preconditioning=False,
)
if 1 < 2:
    opt.indicator.beta = 0.2  # start with a large spreading effect
    X, Y, _ = bootstrap_reference_set(opt, problem, 3)
else:
    X, Y, _ = opt.run()

X2, Y2 = X, Y
ref2 = opt.indicator.ref.reference_set

N = len(X_component1)
metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
igd = InvertedGenerationalDistance(pareto_front)
opt = MMDNewton(
    n_var=problem.n_var,
    n_obj=problem.n_obj,
    ref=ClusteredReferenceSet(ref=ref1_, eta=None, Y_idx=None),
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    N=N,
    X0=X_component1,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    verbose=True,
    metrics=metrics,
    preconditioning=False,
)
if 1 < 2:
    opt.indicator.beta = 0.2  # start with a large spreading effect
    X, Y, _ = bootstrap_reference_set(opt, problem, 3)
else:
    X, Y, _ = opt.run()

X1, Y1 = X, Y
ref1 = opt.indicator.ref.reference_set
X_MMD = np.r_[X1, X2]
Y_MMD = np.r_[Y1, Y2]
ref_MMD = np.r_[ref1, ref2]

N = len(X_MMD)
metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
igd = InvertedGenerationalDistance(pareto_front)
opt = MMDNewton(
    n_var=problem.n_var,
    n_obj=problem.n_obj,
    ref=ClusteredReferenceSet(ref=ref_MMD, eta=None, Y_idx=None),
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    N=N,
    X0=X_MMD,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=5,
    verbose=True,
    metrics=metrics,
    preconditioning=False,
)
X, Y, _ = opt.run()
Y = get_non_dominated(Y)
igd_mmd = igd.compute(Y=Y)

opt_dpn = DpN(
    dim=problem.n_var,
    n_obj=problem.n_obj,
    ref=ref,
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
    Y_label=Y_label,
    pareto_front=pareto_front,
)
X_DpN, Y_DpN, _ = opt_dpn.run()
Y_DpN = get_non_dominated(Y_DpN)
igd_dpn = igd.compute(Y=Y_DpN)

colors = plt.get_cmap("tab20").colors
colors = [colors[2], colors[12], colors[13], colors[15], colors[19]]

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
plt.subplots_adjust(bottom=0.08, top=0.9, right=0.93, left=0.05)
ax0 = fig.add_subplot(1, 3, 1, projection="3d")
ax0.set_box_aspect((1, 1, 1))
ax0.view_init(45, 45)
ax0.plot(Y0[:, 0], Y0[:, 1], Y0[:, 2], "r+", ms=8, alpha=0.6)
ax0.plot(ref1_[:, 0], ref1_[:, 1], ref1_[:, 2], "g.", ms=6, alpha=0.6)
ax0.plot(ref2_[:, 0], ref2_[:, 1], ref2_[:, 2], "g.", ms=6, alpha=0.6)
ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.4)

ax0.set_title("Initialization")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.set_zlabel(r"$f_3$")
lgnd = ax0.legend(
    [r"$Y_0$", "reference set", "Pareto front"],
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
ax1.set_title(f"MMD's IGD: {igd_mmd}")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax1.set_ylabel(r"$f_3$")

ax1 = fig.add_subplot(1, 3, 3, projection="3d")
ax1.set_box_aspect((1, 1, 1))
ax1.view_init(45, 45)
ax1.plot(Y_DpN[:, 0], Y_DpN[:, 1], Y_DpN[:, 2], "r*", ms=8, alpha=0.6)
ax1.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.4)
ax1.set_title(f"DpN's IGD: {igd_dpn}")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax1.set_ylabel(r"$f_3$")

plt.tight_layout()
plt.savefig(f"MMD-{problem.__class__.__name__}_gap.pdf", dpi=1000)
