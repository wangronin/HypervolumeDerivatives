import random
import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import DpN
from hvd.problems import CF2

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
problem = CF2()
pareto_front = CF2.get_pareto_front(500)

df = pd.read_csv("./examples/DpN/CF2_x0.csv", header=0)
x0 = df.iloc[:, 0:10].values
y0 = df.iloc[:, 10:].values
idx = np.nonzero((y0[:, 0] < 1) & (y0[:, 1] < 1))[0]
x0 = x0[idx]
y0 = y0[idx]
x0[:, 0] += 0.01
N = len(x0)
y0 = np.array([problem.objective(x) for x in x0])
ref = pd.read_csv("./examples/DpN/CF2_refset_nofillmeans_shifted.csv", header=None).values
ref -= 0.03
pareto_front = problem.get_pareto_front(N=200)

if 11 < 2:
    ref = pareto_front - 0.03

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax1.plot(ref[:, 0], ref[:, 1], "k.", ms=3)
ax1.plot(y0[:, 0], y0[:, 1], "r+")
ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "r.", ms=1)

opt = DpN(
    dim=problem.n_decision_vars,
    n_obj=problem.n_objectives,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.eq_constraint,
    g_jac=problem.eq_jacobian,
    N=N,
    x0=x0,
    xl=problem.lower_bounds,
    xu=problem.upper_bounds,
    max_iters=max_iters,
    type="igd",
    pareto_front=pareto_front,
    verbose=True,
)

delta = 0.02
while not opt.terminate():
    ref -= delta
    delta *= 0.5
    opt.reference_set = ref
    opt.newton_iteration()
    opt.log()
    # exponential decay of the shift


X = opt._get_primal_dual(opt.X)[0]
Y = opt.Y
# X, Y, stop = opt.run()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5))
# plt.subplots_adjust(right=0.93, left=0.05)
# ax1.plot(ref[:, 0], ref[:, 1], "k.")
# ax1.plot(x0[:, 0], x0[:, 1], "r.")
# ciricle = plt.Circle((0, 0), 1, color="r", fill=False, ls="--", lw=1.5)

# ax0.plot(X[:, 0], X[:, 1], "g*")
# ax0.plot(x0[:, 0], x0[:, 1], "g.", ms=8, clip_on=False)
# ax0.add_patch(ciricle)
# ax0.set_xlim([-2, 2])
# ax0.set_ylim([-2, 2])
# ax0.set_title("Decision space")
# ax0.set_xlabel(r"$x_1$")
# ax0.set_ylabel(r"$x_2$")


# n_per_axis = 30
# x = np.linspace(-2, 2, n_per_axis)
# X1, X2 = np.meshgrid(x, x)
# Z = np.array([problem.objective(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
# Z1 = Z[:, 0].reshape(-1, len(x))
# Z2 = Z[:, 1].reshape(-1, len(x))
# CS1 = ax0.contour(X1, X2, Z1, 10, cmap=plt.cm.gray, linewidths=0.8, alpha=0.6)
# CS2 = ax0.contour(X1, X2, Z2, 10, cmap=plt.cm.gray, linewidths=0.8, linestyles="--", alpha=0.6)

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
            width=0.005,
            alpha=0.5,
            headlength=4.7,
            headwidth=2.7,
        )

x_vals = np.array([0, 6])
y_vals = 6 - x_vals
ax1.plot(Y[:, 0], Y[:, 1], "g*", ms=5)
# ax1.plot(y0[:, 0], y0[:, 1], "g.", ms=8)
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

plt.savefig(f"2D-CF2-{N}.pdf", dpi=1000)

data = np.concatenate([np.c_[[0] * N, y0], np.c_[[max_iters] * N, opt.hist_Y[-1]]], axis=0)
df = pd.DataFrame(data, columns=["iteration", "f1", "f2"])
df.to_csv("CF2_example.csv")
