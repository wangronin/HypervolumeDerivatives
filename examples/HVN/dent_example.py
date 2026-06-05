import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import HVN
from hvd.problems.misc import DENT

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 20
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

ref = np.array([5, 5])
max_iters = 10
problem = DENT()
N = 30
p = np.linspace(-1.6, 1.9, N)
X0 = np.c_[p, -p + 0.3]
Y0 = np.array([problem.objective(_) for _ in X0])
# X0 = problem.get_pareto_set(30)
# Y0 = np.array([problem.objective(x) for x in X0])
pareto_front = problem.get_pareto_front(1000)
N = len(X0)

opt = HVN(
    n_var=2,
    n_obj=2,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    N=N,
    X0=X0,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    verbose=True,
    preconditioning=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 22), layout="compressed")
ax0.set_box_aspect(1)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)

x = np.linspace(-2, 2, 400)
y = -x
lines = []
lines += ax0.plot(X[:, 0], X[:, 1], "r+", ms=12)
lines += ax0.plot(X0[:, 0], X0[:, 1], "k.", ms=8, clip_on=False)
lines += ax0.plot(x, y, "k--", clip_on=False)
ax0.set_xlim([-2, 2])
ax0.set_ylim([-2, 2])
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")
ax0.legend(lines, [r"$X_{\text{final}}$", r"$X_0$", r"Pareto set"])

n_per_axis = 30
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
Z = np.array([problem.objective(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z1 = Z[:, 0].reshape(-1, len(x))
Z2 = Z[:, 1].reshape(-1, len(x))
CS1 = ax0.contour(X1, X2, Z1, 10, cmap=plt.cm.Blues, linewidths=1, alpha=0.8)
CS2 = ax0.contour(X1, X2, Z2, 10, cmap=plt.cm.Reds, linewidths=1, alpha=0.8)

trajectory = np.array([X0] + opt.history_X)
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
        width=0.005,
        alpha=0.5,
        headlength=4.7,
        headwidth=2.7,
    )

lines = []
lines += ax1.plot(Y[:, 0], Y[:, 1], "r+", ms=13)
lines += ax1.plot(Y0[:, 0], Y0[:, 1], "k.", ms=8)
lines += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "k--")
ax1.legend(lines, [r"$Y_{\text{final}}$", r"$Y_0$", r"Pareto front"])
trajectory = np.array([Y0] + opt.history_Y)

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

ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax22 = ax2.twinx()
ax22.set_box_aspect(1)
ax2.plot(range(1, len(opt.history_indicator_value) + 1), opt.history_indicator_value, "b-")
ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
ax2.set_ylabel("")
ax22.set_ylabel("")
ax2.text(0, 1.01, r"$\operatorname{HV}$", transform=ax2.transAxes, color="b", ha="right", va="bottom")
ax22.text(
    1.12, 1.01, r"$||R(\mathbf{X}, \lambda)||$", transform=ax2.transAxes, color="g", ha="right", va="bottom"
)
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
plt.savefig(f"dent-example-HVN-{N}.pdf", dpi=1000)

data = pd.DataFrame(
    np.c_[np.arange(len(opt.history_indicator_value)), opt.history_indicator_value, opt.history_R_norm],
    columns=["Iter", r"$\operatorname{HV}$", r"$\|R(\mathbf{X}, \lambda)\|$"],
)
caption = r"""Convergence of the hypervolume value and the root-finding error $\|R_I(\mathbf{X}, \lambda)\|$.
"""
data.to_latex(
    f"dent-example-HVN-{N}.tex",
    index=False,
    escape=False,
    caption=caption,
    column_format="c|c|c",
    formatters={
        data.columns[0]: lambda x: f"{int(x)}",
        data.columns[1]: lambda x: f"{x:.12f}",
        data.columns[2]: lambda x: f"{x:.12f}",
    },
)
