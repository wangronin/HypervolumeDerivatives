import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import HVN

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 13
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 13
rcParams["ytick.labelsize"] = 13
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)


def concave(x):
    return np.array(x)


def concave_Jacobian(x):
    return np.eye(2)


def concave_Hessian(x):
    return np.zeros((2, 2))


def g(x):
    return 1 - x[0] ** 2 - x[1]


def g_Jacobian(x):
    return np.r_[-2.0 * x[0], -1.0]


def g_Hessian(x):
    return np.array([[-2.0, 0.0], [0.0, 0.0]])


ref = np.array([1, 1])
max_iters = 10
x1 = pd.read_csv("convex_concave/concave_x.csv", header=None, index_col=None).values
x2 = pd.read_csv("convex_concave/concave_y.csv", header=None, index_col=None).values
x0 = np.c_[x1, x2]
y0 = x0.copy()
N = len(x0)
opt = HVN(
    dim=2,
    n_objective=2,
    ref=ref,
    func=concave,
    jac=concave_Jacobian,
    hessian=concave_Hessian,
    h=g,
    h_jac=g_Jacobian,
    h_hessian=g_Hessian,
    mu=len(x0),
    x0=x0,
    lower_bounds=0,
    upper_bounds=1,
    minimization=True,
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax0.set_aspect("equal")
ax0.plot(Y[:, 0], Y[:, 1], "r*", ms=8)
ax0.plot(y0[:, 0], y0[:, 1], "k+", ms=8)
trajectory = np.array([y0] + opt.hist_Y)

if 11 < 2:
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

x_vals = np.linspace(0, 1, 100)
y_vals = 1 - x_vals**2
ax0.plot(x_vals, y_vals, "k--", alpha=0.5)
ax0.set_title("Objective space")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.legend([r"$Y_{\text{final}}$", r"$Y_0$", "Pareto front"])

ax22 = ax1.twinx()
ax1.plot(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
ax22.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_G_norm, "g--")
ax1.set_ylabel("HV", color="b")
ax22.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax1.set_title("Performance")
ax1.set_xlabel("iteration")
ax1.set_xticks(range(1, max_iters + 1))

plt.savefig(f"concave-example-{N}.pdf", dpi=1000)
