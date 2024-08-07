import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import HVN

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["text.usetex"] = True
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


def convex(x):
    return np.array(x)


def convex_Jacobian(x):
    return np.eye(2)


def convex_Hessian(x):
    return np.zeros((2, 2))


def g(x):
    return x[0] ** 2 - 2 * x[0] + 1 - x[1]


def g_Jacobian(x):
    return np.array([2 * x[0] - 2, -1])


def g_Hessian(x):
    return np.array([[2.0, 0], [0, 0]])


ref = np.array([1, 1])
max_iters = 10

x1 = pd.read_csv("convex_concave/convex_x.csv", header=None, index_col=None).values
x2 = pd.read_csv("convex_concave/convex_y.csv", header=None, index_col=None).values
X0 = np.c_[x1, x2]
y0 = X0.copy()
N = len(X0)
opt = HVN(
    n_var=2,
    n_obj=2,
    ref=ref,
    func=convex,
    jac=convex_Jacobian,
    hessian=convex_Hessian,
    g=g,
    g_jac=g_Jacobian,
    g_hessian=g_Hessian,
    N=len(X0),
    X0=X0,
    xl=0,
    xu=1,
    max_iters=max_iters,
    verbose=True,
    preconditioning=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax0.set_aspect("equal")
ax0.plot(Y[:, 0], Y[:, 1], "r*", ms=6)
ax0.plot(y0[:, 0], y0[:, 1], "k+", ms=8)

if 11 < 2:
    trajectory = np.array([y0] + opt.history_Y)
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
y_vals = x_vals**2 - 2 * x_vals + 1
ax0.plot(x_vals, y_vals, "k--", alpha=0.5)
ax0.set_title("Objective space")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.legend([r"$Y_{\text{final}}$", r"$Y_0$", "Pareto front"])

ax22 = ax1.twinx()
ax1.plot(range(1, len(opt.history_indicator_value) + 1), opt.history_indicator_value, "b-")
ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
ax1.set_ylabel("HV", color="b")
ax22.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax1.set_title("Performance")
ax1.set_xlabel("iteration")
ax1.set_xticks(range(1, max_iters + 1))

plt.savefig(f"convex-example-{N}.pdf", dpi=1000)
