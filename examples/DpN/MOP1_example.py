import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "./")
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import DpN

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
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


def MOP1(x):
    x = np.array(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(2), 2 * np.eye(2)])


def h(x):
    x = np.array(x)
    return np.sum(x**2) - 1


def h_Jacobian(x):
    x = np.array(x)
    return 2 * x


theta = np.linspace(-np.pi * 3 / 4, np.pi / 4, 100)
ref_x = np.array([[np.cos(a) * 0.99, np.sin(a) * 0.99] for a in theta])
ref = np.array([MOP1(_) for _ in ref_x])

max_iters = 10
# Pareto set and front
p = np.linspace(-1, 1, 100)
pareto_set = np.c_[p, p]
pareto_front = np.array([MOP1(_) for _ in pareto_set])
# generate the reference set
best_from_angel = pd.read_csv("MOP1_n=30.csv", header=None).values
mu = len(best_from_angel)
# p = np.linspace(-1, 1, mu)
# ref_X = np.c_[p, p]
# ref = np.array([MOP1(_) for _ in ref_X])
ref = best_from_angel - 0.3
# the initial population
p = np.linspace(-1, 0.5, mu)
X0 = np.c_[p, p + 0.5]
Y0 = np.array([MOP1(_) for _ in X0])
dim = Y0.shape[1]

opt = DpN(
    dim=2,
    n_obj=2,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    N=len(X0),
    x0=X0,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    type="igd",
    pareto_front=pareto_front,
    verbose=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
# ciricle = plt.Circle((0, 0), 1, color="r", fill=False, ls="--", lw=1.5)

ax0.plot(X[:, 0], X[:, 1], "kx", ms=5)
ax0.plot(X0[:, 0], X0[:, 1], "g.", ms=8, clip_on=False)
# ax0.add_patch(ciricle)
ax0.set_xlim([-2, 2])
ax0.set_ylim([-2, 2])
ax0.set_title("Decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

# ax1.plot(ref[:, 0], ref[:, 1], "k.")
# ax0.plot(ref_x[:, 0], ref_x[:, 1], "k.")

n_per_axis = 30
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
Z = np.array([MOP1(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z1 = Z[:, 0].reshape(-1, len(x))
Z2 = Z[:, 1].reshape(-1, len(x))
CS1 = ax0.contour(X1, X2, Z1, 10, cmap=plt.cm.gray, linewidths=0.8, alpha=0.6)
CS2 = ax0.contour(X1, X2, Z2, 10, cmap=plt.cm.gray, linewidths=0.8, linestyles="--", alpha=0.6)

if 11 < 2:
    trajectory = np.array([X0] + opt.hist_X)
    for i in range(mu):
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
    trajectory = np.array([Y0] + opt.hist_Y)
    for i in range(mu):
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
ax1.plot(ref[:, 0], ref[:, 1], "r+")
ax1.plot(Y0[:, 0], Y0[:, 1], "g.", ms=8)
ax1.plot(Y[:, 0], Y[:, 1], "kx", ms=5)
ax1.legend(["reference set", r"$Y_0$", r"$Y_{\text{final}}$"])
# ax1.plot(x_vals, y_vals, "r--")
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax22 = ax2.twinx()
# ax2.semilogy(range(1, len(opt.hist_GD) + 1), opt.hist_GD, "b-", label="GD")
ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax22.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
ax2.legend()

plt.savefig(f"MOP1_DpN_{mu}points.pdf", dpi=1000)
np.savetxt("X_DpN.csv", X, delimiter=",")
np.savetxt("Y_DpN.csv", Y, delimiter=",")
