import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "./")
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.reference_set import ReferenceSet

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


def h_Hessian(x):
    x = np.array(x)
    return 2 * np.eye(2)


theta = np.linspace(-np.pi * 3 / 4, np.pi / 4, 100)
ref_x = np.array([[np.cos(a) * 0.99, np.sin(a) * 0.99] for a in theta])
ref = np.array([MOP1(_) for _ in ref_x])

max_iters = 15
mu = 50

# different initializations of the first population
if 1 < 2:
    # option1: linearly spacing
    p = np.linspace(0, 2, mu)
    x0 = np.c_[p, p - 2]
elif 11 < 2:
    # option2: logistic spacing/denser on two tails
    p = 2 / (1 + np.exp(-np.linspace(-3, 3, mu)))
    x0 = np.c_[p, p - 2]
elif 11 < 2:
    # option3: logit spacing/denser in the middle
    p = np.log(1 / (1 - np.linspace(0.09485175, 1.90514825, mu) / 2) - 1)
    p = 2 * (p - np.min(p)) / (np.max(p) - np.min(p))
    x0 = np.c_[p, p - 2]
elif 11 < 2:
    # option4: spherical initialization
    theta = np.linspace(-np.pi * 5 / 8, np.pi / 8, 50)
    x0 = np.array([[2 * np.cos(a), 2 * np.sin(a)] for a in theta])

y0 = np.array([MOP1(_) for _ in x0])
metrics = dict(GD=GenerationalDistance(ref=ref), IGD=InvertedGenerationalDistance(ref=ref))
opt = DpN(
    dim=2,
    n_obj=2,
    ref=ReferenceSet(ref=ref),
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    h=h,
    h_jac=h_Jacobian,
    h_hessian=h_Hessian,
    N=len(x0),
    x0=x0,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    type="igd",
    verbose=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ciricle = plt.Circle((0, 0), 1, color="r", fill=False, ls="--", lw=1.5)

ax0.plot(X[:, 0], X[:, 1], "g*")
ax0.plot(x0[:, 0], x0[:, 1], "g.", ms=8, clip_on=False)
ax0.add_patch(ciricle)
ax0.set_xlim([-2, 2])
ax0.set_ylim([-2, 2])
ax0.set_title("Decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

ax1.plot(ref[:, 0], ref[:, 1], "k.")
# ax0.plot(ref_x[:, 0], ref_x[:, 1], "k.")

n_per_axis = 30
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
Z = np.array([MOP1(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z1 = Z[:, 0].reshape(-1, len(x))
Z2 = Z[:, 1].reshape(-1, len(x))
CS1 = ax0.contour(X1, X2, Z1, 10, cmap=plt.cm.gray, linewidths=0.8, alpha=0.6)
CS2 = ax0.contour(X1, X2, Z2, 10, cmap=plt.cm.gray, linewidths=0.8, linestyles="--", alpha=0.6)

if 1 < 2:
    trajectory = np.array([x0] + opt.history_X)
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
    trajectory = np.array([y0] + opt.history_Y)
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
ax1.plot(Y[:, 0], Y[:, 1], "g*")
ax1.plot(y0[:, 0], y0[:, 1], "g.", ms=8)
ax1.plot(ref[:, 0], ref[:, 1], "k.", ms=1)
ax1.plot(x_vals, y_vals, "r--")
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax22 = ax2.twinx()
# ax2.semilogy(range(1, len(opt.hist_GD) + 1), opt.hist_GD, "b-", label="GD")
# ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
ax2.legend()

plt.savefig(f"2D-example_convex-{mu}.pdf", dpi=1000)

# df = pd.DataFrame(dict(iteration=range(1, len(opt.hist_HV) + 1), HV=opt.hist_HV, G_norm=opt.hist_G_norm))
# df.to_latex(buf=f"2D-example-{mu}.tex", index=False)
