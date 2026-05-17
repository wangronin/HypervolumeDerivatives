import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import HVN

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


def func(x):
    x = np.array(x)
    return np.array([np.sum((x + 1) ** 2), np.sum((x - 1) ** 2)])


def jacobian(x):
    x = np.array(x)
    return np.array([2 * (x + 1), 2 * (x - 1)])


def hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(2), 2 * np.eye(2)])


def h(x):
    x = np.array(x)
    return x[1]


def h_jacobian(_):
    return np.array([0, 1])


def h_hessian(_):
    return np.zeros((2, 2))


ref = np.array([20, 20])
max_iters = 6
N = 30

# different initializations of the first population
if 1 < 2:
    # option1: linearly spacing
    p = np.linspace(-1, 1, N)
elif 11 < 2:
    # option2: logistic spacing/denser on two tails
    p = 2 / (1 + np.exp(-np.linspace(-3, 3, N)))
elif 11 < 2:
    # option3: logit spacing/denser in the middle
    p = np.log(1 / (1 - np.linspace(0.09485175, 1.90514825, N) / 2) - 1)
    p = 2 * (p - np.min(p)) / (np.max(p) - np.min(p))

X0 = np.c_[p, [0.2] * len(p)]
Y0 = np.array([func(_) for _ in X0])
z = np.linspace(-1, 1, 500)
pareto_set = np.c_[z, [0] * len(z)]
pareto_front = np.array([func(x) for x in pareto_set])

opt = HVN(
    n_var=2,
    n_obj=2,
    ref=ref,
    func=func,
    jac=jacobian,
    hessian=hessian,
    h=h,
    h_jac=h_jacobian,
    h_hessian=h_hessian,
    N=N,
    X0=X0,
    xl=-1.5,
    xu=1.5,
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y, stop = opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(right=0.93, left=0.05)
lines = []
lines += ax0.plot(X[:, 0], X[:, 1], "r+", ms=12)
lines += ax0.plot(X0[:, 0], X0[:, 1], "k.", ms=8, clip_on=False)
lines += ax0.plot(pareto_set[:, 0], pareto_set[:, 1], "k--", clip_on=False)
ax0.set_xlim([-1.2, 1.2])
ax0.set_ylim([-1.2, 1.2])
ax0.set_title("Decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")
ax0.legend(lines, [r"$X_{\text{final}}$", r"$X_0$", r"Pareto set"])

n_per_axis = 30
x = np.linspace(-1.2, 1.2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
Z = np.array([func(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z1 = Z[:, 0].reshape(-1, len(x))
Z2 = Z[:, 1].reshape(-1, len(x))
CS1 = ax0.contour(X1, X2, Z1, 10, cmap=plt.cm.Blues, linewidths=0.8, alpha=0.8)
CS2 = ax0.contour(X1, X2, Z2, 10, cmap=plt.cm.Reds, linewidths=0.8, alpha=0.8)

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

ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax22 = ax2.twinx()
ax2.plot(range(1, len(opt.history_indicator_value) + 1), opt.history_indicator_value, "b-")
ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
ax2.set_ylabel("HV", color="b")
ax22.set_ylabel(r"$||R_I(\mathbf{X},\lambda)||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))

plt.savefig(f"EqCon1-example-HVN-{N}.pdf", dpi=1000)

data = pd.DataFrame(
    np.c_[np.arange(len(opt.history_indicator_value)), opt.history_indicator_value, opt.history_R_norm],
    columns=["Iter", r"$\operatorname{HV}$", r"$\|R_I(\mathbf{X}, \lambda)\|$"],
)
caption = r"""Convergence of the hypervolume value and the root-finding error $\|R_I(\mathbf{X}, \lambda)\|$.
"""
data.to_latex(
    f"EqCon1-example-HVN-{N}.tex",
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

# df = pd.DataFrame(dict(iteration=range(1, len(opt.hist_HV) + 1), HV=opt.hist_HV, G_norm=opt.hist_G_norm))
# df.to_latex(buf=f"2D-example-{mu}.tex", index=False)
