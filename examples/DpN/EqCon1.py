import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.reference_set import ReferenceSet

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
    return np.array([np.sum((x + 1) ** 2), np.sum((x - np.array([1, 0.5])) ** 2)])


def jacobian(x):
    x = np.array(x)
    return np.array([2 * (x + 1), 2 * (x - np.array([1, 0.5]))])


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


max_iters = 5
N = 30
ref = np.array(
    [
        [2.39509217, 0.92143131],
        [1.0407919, 3.48657002],
        [4.44697691, 0.27095062],
        [3.37029007, 0.46257782],
        [2.06251775, 1.19043157],
        [1.43849125, 2.04123165],
        [3.7938174, 0.35842383],
        [1.2070193, 2.63895819],
        [2.76540005, 0.70142941],
        [1.27299511, 2.43479765],
        [1.65142809, 1.67427599],
        [1.00109129, 4.14059259],
        [1.10567859, 3.05783547],
        [1.02044507, 3.7034503],
        [2.57615595, 0.80518987],
        [3.16370481, 0.53053286],
        [1.91327286, 1.34178139],
        [1.06899952, 3.27123067],
        [1.77615946, 1.50335677],
        [1.15146768, 2.84687837],
        [4.88929416, 0.25111594],
        [3.58047673, 0.40546354],
        [2.96162788, 0.61000058],
        [4.66765353, 0.25756658],
        [4.00967727, 0.3207637],
        [4.22753784, 0.29181774],
        [1.34997563, 2.23523211],
        [1.53891124, 1.85386348],
        [1.00738723, 3.9215453],
        [2.22342601, 1.05005941],
    ]
)

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

metrics = dict(GD=GenerationalDistance(ref=ref), IGD=InvertedGenerationalDistance(ref=ref))
opt = DpN(
    dim=2,
    n_obj=2,
    ref=ReferenceSet(ref=ref),
    func=func,
    jac=jacobian,
    hessian=hessian,
    h=h,
    h_jac=h_jacobian,
    h_hessian=h_hessian,
    N=N,
    x0=X0,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    metrics=metrics,
    type="igd",
    verbose=True,
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
reference_set = opt.ref.reference_set
lines += ax1.plot(Y[:, 0], Y[:, 1], "r+", ms=13)
lines += ax1.plot(Y0[:, 0], Y0[:, 1], "k.", ms=8)
lines += ax1.plot(reference_set[:, 0], reference_set[:, 1], "gs", ms=8, mfc="none")
lines += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "k--")
ax1.legend(lines, [r"$Y_{\text{final}}$", r"$Y_0$", "reference set", r"Pareto front"])
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
ax2.set_ylabel(r"$\Delta_p$", color="b")
ax22.set_ylabel(r"$||R_I(\mathbf{X},\lambda)||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))

plt.savefig(f"EqCon1-example-DpN-{N}.pdf", dpi=1000)

data = pd.DataFrame(
    np.c_[np.arange(len(opt.history_indicator_value)), opt.history_indicator_value, opt.history_R_norm],
    columns=["Iter", r"$\Delta_p$", r"$\|R_I(\mathbf{X}, \lambda)\|$"],
)
caption = r"""Convergence of the hypervolume value and the root-finding error $\|R_I(\mathbf{X}, \lambda)\|$.
"""
data.to_latex(
    f"EqCon1-example-DpN-{N}.tex",
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
