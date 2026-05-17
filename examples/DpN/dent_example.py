import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems.misc import DENT
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

ref = np.array(
    [
        [4.4529718, 0.56325987],
        [0.61205884, 2.86008543],
        [2.01924296, 1.03342935],
        [3.37332561, 0.58744177],
        [0.56815463, 4.16981022],
        [1.04858488, 2.01562282],
        [1.92897908, 1.70609874],
        [2.74103536, 0.62200239],
        [3.79626798, 0.57589864],
        [0.57349579, 3.90392566],
        [0.70227813, 2.33683266],
        [1.96962766, 1.47821309],
        [0.563619, 4.43112069],
        [2.53343767, 0.64992858],
        [0.83602991, 2.12687816],
        [3.58621716, 0.58114652],
        [0.5795679, 3.64677233],
        [1.78998323, 1.88800179],
        [0.58710593, 3.38463999],
        [1.98381354, 1.25218686],
        [0.59693691, 3.12429162],
        [0.64029394, 2.59460177],
        [4.0089585, 0.57127557],
        [3.15934022, 0.5953417],
        [4.22577177, 0.56712173],
        [2.12854806, 0.83283188],
        [1.29830957, 1.98070324],
        [1.55216929, 1.96208594],
        [2.95247052, 0.60574038],
        [2.31276336, 0.71072459],
    ]
)
max_iters = 7
problem = DENT()
N = 30
p = np.linspace(-1.5, 2.0, N)
X0 = np.c_[p, -p + 0.3]
Y0 = np.array([problem.objective(_) for _ in X0])
pareto_front = problem.get_pareto_front(1000)
N = len(X0)

metrics = dict(GD=GenerationalDistance(ref=ref), IGD=InvertedGenerationalDistance(ref=ref))
opt = DpN(
    dim=2,
    n_obj=2,
    ref=ReferenceSet(ref=ref),
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    N=N,
    x0=X0,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    metrics=metrics,
    type="igd",
    verbose=True,
    preconditioning=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(right=0.93, left=0.05)
x = np.linspace(-2, 2, 400)
y = -x
lines = []
lines += ax0.plot(X[:, 0], X[:, 1], "r+", ms=12)
lines += ax0.plot(X0[:, 0], X0[:, 1], "k.", ms=8, clip_on=False)
lines += ax0.plot(x, y, "k--", clip_on=False)
ax0.set_xlim([-2, 2])
ax0.set_ylim([-2, 2])
ax0.set_title("Decision space")
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

reference_set = opt.ref.reference_set
lines = []
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
ax2.set_ylabel("HV", color="b")
ax22.set_ylabel(r"$||R_I(\mathbf{X}, \lambda)||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))

plt.savefig(f"dent-example-DpN-{N}.pdf", dpi=1000)

data = pd.DataFrame(
    np.c_[np.arange(len(opt.history_indicator_value)), opt.history_indicator_value, opt.history_R_norm],
    columns=["Iter", r"$\operatorname{HV}$", r"$\|R_I(\mathbf{X}, \lambda)\|$"],
)
caption = r"""Convergence of the hypervolume value and the root-finding error $\|R_I(\mathbf{X}, \lambda)\|$.
"""
data.to_latex(
    f"dent-example-DpN-{N}.tex",
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
