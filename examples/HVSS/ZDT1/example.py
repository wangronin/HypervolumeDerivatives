import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from spline import pareto_front_approx

from hvd.newton import HVN

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 11
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

ref = np.array([2, 2])
max_iters = 15
Y0 = X0 = pd.read_csv("examples/subset_selection/ZDT1/points.csv", header=None, index_col=None).values
N = len(X0)


def objective(x: np.ndarray) -> np.ndarray:
    return x


def jacobian(x: np.ndarray) -> np.ndarray:
    # whenever a point is on the boundary, we set the corresponding Jacobian to zero
    # to make the point stationary
    jac = np.eye(2)
    idx = np.nonzero(x == 0)[0]
    jac[idx, idx] = 0
    idx = np.nonzero(x == 1)[0]
    jac[idx, idx] = 0
    return jac


def hessian(_) -> np.ndarray:
    return np.array([np.zeros((2, 2))] * 2)


def h(x: np.ndarray) -> float:
    return pareto_front_approx(x[0])[0] - x[1]


def h_Jacobian(x: np.ndarray) -> np.ndarray:
    return np.array([pareto_front_approx(x[0])[1], -1])


def h_Hessian(x: np.ndarray) -> np.ndarray:
    return np.array([[pareto_front_approx(x[0])[2], 0], [0, 0]])


opt = HVN(
    n_var=2,
    n_obj=2,
    ref=ref,
    func=objective,
    jac=jacobian,
    hessian=hessian,
    h=h,
    h_jac=h_Jacobian,
    h_hessian=h_Hessian,
    N=len(X0),
    X0=X0,
    xl=[0, 0],
    xu=[1, 1],
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y, stop = opt.run()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax0.set_aspect("equal")
ax0.plot(Y0[:, 0], Y0[:, 1], "k+", ms=8)

if 11 < 2:
    trajectory = np.array([Y0] + opt.history_Y)
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

x = np.linspace(0, 1, 1000)
y = np.array([pareto_front_approx(_)[0] for _ in x])
pareto_front = np.c_[x, y]

ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.5)
ax0.set_title("Objective space")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.legend([r"$Y_0$", "Approximated Pareto front"])

ax1.plot(Y[:, 0], Y[:, 1], "r+", ms=8)
ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.5)
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax1.legend([r"$Y_{\text{final}}$", "Approximated Pareto front"])

# ax22 = ax2.twinx()
# HV0 = opt.history_indicator_value[0]
# impr = (np.array(opt.history_indicator_value) - HV0) / HV0
# lns = ax2.plot(range(1, len(opt.history_indicator_value) + 1), opt.history_indicator_value, "b-")
# lns += ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
# ax2.legend(lns, ["HV", r"$||R(\mathbf{X})||$"], loc=1)
# ax2.set_ylabel("HV", color="b")
# ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
# ax2.set_title(f"Performance with ref: {ref}")
# ax2.set_xlabel("iteration")
# ax2.set_xticks(range(1, max_iters + 1))
plt.tight_layout()
plt.savefig(f"ZDT1-example-{N}.pdf", dpi=1000)

# data = pd.DataFrame(np.c_[Y0, Y], columns=["initial y1", "initial y2", "final y1", "final y2"])
# data.to_csv(f"ZDT1-example-{N}.csv")
