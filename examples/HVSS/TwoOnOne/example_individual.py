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


def objective(x: np.ndarray) -> np.ndarray:
    return x


def hessian(_) -> np.ndarray:
    return np.array([np.zeros((2, 2))] * 2)


def h(x: np.ndarray) -> float:
    return pareto_front_approx(x[0])[0] - x[1]


def h_jac(x: np.ndarray) -> np.ndarray:
    return np.array([pareto_front_approx(x[0])[1], -1])


def h_hessian(x: np.ndarray) -> np.ndarray:
    return np.array([[pareto_front_approx(x[0])[2], 0], [0, 0]])


max_iters = 25
X0 = pd.read_csv("examples/subset_selection/TwoOnOne/Y0.csv", index_col=None, header=None).values
Y0 = X0
N = len(X0)
X0_comp2 = X0[X0[:, 1] <= 5, :]
X0_comp1 = X0[X0[:, 1] > 5, :]

ref1 = np.array([8, 9.2])
ref2 = np.array([30, 3.1])


def jacobian(x: np.ndarray) -> np.ndarray:
    jac = np.eye(2)
    idx = np.nonzero(x == np.array([6.83341008507533, 7.35434168701381]))[0]
    jac[idx, idx] = 0
    idx = np.nonzero(x == np.array([7.66430625863304, 9.07976975741695]))[0]
    jac[idx, idx] = 0
    return jac


opt = HVN(
    n_var=2,
    n_obj=2,
    ref=ref1,
    func=objective,
    jac=jacobian,
    hessian=hessian,
    h=h,
    h_jac=h_jac,
    h_hessian=h_hessian,
    N=len(X0_comp1),
    X0=X0_comp1,
    xl=[6.83341008507533, 7.35434168701381],
    xu=[7.66430625863304, 9.07976975741695],
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y1, stop = opt.run()


def jacobian(x: np.ndarray) -> np.ndarray:
    jac = np.eye(2)
    idx = np.nonzero(x == np.array([7.66439406746696, 0]))[0]
    # jac[idx, idx] = 0
    idx = np.nonzero(x == np.array([21.95, 2.98489867805959]))[0]
    # jac[idx, idx] = 0
    return jac


opt = HVN(
    n_var=2,
    n_obj=2,
    ref=ref2,
    func=objective,
    jac=jacobian,
    hessian=hessian,
    h=h,
    h_jac=h_jac,
    h_hessian=h_hessian,
    N=len(X0_comp2),
    X0=X0_comp2,
    xl=[7.66439406746696, 0],
    xu=[21.95, 2.98489867805959],
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y2, stop = opt.run()
Y = np.r_[Y1, Y2]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
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

x = np.linspace(6.83341008507540, 7.66430625863300, 500)
y = np.array([pareto_front_approx(_)[0] for _ in x])
pareto_front1 = np.c_[x, y]
ax0.plot(pareto_front1[:, 0], pareto_front1[:, 1], "k--", alpha=0.5)

x = np.linspace(7.66439406746696, 21.9882877870465, 500)
y = np.array([pareto_front_approx(_)[0] for _ in x])
pareto_front2 = np.c_[x, y]
ax0.plot(pareto_front2[:, 0], pareto_front2[:, 1], "k--", alpha=0.5)
ax0.set_title("Objective space")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.legend([r"$Y_0$", "Approximated Pareto front"])

ax1.plot(Y[:, 0], Y[:, 1], "r+", ms=8)
ax1.plot(pareto_front1[:, 0], pareto_front1[:, 1], "k--", alpha=0.5)
ax1.plot(pareto_front2[:, 0], pareto_front2[:, 1], "k--", alpha=0.5)
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax1.legend([r"$Y_{\text{final}}$", "Approximated Pareto front"])

# ax22 = ax1.twinx()
# HV0 = opt.history_indicator_value[0]
# impr = (np.array(opt.history_indicator_value) - HV0) / HV0
# lns = ax1.plot(range(1, len(opt.history_indicator_value) + 1), opt.history_indicator_value, "b-")
# lns += ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
# ax1.legend(lns, ["HV", r"$||R(\mathbf{X})||$"], loc=1)
# ax1.set_ylabel("HV", color="b")
# ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
# ax1.set_title(f"Performance with ref: {ref}")
# ax1.set_xlabel("iteration")
# ax1.set_xticks(range(1, max_iters + 1))
plt.tight_layout()
plt.savefig(f"TwoOnOne-example-{len(X0)}.pdf", dpi=1000)
