import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from B_spline import hessian1, hessian2, jacobian1, jacobian2, objective1, objective2
from matplotlib import rcParams
from scipy.optimize import minimize_scalar

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


# we should be able to get it for free from the fitting of B-spline
def inverse_search(y: np.ndarray, func: callable) -> np.ndarray:
    def objective(t: float) -> float:
        return np.linalg.norm(func(t) - y)

    result = minimize_scalar(
        objective, bounds=(0, 1), method="bounded", options=dict(xatol=1e-8, maxiter=1000)
    )
    return result.x


max_iters = 25
X0 = pd.read_csv("examples/subset_selection/TwoOnOne/Y0.csv", index_col=None, header=None).values
Y0 = X0
N = len(X0)
X0_comp1 = X0[X0[:, 1] <= 5, :]
X0_comp2 = X0[X0[:, 1] > 5, :]

T0_comp1 = np.array([inverse_search(y, objective1) for y in X0_comp1])
T0_comp2 = np.array([inverse_search(y, objective2) for y in X0_comp2])
ref1 = np.array([8, 9.2])
ref2 = np.array([30, 3.1])

opt = HVN(
    n_var=1,
    n_obj=2,
    ref=ref1,
    func=objective1,
    jac=jacobian1,
    hessian=hessian1,
    N=len(T0_comp1),
    X0=T0_comp1,
    xl=0,
    xu=1,
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y1, stop = opt.run()

opt = HVN(
    n_var=1,
    n_obj=2,
    ref=ref2,
    func=objective2,
    jac=jacobian2,
    hessian=hessian2,
    N=len(T0_comp2),
    X0=T0_comp2,
    xl=0,
    xu=1,
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

pareto_front1 = np.array([objective1(t) for t in np.linspace(0, 1, 500)])
ax0.plot(pareto_front1[:, 0], pareto_front1[:, 1], "k--", alpha=0.5)

pareto_front2 = np.array([objective2(t) for t in np.linspace(0, 1, 500)])
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
