import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import HVN
from hvd.problems.misc import DISCONNECTED

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

ref = np.array([1, 1])
max_iters = 10
X0 = pd.read_csv("disconnected_y.csv", header=None, index_col=None).values
Y0 = pd.read_csv("disconnected_y.csv", header=None, index_col=None).values
problem = DISCONNECTED()
# pareto_front = problem.get_pareto_front(1000)
N = len(X0)

opt = HVN(
    n_var=2,
    n_obj=2,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    h=problem.eq_constraint,
    h_jac=problem.eq_jacobian,
    h_hessian=problem.eq_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    g_hessian=problem.ieq_hessian,
    N=len(X0),
    X0=X0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y, stop = opt.run()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax0.set_aspect("auto")
ax0.plot(Y[:, 0], Y[:, 1], "r+", ms=8)
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

# ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.5)
ax0.set_title("Objective space")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")
ax0.legend([r"$Y_{\text{final}}$", r"$Y_0$", "Pareto front"])

ax22 = ax1.twinx()
ax1.plot(range(1, len(opt.history_indicator_value) + 1), opt.history_indicator_value, "b-")
ax22.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
ax1.set_ylabel("HV", color="b")
ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax1.set_title("Performance")
ax1.set_xlabel("iteration")
ax1.set_xticks(range(1, max_iters + 1))

plt.savefig(f"disconnected-example-{N}.pdf", dpi=1000)

data = pd.DataFrame(np.c_[Y0, Y], columns=["initial y1", "initial y2", "final y1", "final y2"])
data.to_csv(f"disconnected-example-{N}.csv")
