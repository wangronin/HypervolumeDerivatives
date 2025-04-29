import sys

sys.path.insert(0, "./")

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from utils import PiecewiseBS, get_preimage

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

max_iters = 10
path = "./spline"
problem_name = sys.argv[1]
print(problem_name)
components = glob(f"{path}/fit_Sy_{problem_name}_comp*.csv")
data = pd.read_csv(f"{path}/Y0_{problem_name}.csv", header=None, index_col=False).values
bs = PiecewiseBS(components)

ref = 1.5 * np.max(data, axis=0)  # set the reference point based on the data
X0 = np.array([get_preimage(p, bs) for p in data])
Y0 = np.array([bs.objective(x) for x in X0])
N = len(X0)

opt = HVN(
    n_var=1,
    n_obj=2,
    ref=ref,
    func=bs.objective,
    jac=bs.jacobian,
    hessian=bs.hessian,
    N=N,
    X0=X0,
    xl=0,
    xu=1,
    max_iters=max_iters,
    verbose=True,
    preconditioning=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax0.set_aspect("auto")
ax1.set_aspect("auto")
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

t = np.linspace(0, 1, 1000)
pareto_front = np.array([bs.objective(_) for _ in t])

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

HV0 = opt.history_indicator_value[0]
impr = np.maximum.accumulate(np.array(opt.history_indicator_value))
lns = ax2.plot(range(1, len(opt.history_indicator_value) + 1), impr, "b-")
ax2.legend(lns, ["HV", r"$||R(\mathbf{X})||$"], loc=1)
ax2.set_ylabel("HV", color="b")
ax2.set_title(f"Performance with ref: {ref}")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
plt.tight_layout()
plt.savefig(f"HVN_subset_selection_{problem_name}-{N}.pdf", dpi=1000)

# data = pd.DataFrame(np.c_[Y0, Y], columns=["initial y1", "initial y2", "final y1", "final y2"])
# data.to_csv(f"ZDT1-example-{N}.csv")
