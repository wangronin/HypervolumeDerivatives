import sys

sys.path.insert(0, "./")

import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib import rcParams
from utils import PiecewiseBS, get_preimage

from hvd.hypervolume import hypervolume
from hvd.newton import HVN

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 13
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


def HV_subset_selection(problem_name: str):
    print(problem_name)
    t0 = time.process_time()
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
        verbose=False,
        preconditioning=False,
    )
    Y = opt.run()[1]
    best_so_far = np.maximum.accumulate(np.array(opt.history_indicator_value))
    HV0, HV1 = best_so_far[0], best_so_far[-1]
    # select points from the archive
    mat = scipy.io.loadmat(f"{path}/A_{problem_name}.mat")
    archive_Y = mat["Ay"]
    D = scipy.spatial.distance.cdist(Y, archive_Y, metric="euclidean")
    indices = np.argmin(D, axis=1)
    Y_sel = archive_Y[indices]
    CPU_time = time.process_time() - t0

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 12))
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

    t = np.linspace(0, 1, 1000)
    pareto_front = np.array([bs.objective(_) for _ in t])

    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.7)
    ax0.set_title(f"Initial HV: {HV0}")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.legend([r"$Y_0$", "Approximated Pareto front"])

    ax1.plot(Y_sel[:, 0], Y_sel[:, 1], "r+", ms=8)
    ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.7)
    ax1.set_title(f"Final HV: {HV1}")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.legend([r"$Y_{\text{final}}$", "Approximated Pareto front"])

    ax2.plot(Y[:, 0], Y[:, 1], "r*", mfc="none", ms=8)
    ax2.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.7)
    ax2.set_title(f"HV of the subset: {hypervolume(Y_sel, ref)}")
    ax2.set_xlabel(r"$f_1$")
    ax2.set_ylabel(r"$f_2$")
    ax2.legend([r"$Y_{\text{selected}}$", "Approximated Pareto front"])

    lns = ax3.plot(range(1, len(opt.history_indicator_value) + 1), best_so_far, "b-")
    ax3.legend(lns, ["HV", r"$||R(\mathbf{X})||$"], loc=1)
    ax3.set_ylabel("HV", color="b")
    ax3.set_title(f"reference point: {ref}")
    ax3.set_xlabel("iteration")
    ax3.set_xticks(range(1, max_iters + 1))
    plt.tight_layout()
    plt.savefig(f"{path}/plots/HV_subset_selection_{problem_name}-{N}.pdf", dpi=1000)
    plt.close(fig)
    return indices, ref, HV0, HV1, CPU_time


problem_names = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DENT", "CONV2", "2on1"]
# problem_names = ["ZDT1"]
selection_indices = []
reference_points = []
HV_values = []
CPU_times = []
for problem_name in problem_names:
    indices, ref, HV0, HV1, CPU_time = HV_subset_selection(problem_name)
    selection_indices.append(indices)
    reference_points.append(ref)
    HV_values.append([HV0, HV1])
    CPU_times.append(CPU_time)

pd.DataFrame(selection_indices, index=problem_names).to_csv(f"{path}/results/HV_subset_selection_indices.csv")
pd.DataFrame(reference_points, index=problem_names, columns=["y1", "y2"]).to_csv(
    f"{path}/results/HV_subset_selection_ref.csv"
)
pd.DataFrame(HV_values, index=problem_names, columns=["HV initial", "HV final"]).to_csv(
    f"{path}/results/HV_subset_selection_HV.csv"
)
pd.DataFrame(np.array(CPU_times).reshape(-1, 1), index=problem_names, columns=["Seconds"]).to_csv(
    f"{path}/results/HV_subset_selection_CPU_time.csv"
)
