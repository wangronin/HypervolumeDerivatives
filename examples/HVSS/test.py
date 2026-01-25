import sys
from typing import Tuple

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
rcParams["font.size"] = 15
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


def HV_subset_selection(
    problem_name: str,
    run_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray, np.ndarray]:
    print(problem_name)
    t0 = time.time()
    components = glob(f"{path}/fit_Sy_{problem_name}_comp*.csv")
    data = pd.read_csv(f"{path}/Y0_{problem_name}_run{run_id}.csv", header=None, index_col=False).values
    bs = PiecewiseBS(components)

    ref = 1.5 * np.max(data, axis=0)  # set the reference point based on the data
    X0 = np.array([get_preimage(p, bs) for p in data])
    # the number of points of each connected component of the Pareto front initially
    labels0, counts0 = np.unique([bs.get_index(t) for t in X0], return_counts=True)
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
    # the number of points of each connected component of the Pareto front after optimization
    labels1, counts1 = np.unique([bs.get_index(t) for t in opt.state.X], return_counts=True)

    best_so_far_HV = np.maximum.accumulate(np.array(opt.history_indicator_value))
    best_so_far_R_norm = np.minimum.accumulate(np.array(opt.history_R_norm))
    HV0, HV1 = best_so_far_HV[0], best_so_far_HV[-1]
    # select points from the archive
    mat = scipy.io.loadmat(f"{path}/A_{problem_name}_run{run_id}.mat")
    archive_Y = mat["Ay"]
    # archive_Y = pd.read_csv(f"{path}/Ay_{problem_name}.csv", index_col=None, header=None).values
    D = scipy.spatial.distance.cdist(Y, archive_Y, metric="euclidean")
    indices = np.argmin(D, axis=1)
    Y_sel = archive_Y[indices]
    wall_clock_time = time.time() - t0
    HV2 = hypervolume(Y_sel, ref)

    t = np.linspace(0, 1, 1000)
    pareto_front = np.array([bs.objective(_) for _ in t])

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(right=0.93, left=0.05)
    ax0.plot(Y0[:, 0], Y0[:, 1], "k+", ms=8)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.7)
    ax0.set_title(f"Initial HV: {HV0}")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.text(0.38, 0.8, f"{counts0[0]} points in component 1", transform=ax0.transAxes, ha="center")
    ax0.text(0.7, 0.2, f"{counts0[1]} points in component 2", transform=ax0.transAxes, ha="center")
    ax0.legend([r"$Y_0$", "Approximated Pareto front"])

    ax1.plot(Y_sel[:, 0], Y_sel[:, 1], "r+", ms=8)
    ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.7)
    ax1.set_title(f"Final HV: {HV1}")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.text(0.38, 0.8, f"{counts1[0]} points in component 1", transform=ax1.transAxes, ha="center")
    ax1.text(0.7, 0.2, f"{counts1[1]} points in component 2", transform=ax1.transAxes, ha="center")
    ax1.legend([r"$Y_{\text{final}}$", "Approximated Pareto front"])

    ax2.plot(Y[:, 0], Y[:, 1], "r*", mfc="none", ms=8)
    ax2.plot(pareto_front[:, 0], pareto_front[:, 1], "k--", alpha=0.7)
    ax2.set_title(f"HV of the subset: {hypervolume(Y_sel, ref)}")
    ax2.set_xlabel(r"$f_1$")
    ax2.set_ylabel(r"$f_2$")
    ax2.legend([r"$Y_{\text{selected}}$", "Approximated Pareto front"])

    lns = []
    ax1 = ax3.twinx()
    lns += ax1.semilogy(range(1, len(best_so_far_R_norm) + 1), best_so_far_R_norm, "g--")
    ax1.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    lns += ax3.plot(range(1, len(best_so_far_HV) + 1), best_so_far_HV, "b-")
    ax3.legend(lns, [r"$||R(\mathbf{X})||$", "HV"], loc="lower center")
    ax3.set_ylabel("HV", color="b")
    ax3.set_title(f"reference point: {ref}")
    ax3.set_xlabel("iteration")
    ax3.set_xticks(range(1, max_iters + 1))
    plt.tight_layout()
    plt.savefig(f"{path}/plots/HV_subset_selection_{problem_name}_run{run_id}.pdf", dpi=1000)
    plt.close(fig)
    # save the performance chart of HVN as a standalone figure
    # fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    # plt.subplots_adjust(right=0.93, left=0.05)
    # lns = []
    # ax1 = ax.twinx()
    # lns += ax1.semilogy(range(1, len(best_so_far_R_norm) + 1), best_so_far_R_norm, "g--", marker=".")
    # ax1.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    # lns += ax.plot(range(1, len(best_so_far_HV) + 1), best_so_far_HV, "b-", marker=".")
    # ax.legend(lns, [r"$||R(\mathbf{X})||$", "HV"], loc="lower center")
    # ax.set_ylabel("HV", color="b")
    # ax.set_title(f"reference point: {ref}")
    # ax.set_xlabel("iteration")
    # ax.set_xticks(range(1, max_iters + 1))
    # plt.tight_layout()
    # plt.savefig(f"{path}/plots/HV_subset_selection_{problem_name}_performance_run{run_id}.pdf", dpi=1000)
    # plt.close(fig)
    return Y, indices, ref, HV0, HV1, HV2, wall_clock_time, best_so_far_HV, best_so_far_R_norm


# problem_names = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DENT", "CONV2", "2on1"]
problem_names = ["2on1"]
for run in range(10, 11):
    print(run)
    selection_indices = []
    reference_points = []
    HV_values = []
    CPU_times = []
    best_so_far_HV = []
    best_so_far_R_norm = []
    for problem_name in problem_names:
        Y, indices, ref, HV0, HV1, HV2, CPU_time, HV_hist, R_norm_hist = HV_subset_selection(
            problem_name, run
        )
        pd.DataFrame(Y, columns=["f1", "f2"]).to_csv(
            f"{path}/results/{problem_name}_HV_final_points_run{run}.csv"
        )
        selection_indices.append(indices)
        reference_points.append(ref)
        HV_values.append([HV0, HV1, HV2])
        CPU_times.append(CPU_time)
        best_so_far_HV.append(HV_hist)
        best_so_far_R_norm.append(R_norm_hist)

    # pd.DataFrame(selection_indices, index=problem_names).to_csv(
    #     f"{path}/results/HV_subset_selection_indices_run{run}.csv"
    # )
    # pd.DataFrame(reference_points, index=problem_names, columns=["y1", "y2"]).to_csv(
    #     f"{path}/results/HV_subset_selection_ref_run{run}.csv"
    # )
    # pd.DataFrame(HV_values, index=problem_names, columns=["HV initial", "HV final", "HV selected"]).to_csv(
    #     f"{path}/results/HV_subset_selection_HV_run{run}.csv"
    # )
    # pd.DataFrame(np.array(CPU_times).reshape(-1, 1), index=problem_names, columns=["Seconds"]).to_csv(
    #     f"{path}/results/HV_subset_selection_CPU_time_run{run}.csv"
    # )
    # pd.DataFrame(best_so_far_HV, index=problem_names).to_csv(
    #     f"{path}/results/HV_subset_selection_HV_history_run{run}.csv"
    # )
    # pd.DataFrame(best_so_far_R_norm, index=problem_names).to_csv(
    #     f"{path}/results/HV_subset_selection_R_norm_history_run{run}.csv"
    # )
