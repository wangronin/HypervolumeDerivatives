import sys
from typing import Dict

sys.path.insert(0, "./")
import random
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import rcParams
from pymoo.core.problem import Problem as PymooProblem
from scipy.linalg import qr
from scipy.spatial.distance import directed_hausdorff
from sklearn.decomposition import PCA, TruncatedSVD

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.newton import DpN
from hvd.problems import CONV4_2F
from hvd.utils import get_non_dominated

plt.style.use("ggplot")
rcParams["font.size"] = 12
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
# rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

max_iters = 5
n_jobs = 30
gen = 400
emoa = "NSGA-III"
problem_name = "CONV4_2F"
problem = locals()[problem_name](boundry_constraints=False)
pareto_front = problem.get_pareto_front(5000)
reference_point = np.array([2, 2, 2, 10])
path = "./data-reference/CONV4_2F/"


def match_cluster(Y: np.ndarray, ref: Dict) -> np.ndarray:
    idx = []
    for y in Y:
        idx.append(np.argmin([directed_hausdorff(ref[i], np.atleast_2d(y))[0] for i in range(len(ref))]))
    return np.array(idx)


def plot_4d(y0, Y, ref, hist_Y, history_medoids, hist_IGD, hist_R_norm, fig_name):
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[15], colors[19]]
    medoids0 = np.array([h[0] for h in history_medoids])

    fig = plt.figure(figsize=plt.figaspect(1 / 2.0))
    plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)

    lines = []
    lines += ax0.plot(
        pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.2
    )
    shifts = []
    ax0.plot(
        medoids0[:, 0],
        medoids0[:, 1],
        medoids0[:, 2],
        color=colors[0],
        ls="none",
        marker="^",
        mec="none",
        ms=7,
        alpha=0.7,
    )
    # for i, M in enumerate(history_medoids):
    #     c = colors[len(M) - 1]
    #     for j, x in enumerate(M):
    #         line = ax0.plot(x[0], x[1], x[2], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
    #         if j == len(shifts):
    #             shifts.append(line)

    lines += shifts
    lines += ax0.plot(Y[:, 0], Y[:, 1], Y[:, 2], "k*", mec="none", ms=8, alpha=0.9)
    lines += ax0.plot(y0[:, 0], y0[:, 1], y0[:, 2], "r+", ms=8, alpha=0.9)
    counts = np.unique([len(m) for m in history_medoids], return_counts=True)[1]
    lgnd = ax0.legend(
        handles=lines,
        labels=["Pareto front"]
        # + [f"{i + 1} shift(s): {k} medoids" for i, k in enumerate(counts)]  # ["Pareto front"]
        + [r"$Y_{\mathrm{final}}$"] + [r"$Y_0$"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fancybox=True,
    )
    # for i in range(len(y0)):
    #     ax0.plot(
    #         (medoids0[i, 0], y0[i, 0]),
    #         (medoids0[i, 1], y0[i, 1]),
    #         (medoids0[i, 2], y0[i, 2]),
    #         "k-",
    #         alpha=0.5,
    #         lw=1,
    #     )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    ax0.set_title("Objective space")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.set_zlabel(r"$f_3$")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(45, 45)
    lines = []
    lines += ax1.plot(
        pareto_front[:, 1], pareto_front[:, 2], pareto_front[:, 3], "g.", mec="none", ms=5, alpha=0.2
    )
    shifts = []
    ax1.plot(
        medoids0[:, 1],
        medoids0[:, 2],
        medoids0[:, 3],
        color=colors[0],
        ls="none",
        marker="^",
        mec="none",
        ms=7,
        alpha=0.7,
    )
    # for i, M in enumerate(history_medoids):
    #     c = colors[len(M) - 1]
    #     for j, x in enumerate(M):
    #         line = ax1.plot(x[1], x[2], x[3], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
    #         if j == len(shifts):
    #             shifts.append(line)

    lines += shifts
    lines += ax1.plot(Y[:, 1], Y[:, 2], Y[:, 3], "k*", mec="none", ms=8, alpha=0.9)
    lines += ax1.plot(y0[:, 1], y0[:, 2], y0[:, 3], "r+", ms=8, alpha=0.9)
    counts = np.unique([len(m) for m in history_medoids], return_counts=True)[1]
    lgnd = ax1.legend(
        handles=lines,
        labels=["Pareto front"]
        # + [f"{i + 1} shift(s): {k} medoids" for i, k in enumerate(counts)]  # ["Pareto front"]
        + [r"$Y_{\mathrm{final}}$"] + [r"$Y_0$"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fancybox=True,
    )
    # for i in range(len(y0)):
    #     ax1.plot((medoids0[i, 1], y0[i, 1]), (medoids0[i, 2], y0[i, 2]), (medoids0[i, 3], y0[i, 3]), "k-")
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    ax1.set_title("Objective space")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.set_zlabel(r"$f_3$")

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6.5))
    plt.subplots_adjust(right=0.85, left=0.2)

    ax22 = ax2.twinx()
    ax2.semilogy(range(1, len(hist_IGD) + 1), hist_IGD, "r-", label="IGD")
    ax22.semilogy(range(1, len(hist_R_norm) + 1), hist_R_norm, "g--")
    ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    ax2.set_ylabel("IGD", color="r")
    # ax22.set_ylabel(r"R norm", color="g")
    ax2.set_title("Performance")
    ax2.set_xlabel("iteration")
    ax2.set_xticks(range(1, max_iters + 1))
    ax2.legend()
    # plt.tight_layout()
    plt.savefig(fig_name, dpi=1000)

    # plt.show()
    # plt.savefig(fig_name, dpi=1000)
    # plt.close(fig)


def execute(run: int):
    ref_label = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_component_id.csv", header=None).values[0]
    n_cluster = len(np.unique(ref_label))
    ref = dict()
    eta = dict()
    # load the reference set
    for i in range(n_cluster):
        if n_cluster == 1:
            ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_filling_comp.csv"
        else:
            ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_{i+1}_gen{gen}.csv"
            # ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_filling_comp{i+1}.csv"
        try:
            r = pd.read_csv(ref_file, header=None).values
        except:
            continue
        # downsample the reference; otherwise, initial clustering take too long
        ref[i] = np.array(random.sample(r.tolist(), 5000)) if len(r) >= 5000 else r
        try:
            eta[i] = pd.read_csv(
                f"{path}/{problem_name}_{emoa}_run_{run}_eta_{i+1}_gen{gen}.csv", header=None
            ).values.ravel()
        except:
            if i > 0 and eta[i - 1] is not None:  # copy the shift direction from the last cluster
                eta[i] = eta[i - 1]
            else:
                eta = None

    # sometimes the precomputed `eta` value can be `nan`
    if (eta is not None) and (np.any([np.any(np.isnan(_eta)) for _eta in eta.values()])):
        eta = None

    # from sklearn.preprocessing import MinMaxScaler

    # for i in range(len(eta)):
    #     svd = TruncatedSVD(n_components=4).fit(ref[i])
    #     idx = np.argmin(svd.explained_variance_)
    #     # pca = PCA(n_components=4).fit(ref[i])
    #     v = svd.components_[idx]
    #     if np.all(v > 0):
    #         v *= -1
    #     eta[i] = v

    # the load the final population from an EMOA
    x0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_x.csv", header=None).values
    y0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_y.csv", header=None).values
    # Y_label = match_cluster(y0, ref)
    Y_label = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_labels.csv", header=None
    ).values.ravel()
    Y_label = Y_label - 1
    # removing the outliers in `Y`
    idx = (Y_label != -2) & (Y_label != -1)
    x0 = x0[idx]
    y0 = y0[idx]
    Y_label = Y_label[idx]
    all_ref = np.vstack([r for r in ref.values()])

    # TODO: this is an ad-hoc solution. Maybe fix this special case in the `ReferenceSet` class
    # if the minimal number of points in the `ref` clusters is smaller than
    # the maximal number of points in `y0` clusters, then we merge all clusters
    min_point_ref_cluster = np.min([len(r) for r in ref.values()])
    max_point_y_cluster = np.max(np.unique(Y_label, return_counts=True)[1])
    # if the number of clusters of `Y` is more than that of the reference set
    if (1 < 2) and ((len(np.unique(Y_label)) > len(ref)) or (max_point_y_cluster > min_point_ref_cluster)):
        ref = np.vstack([r for r in ref.values()])
        Y_label = np.zeros(len(y0))
        eta = None
        # ensure the number of approximation points is less than the number of reference points
        if len(ref) < len(y0):
            n = len(ref)
            x0 = x0[:n]
            y0 = y0[:n]
            Y_label = Y_label[:n]

    opt = DpN(
        dim=problem.n_var,
        n_obj=problem.n_obj,
        ref=ref,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        # g=problem.ieq_constraint,
        # g_jac=problem.ieq_jacobian,
        N=len(x0),
        x0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=max_iters,
        type="igd",
        verbose=True,
        pareto_front=pareto_front,
        eta=eta,
        Y_label=Y_label,
    )
    X, Y, _ = opt.run()
    # remove the dominated solution in Y
    Y = get_non_dominated(Y)
    if 1 < 2:  # plotting the final approximation set
        fig_name = f"./plots/{problem_name}_DpN_{emoa}_run{run}_{gen}.pdf"
        plot_4d(y0, Y, all_ref, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    if 1 < 2:  # save the final approximation set
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y.csv", index=False)
        df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
        df_y0.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y0.csv", index=False)

    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    hv_value = hypervolume(Y, reference_point)
    return np.array([igd_value, gd_value, hv_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0]) for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x.csv")
]
if 1 < 2:
    data = []
    for i in [10]:
        print(i)
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV", "Jac_calls"])
# df.to_csv(f"results/{problem_name}-DpN-{emoa}-{gen}.csv", index=False)
