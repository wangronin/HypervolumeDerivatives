import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

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


def read_reference_set_data(
    path: str, problem_name: str, emoa: str, run: int, gen: int
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """_summary_

    Args:
        path (str): path to the data folder
        problem_name (str): problem name
        emoa (str): EMOA algorithm name
        run (int): run ID
        gen (int): the stopping generation of EMOA

    Returns:
        Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]: _description_
    """
    ref_label = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_component_id_gen{gen}.csv", header=None
    ).values[0]
    n_cluster = len(np.unique(ref_label))
    ref = dict()
    eta = dict()
    # load the reference set
    for i in range(n_cluster):
        if problem_name in ["DTLZ6", "DTLZ7"]:
            # for DTLZ7 we need to load the dense fillings of the reference set
            r = pd.read_csv(
                f"{path}/{problem_name}_{emoa}_run_{run}_filling_comp{i+1}_gen{gen}.csv", header=None
            ).values
        else:
            if n_cluster == 1:
                ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_gen{gen}.csv"
            else:
                ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_{i+1}_gen{gen}.csv"
            try:
                r = pd.read_csv(ref_file, header=None).values
            except:
                continue
        # downsample the reference; otherwise, initial clustering take too long
        ref[i] = np.array(random.sample(r.tolist(), 3000)) if len(r) >= 3000 else r
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

    # the load the final population from an EMOA
    x0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_x_gen{gen}.csv", header=None).values
    y0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_y_gen{gen}.csv", header=None).values
    Y_label = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_labels_gen{gen}.csv", header=None
    ).values.ravel()
    Y_label = Y_label - 1  # index starts at 0
    # removing the outliers in `Y`
    idx = (Y_label != -2) & (Y_label != -1)
    x0 = x0[idx]
    y0 = y0[idx]
    Y_label = Y_label[idx]

    # TODO: this is an ad-hoc solution. Maybe fix this special case in the `ReferenceSet` class
    # if the minimal number of points in the `ref` clusters is smaller than
    # the maximal number of points in `y0` clusters, then we merge all clusters
    min_point_ref_cluster = np.min([len(r) for r in ref.values()])
    max_point_y_cluster = np.max(np.unique(Y_label, return_counts=True)[1])
    # if the number of clusters of `Y` is more than that of the reference set
    # NOTE: for simple MMD, we always merge the clusters of the reference set
    if (len(np.unique(Y_label)) > len(ref)) or (max_point_y_cluster > min_point_ref_cluster):
        # if len(ref) > 1 or (len(np.unique(Y_label)) > len(ref)) or (max_point_y_cluster > min_point_ref_cluster):
        ref = {0: np.vstack([r for r in ref.values()])}
        Y_label = np.zeros(len(y0), dtype=int)
        eta = None
        # ensure the number of approximation points is less than the number of reference points
        if len(ref[0]) < len(y0):
            n = len(ref)
            x0 = x0[:n]
            y0 = y0[:n]
            Y_label = Y_label[:n]
    Y_index = [np.nonzero(Y_label == i)[0] for i in np.unique(Y_label)]
    return ref, x0, y0, Y_index, eta


def plot_2d(
    Y0: np.ndarray,
    Y: np.ndarray,
    ref: np.ndarray,
    pareto_front: np.ndarray,
    hist_Y: List[np.ndarray],
    history_medoids: List[np.ndarray],
    history_metric: np.ndarray,
    hist_R_norm: np.ndarray,
    fig_name: str,
    plot_trajectory: bool = True,
) -> None:
    """plot the convergence results in 2D objective space for reference set-based methods, e.g., DpN.

    Args:
        Y0 (np.ndarray): initial approximation set
        Y (np.ndarray): the final approximation set
        ref (np.ndarray): the initial reference set
        pareto_front (np.ndarray): target Pareto front
        hist_Y (List[np.ndarray]): history/trajectory of the approximation set
        history_medoids (List[np.ndarray]): history/trajectory of medoids of the reference set
        history_metric (np.ndarray): history/trajectory of performance metrics
        hist_R_norm (np.ndarray): history/trajectory of the norm of the root-finding problem
        fig_name (str): name to save the figure
        plot_trajectory (bool, optional): if plotting the trajectory of approximation points. Defaults to True.
    """
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[17], colors[19]]
    n_colors = len(colors)
    quiver_kwargs = dict(
        scale_units="xy",
        angles="xy",
        scale=1,
        color="k",
        width=0.003,
        alpha=0.5,
        headlength=4.5,
        headwidth=2.5,
    )
    plt.style.use("ggplot")
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.5))
    plt.subplots_adjust(right=0.93, left=0.05)

    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=6, alpha=0.4)
    ax0.plot(Y0[:, 0], Y0[:, 1], "k+", ms=9, alpha=1)
    ax0.plot(ref[:, 0], ref[:, 1], "b.", mec="none", ms=5, alpha=0.3)
    ax0.set_title("Objective space (Initialization)")
    ax0.set_xlabel(r"f_1")
    ax0.set_ylabel(r"f_2")
    lgnd = ax0.legend(["Pareto front", "Y0", "reference set", "matched points"])
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    N = len(Y0)
    if plot_trajectory:
        trajectory = np.array([Y0] + hist_Y)
        for i in range(N):
            x, y = trajectory[:, i, 0], trajectory[:, i, 1]
            ax1.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], **quiver_kwargs)

    lines = []
    lines += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.3)
    shifts = []
    for i, M in history_medoids.items():
        c = colors[len(M) - 1]
        for j, x in enumerate(M):
            line = ax1.plot(x[0], x[1], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
            if j == len(shifts):
                shifts.append(line)

    lines += shifts
    lines += ax1.plot(Y[:, 0], Y[:, 1], "r+", ms=9, alpha=0.9)
    counts = np.unique([len(m) for _, m in history_medoids.items()], return_counts=True)[1]
    lgnd = ax1.legend(
        lines,
        ["Pareto front"] + [f"{i + 1} shift(s): {k} points" for i, k in enumerate(counts)] + [r"Y final"],
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(12)

    ax1.set_title("Objective space")
    ax1.set_xlabel(r"f1")
    ax1.set_ylabel(r"f2")

    # ax22 = ax2.twinx()
    for i, (k, v) in enumerate(history_metric.items()):
        ax2.semilogy(range(1, len(v) + 1), v, color=colors[i % n_colors], ls="solid", label=k)
    # ax22.semilogy(range(1, len(hist_R_norm) + 1), hist_R_norm, "g--")
    # ax22.set_ylabel(r"R norm", color="g")
    # ax22.set_ylabel(r"R norm", color="g")
    ax2.set_title("Performance")
    ax2.set_xlabel("iteration")
    ax2.set_xticks(range(1, len(hist_R_norm) + 1))
    ax2.legend()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=1000)
    plt.close(fig)


def plot_3d(
    Y0: np.ndarray,
    Y: np.ndarray,
    ref: np.ndarray,
    pareto_front: np.ndarray,
    hist_Y: List[np.ndarray],
    history_medoids: List[np.ndarray],
    history_metric: np.ndarray,
    hist_R_norm: np.ndarray,
    fig_name: str,
    plot_trajectory: bool = False,
) -> None:
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[15], colors[19]]
    medoids0 = np.array([h[0] for h in history_medoids.values()])

    fig = plt.figure(figsize=plt.figaspect(1 / 2.0))
    plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    # fig = plt.figure(figsize=plt.figaspect(1 / 1.0))
    # plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    # ax0 = fig.add_subplot(1, 1, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)
    ax0.plot(Y0[:, 0], Y0[:, 1], Y0[:, 2], "k+", ms=8, alpha=0.6)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.4)
    ax0.plot(ref[:, 0], ref[:, 1], ref[:, 2], "b.", mec="none", ms=5, alpha=0.2)
    ax0.plot(
        medoids0[:, 0],
        medoids0[:, 1],
        medoids0[:, 2],
        color=colors[0],
        ls="none",
        marker="^",
        mec="none",
        ms=5,
        alpha=0.8,
    )

    ax0.set_title("Initialization")
    ax0.set_xlabel(r"f1")
    ax0.set_ylabel(r"f2")
    ax0.set_zlabel(r"f3")
    lgnd = ax0.legend(
        # [r"$Y_0$", "Pareto front", "medoids"],
        [r"Y0", "reference set"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
        fancybox=True,
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)
    # plt.savefig(fig_name + "_1.pdf", dpi=1000)

    # for i in range(len(y0)):
    # ax0.plot((medoids0[i, 0], y0[i, 0]), (medoids0[i, 1], y0[i, 1]), (medoids0[i, 2], y0[i, 2]), "k-")

    # fig = plt.figure(figsize=plt.figaspect(1 / 1.0))
    # # plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
    # plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    # ax1 = fig.add_subplot(1, 1, 1, projection="3d")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(45, 45)
    if plot_trajectory:
        trajectory = np.array([Y0] + hist_Y)
        for i in range(len(Y0)):
            x, y, z = trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2]
            ax1.quiver(
                x[:-1],
                y[:-1],
                z[:-1],
                x[1:] - x[:-1],
                y[1:] - y[:-1],
                z[1:] - z[:-1],
                color="k",
                alpha=0.5,
                arrow_length_ratio=0.05,
            )

    lines = []
    # lines += ax1.plot(
    #     pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.2
    # )
    shifts = []
    # for i, M in enumerate(history_medoids):
    #     c = colors[len(M) - 1]
    #     for j, x in enumerate(M):
    #         line = ax1.plot(x[0], x[1], x[2], color=c, ls="none", marker="^", mec="none", ms=7, alpha=0.7)[0]
    #         if j == len(shifts):
    #             shifts.append(line)

    # lines += shifts
    lines += ax1.plot(Y0[:, 0], Y0[:, 1], Y0[:, 2], "k+", mfc="none", ms=6, alpha=0.9)
    lines += ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r*", mfc="none", ms=8, alpha=0.9)
    lines += ax1.plot(
        pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.4
    )
    # counts = np.unique([len(m) for m in history_medoids], return_counts=True)[1]
    lgnd = ax1.legend(
        handles=lines,
        # labels=[f"{i + 1} shift(s): {k} medoids" for i, k in enumerate(counts)]  # ["Pareto front"]
        labels=["Y0", "Y-final", "Pareto front"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
        fancybox=True,
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    ax1.set_title("Final population")
    ax1.set_xlabel(r"f1")
    ax1.set_ylabel(r"f2")
    ax1.set_ylabel(r"f3")
    # plt.savefig(fig_name + "_2.pdf", dpi=1000)

    # ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    # ax2.set_box_aspect((1, 1, 1))
    # ax2.view_init(45, 45)
    # ax2.set_title("Before/After")
    # ax2.set_xlabel(r"$f_1$")
    # ax2.set_ylabel(r"$f_2$")
    # ax2.set_ylabel(r"$f_3$")
    # ax2.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k.", ms=12, alpha=0.3)
    # ax2.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g+", ms=8, alpha=0.8)
    # lgnd = ax2.legend(
    #     [r"$Y_0$", r"$Y_{\mathrm{final}}$"],
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.25),
    #     ncol=2,
    #     fancybox=True,
    # )
    # for handle in lgnd.legend_handles:
    #     handle.set_markersize(10)

    # fig, ax2 = plt.subplots(1, 1, figsize=(8, 6.5))
    # plt.subplots_adjust(right=0.85, left=0.2)

    # ax22 = ax2.twinx()
    # ax2.semilogy(range(1, len(hist_IGD) + 1), hist_IGD, "r-", label="IGD")
    # ax22.semilogy(range(1, len(hist_R_norm) + 1), hist_R_norm, "g--")
    # ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
    # ax2.set_ylabel("IGD", color="r")
    # # ax22.set_ylabel(r"R norm", color="g")
    # ax2.set_title("Performance")
    # ax2.set_xlabel("iteration")
    # ax2.set_xticks(range(1, len(hist_IGD) + 1))
    # ax2.legend()
    # # plt.tight_layout()
    # plt.savefig(fig_name + "_3.pdf", dpi=1000)
    plt.show()
    plt.savefig(fig_name, dpi=1000)
    plt.close(fig)


def plot_4d(y0, Y, ref, pareto_front, hist_Y, history_medoids, hist_IGD, hist_R_norm, fig_name):
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
    ax2.set_xticks(range(1, len(hist_IGD) + 1))
    ax2.legend()
    # plt.tight_layout()
    plt.savefig(fig_name, dpi=1000)

    # plt.show()
    # plt.savefig(fig_name, dpi=1000)
    # plt.close(fig)
