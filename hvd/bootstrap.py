from typing import Dict, List, Tuple

# import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.neighbors import LocalOutlierFactor
from sklearn_extra.cluster import KMedoids

from .problems import MOOAnalytical
from .reference_set import ReferenceSet
from .rsg import RSG
from .utils import compute_chim, get_non_dominated

plt.style.use("ggplot")
rcParams["font.size"] = 15
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1


def plot_bootstrap(
    ref0: np.ndarray, ref: np.ndarray, Y0: np.ndarray, Y: np.ndarray, pareto_front: np.ndarray, name: str
):
    colors = plt.get_cmap("tab20").colors
    colors = [colors[2], colors[12], colors[13], colors[15], colors[19]]
    fig = plt.figure(figsize=plt.figaspect(1 / 2.0))
    plt.subplots_adjust(bottom=0.08, top=0.9, right=0.93, left=0.05)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax0.set_box_aspect((1, 1, 1))
    ax0.view_init(45, 45)
    ax0.plot(ref0[:, 0], ref0[:, 1], ref0[:, 2], "r+", ms=6, alpha=0.6)
    ax0.plot(ref[:, 0], ref[:, 1], ref[:, 2], "g.", ms=6, alpha=0.6)
    ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.4)
    ax0.set_title("reference set")
    ax0.set_xlabel(r"$f_1$")
    ax0.set_ylabel(r"$f_2$")
    ax0.set_zlabel(r"$f_3$")
    lgnd = ax0.legend(
        [r"$R_0$", r"$R_1$", "PF"], loc="lower center", bbox_to_anchor=(0.5, -0.15), ncols=3, fancybox=True
    )
    for handle in lgnd.legend_handles:
        handle.set_markersize(10)

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(45, 45)
    ax1.plot(Y0[:, 0], Y0[:, 1], Y0[:, 2], "g+", ms=8, alpha=0.6)
    ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r.", ms=8, alpha=0.6)
    ax1.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", mec="none", ms=5, alpha=0.4)
    ax1.set_title("Population")
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.set_ylabel(r"$f_3$")
    lgnd = ax1.legend(
        [r"$Y_0$", r"filtered $Y_1$", "PF"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncols=3,
        fancybox=True,
    )
    plt.tight_layout()
    plt.savefig(name, dpi=1000)


def filter_outliers(data: List) -> np.ndarray:
    X, Y = data[0], data[1]
    idx = get_non_dominated(Y, return_index=True)
    X, Y = X[idx], Y[idx]
    idx = LocalOutlierFactor(n_neighbors=3).fit_predict(Y)
    X, Y = X[idx == 1], Y[idx == 1]
    if 11 < 2:
        idx = np.isclose(Y.sum(axis=1), 0.5, atol=1e-1)
        X, Y = X[idx == 1], Y[idx == 1]
    return X, Y


def bootstrap_reference_set(
    optimizer,
    problem: MOOAnalytical,
    interval: int = 5,
    with_rsg: bool = True,
    plot: bool = True,
    save_reference_set: bool = False,
    save_population: bool = False,
    last_n: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Bootstrap the reference set with the intermediate population of an MOO algorithm

    Args:
        optimizer (_type_): an MOO algorithm
        problem (MOOAnalytical): the MOO problem to solve
        init_ref (np.ndarray): the initial reference set
        interval (int, optional): intervals at which bootstrapping is performed. Defaults to 5
        with_rsg (bool, optional): whether to use Angel's RSG method to interpolate the reference set
        save_reference_set (bool, optional): whether to save the intermediate reference sets or not
        save_population (bool, optional): whether to save the population in each generation
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]: (efficient set, Pareto front approximation,
            the stopping criteria)
    """
    opt_name: str = optimizer.__class__.__name__
    dim: int = optimizer.dim_p
    N: int = optimizer.N
    alpha: float = 0.05
    Y0: np.ndarray = optimizer.state.Y.copy()
    ref0: np.ndarray = optimizer.ref.reference_set.copy()
    ref_list: list = []
    X_list: list = []
    Y_list: list = []
    pareto_front: np.ndarray = problem.get_pareto_front()
    problem_name: str = problem.__class__.__name__

    # if with_rsg:  # initialize the matlab backend
    # eng = matlab.engine.start_matlab()
    # eng.cd(r"./RSG/", nargout=0)

    for i in range(optimizer.max_iters):
        X_list.append(optimizer.state.X.copy()[:, :dim])  # only keep the primal variables
        Y_list.append(optimizer.state.Y.copy())
        if save_population:
            X, Y = optimizer.state.X, optimizer.state.Y
            pd.DataFrame(X).to_csv(f"X_{opt_name}_{problem_name}_iteration{i}.csv", index=False, header=False)
            pd.DataFrame(Y).to_csv(f"Y_{opt_name}_{problem_name}_iteration{i}.csv", index=False, header=False)

        # TODO: generalize the condition to trigger the boostrap:
        # maybe if the R norm for most points are near zero?
        if i % interval == 0 and i > 0:
            # relax dominance resistant points (improper dominance)
            func = lambda y: (1 - alpha) * y + alpha * y.mean(axis=1).reshape(len(y), -1)
            Y = func(optimizer.state.Y)
            if 11 < 2:
                is_feasible = optimizer.state.is_feasible()
                is_KKT = optimizer.state.check_KKT()
                idx = np.bitwise_and(is_KKT, is_feasible)
                Y = Y[idx]

            # only take the non-dominated points
            idx = get_non_dominated(Y, return_index=True)
            Y = Y[idx]
            # take out the outliers
            # TODO: not robust FIXIT!
            # idx = np.isclose(Y.sum(axis=1), 0.5, atol=1e-1)
            # Y = Y[idx]
            indices = LocalOutlierFactor(n_neighbors=3).fit_predict(Y)
            Y = Y[indices == 1]
            if with_rsg:  # call the RSG method written in Matlab to fill the reference set
                ref = RSG(Py=Y, Nf=N)
                # pd.DataFrame(Y).to_csv("./RSG/MMD_boostrap.csv", index=False, header=False)
                # ref = np.array(eng.RSG())
            else:
                ref = Y.copy()
            # keep track of the new reference set
            ref_list.append(ref.copy())
            if plot:  # plot the interpolated reference set for visual inspection
                plot_bootstrap(ref0, ref, Y0, Y, pareto_front, f"{opt_name}_{problem_name}_iteration{i}.pdf")
                Y0 = Y.copy()
                ref0 = ref.copy()
            # whether to use the reference set in the history as the new one
            if 11 < 2:
                R = np.concatenate(ref_list, axis=0)
                km = KMedoids(n_clusters=N, method="alternate", random_state=0, init="k-medoids++").fit(R)
                ref = R[km.medoid_indices_]
            if save_reference_set:
                pd.DataFrame(ref).to_csv(f"ref_{opt_name}-iteration{i}.csv", index=False, header=False)

            # TODO: computing shift direction seems to be problematic. FIXIT
            eta = compute_chim(ref)
            ref += 0.05 * eta
            ref = ReferenceSet(ref=ref, eta={0: eta}, Y_idx=None)
            optimizer.indicator.ref = optimizer.ref = ref
            optimizer.indicator.compute(Y=optimizer.state.Y)
            # discount the weight of the spread term to reduce spread effect in the following iterations
            optimizer.indicator.beta = optimizer.indicator.beta * 0.8

        # the next iteration
        optimizer.newton_iteration()
        optimizer.log()

    # filter out the outliers in the approximation set
    filtered = map(filter_outliers, zip(X_list[-last_n:], Y_list[-last_n:]))
    X_, Y_ = list(zip(*filtered))
    X_ = np.concatenate(X_, axis=0)
    Y_ = np.concatenate(Y_, axis=0)
    km = KMedoids(n_clusters=N, method="alternate", random_state=42, init="k-medoids++").fit(Y_)
    X0_bootstrap = X_[km.medoid_indices_]
    Y0_bootstrap = Y_[km.medoid_indices_]
    return X0_bootstrap, Y0_bootstrap, ref, ref_list
