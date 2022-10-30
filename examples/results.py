import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hvd.hypervolume import hypervolume
from hvd.problems import *
from hvd.utils import non_domin_sort

refs = {
    "Eq1DTLZ1": np.array([1, 1, 1]),
    "Eq1DTLZ2": np.array([1, 1, 1]),
    "Eq1DTLZ3": np.array([1, 1, 1]),
    "Eq1DTLZ4": np.array([1.2, 5e-3, 5e-4]),
    "Eq1IDTLZ1": np.array([1, 1, 1]),
    "Eq1IDTLZ2": np.array([1, 1, 1]),
    "Eq1IDTLZ3": np.array([1, 1, 1]),
    "Eq1IDTLZ4": np.array([-0.4, 0.6, 0.6]),
}


def generate_plot(X, index, problem, algorithm):
    f = globals()[problem](3, 11)
    pareto_set = f.get_pareto_set(500)
    pareto_front = f.get_pareto_front(500)
    Y = np.array([f.objective(x) for x in X])

    fig = plt.figure(figsize=plt.figaspect(1 / 1))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    # ax.set_box_aspect((1, 1, 1))
    # ax.view_init(50, -25)

    # # plot the initial and final approximation set
    # ax.plot(X[:, 0], X[:, 1], X[:, 2], "g.", ms=7, alpha=0.5)
    # # plot the constraint boundary
    # ax.plot3D(pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2], "gray", alpha=0.5)

    # ax.set_xlabel("x1")
    # ax.set_ylabel("x2")
    # ax.set_zlabel("x3")
    # ax.set_xlim([0.1, 0.9])
    # ax.set_ylim([0.1, 0.9])
    # ax.set_zlim([0.1, 0.9])
    # ax.set_title("decision space")
    # ax.text2D(-0.05, 0.4, type(f).__name__, transform=ax.transAxes, rotation=90, fontsize=15)

    # ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(45, 45)
    # plot the initial and final approximation set
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g.", ms=7, alpha=0.5)
    ax.plot3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "gray", alpha=0.4)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    # ax.set_xlim([0, 0.3])
    # ax.set_ylim([0, 0.3])
    # ax.set_zlim([0, 0.5])
    # ax.set_title("objective space")
    ax.set_title(algorithm)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(f"{problem_name}-{algorithm}-{index}.pdf", dpi=100)


dfs = []
problems = []
files = sorted(glob.glob("*IDTLZ*-hybrid.npz"))
# for the hybrid algorithm
for file in files:
    problem_name = file.split("-")[0]
    ref = refs[problem_name]
    # problem_name = file.split("/")[1].split("-")[0]

    problems.append(problem_name)
    f = globals()[problem_name](3, 11)

    data = np.load(file, allow_pickle=True)["data"]
    HV, HV0, pop_size, pop_size_ = [], [], [], []
    for i, (X, X_, _CPU_time, _HV0, _HV) in enumerate(data):
        if 11 < 2:
            generate_plot(X, i, problem_name, "hybrid")

        if problem_name == "Eq1IDTLZ2":
            breakpoint()

        Y = np.array([f.objective(x) for x in X])
        Y_ = np.array([f.objective(x) for x in X_])
        idx = non_domin_sort(Y, only_front_indices=True)[0]
        idx_ = non_domin_sort(Y_, only_front_indices=True)[0]
        HV0.append(hypervolume(Y_[idx_], ref))
        HV.append(hypervolume(Y[idx], ref))
        pop_size.append(len(X[idx]))
        pop_size_.append(len(X_[idx_]))

    results = [
        [np.mean(HV0), np.std(HV0) / np.sqrt(15), np.mean(pop_size_), np.std(pop_size_) / np.sqrt(15)],
        [
            np.mean(HV),
            np.std(HV) / np.sqrt(15),
            np.mean(pop_size),
            np.std(pop_size) / np.sqrt(15),
        ],
    ]
    # for NSGA-III
    HV = []
    pop_size = []
    data = np.load(f"{problem_name}-NSGA3.npz", allow_pickle=True)["data"]
    for i, (X, _HV) in enumerate(data):
        if 11 < 2:
            generate_plot(X, i, problem_name, "NSGA3")

        Y = np.array([f.objective(x) for x in X])
        idx = non_domin_sort(Y, only_front_indices=True)[0]
        HV.append(hypervolume(Y[idx], ref))
        pop_size.append(len(X[idx]))

    results.append(
        [
            np.mean(HV),
            np.std(HV) / np.sqrt(15),
            np.mean(pop_size),
            np.std(pop_size) / np.sqrt(15),
        ]
    )
    dfs.append(pd.DataFrame(results))

columns = pd.MultiIndex.from_product(
    [
        problems,
        ["mean HV", "std. err HV", "mean #nondominated", "std. err #nondominated"],
    ],
)
df = pd.concat(dfs, axis=1, ignore_index=True)
df.columns = columns
df.insert(
    0,
    "Algorithm",
    ["NSGA-III (iter = 1000)", "NSGA-III (iter = 1000) + HVN (iter = 10)", "NSGA-III (iter = 2400)"],
)
df.to_latex(buf=f"Eq1DTLZ.tex", index=False)
