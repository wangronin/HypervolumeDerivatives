import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hvd.problems import *


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
# for the hybrid algorithm
for file in glob.glob("data-DTLZ/*-hybrid.npz"):
    problem_name = file.split("/")[1].split("-")[0]
    problems.append(problem_name)
    data = np.load(file, allow_pickle=True)["data"]
    HV0 = []
    HV = []
    pop_size, pop_size_ = [], []
    for i, (X, X_, _CPU_time, _HV0, _HV) in enumerate(data):
        if 11 < 2:
            generate_plot(X, i, problem_name, "hybrid")

        HV0.append(_HV0)
        HV.append(_HV)
        pop_size.append(len(X))
        pop_size_.append(len(X_))

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
    data = np.load(f"data-DTLZ/{problem_name}-NSGA3.npz", allow_pickle=True)["data"]
    for i, (X, _HV) in enumerate(data):
        if 11 < 2:
            generate_plot(X, i, problem_name, "NSGA3")
        HV.append(_HV)
        pop_size.append(len(X))

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
