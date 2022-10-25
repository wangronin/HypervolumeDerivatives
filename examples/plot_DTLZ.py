import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3

f = Eq1DTLZ1(3, 11)
pareto_set = f.get_pareto_set(300)
pareto_front = f.get_pareto_front(300)

data = np.load("Eq1DTLZ1-NSGA3.npz", allow_pickle=True)["data"]
# CPU_time = []
pop_size = []
# for i, (X, _time) in enumerate(data):
for i, X in enumerate(data):
    # X = X[:, :-1]
    Y = np.array([f.objective(x) for x in X])
    mu = len(X)
    # print(f"CPU time: {_time}s, pop. size: {mu}")
    # CPU_time.append(_time)
    pop_size.append(mu)

    fig = plt.figure(figsize=plt.figaspect(1 / 2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(50, -25)
    # ax.view_init(45, 45)

    # plot the initial and final approximation set
    # ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "r.", ms=5, alpha=0.5)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], "g.", ms=7, alpha=0.5)
    # plot the constraint boundary
    ax.plot3D(pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2], "gray", alpha=0.5)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    ax.set_zlim([0.1, 0.9])
    ax.set_title("decision space")
    ax.text2D(-0.05, 0.4, type(f).__name__, transform=ax.transAxes, rotation=90, fontsize=15)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(45, 45)
    # plot the initial and final approximation set
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g.", ms=7, alpha=0.5)
    ax.plot3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "gray", alpha=0.4)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0, 0.3])
    ax.set_zlim([0, 0.5])
    ax.set_title("objective space")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(f"{type(f).__name__}-{mu}-NSGA3-{i}.pdf", dpi=100)

# print(f"median CPU time: {np.median(CPU_time)}")
# print(f"mean CPU time: {np.mean(CPU_time)}")
# print(f"std. CPU time: {np.std(CPU_time)}")
print(f"mean pop size: {np.mean(pop_size)}")
