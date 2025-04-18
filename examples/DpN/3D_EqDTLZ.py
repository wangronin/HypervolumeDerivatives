import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import DpN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3, Eq1DTLZ4, Eq1IDTLZ1, Eq1IDTLZ2, Eq1IDTLZ3, Eq1IDTLZ4


def bilog(x):
    return np.sign(x) * np.log10(1 + np.abs(x))


plt.style.use("ggplot")
rcParams["font.size"] = 12
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

seeds = {
    "Eq1DTLZ1": 66,
    "Eq1DTLZ2": 42,
    "Eq1DTLZ3": 66,
    "Eq1DTLZ4": 42,
    "Eq1IDTLZ1": 66,
    "Eq1IDTLZ2": 42,
    "Eq1IDTLZ3": 123,
    "Eq1IDTLZ4": 42,
}

f = Eq1IDTLZ1()
dim = 11
max_iters = 20
seed = seeds[type(f).__name__]
np.random.seed(seed)

# create the utopian set
x_ref = f.get_pareto_set(500, kind="even")
x_ref[:, 0:2] = (x_ref[:, 0:2] - 0.5) * 0.35 / 0.4 + 0.5
ref = np.array([f.objective(x) for x in x_ref])

# the Pareto front/set
pareto_set = f.get_pareto_set(100)
pareto_front = f.get_pareto_front(100)
pareto_set = np.vstack([pareto_set, pareto_set[0, :]])

x0 = f.get_pareto_set(50, kind="uniform")
x0[:, 0:2] += 0.08 * np.random.rand(len(x0), 2)  # perturb the initial solution a bit
# x0 += 0.02 * np.random.rand(len(x0), dim)  # perturb the initial solution a bit
# y0 = np.array([f.objective(x) for x in x0])
N = len(x0)
opt = DpN(
    dim=dim,
    n_obj=3,
    ref=ref,
    func=f.objective,
    jac=f.objective_jacobian,
    hessian=f.objective_hessian,
    h=f.constraint,
    h_jac=f.constraint_jacobian,
    N=N,
    xl=0,
    xu=1,
    X0=x0,
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(50, -20)

# plot the initial and final approximation set
ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "g.", ms=5, alpha=0.5)
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g*", ms=7, alpha=0.5)

# plot the constraint boundary
ax.plot3D(pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2], "gray", alpha=0.4)
# plot the trajectory of decision points
trajectory = np.atleast_3d([x0] + opt.history_X)
for i in range(len(trajectory[0])):
    x, y, z = trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2]
    ax.quiver(
        x[:-1],
        y[:-1],
        z[:-1],
        x[1:] - x[:-1],
        y[1:] - y[:-1],
        z[1:] - z[:-1],
        color="k",
        arrow_length_ratio=0.2,
        alpha=1,
    )

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
# ax.set_zlabel("$x_3$")
ax.set_xlim([0.1, 0.9])
ax.set_ylim([0.1, 0.9])
ax.set_zlim([0.1, 0.9])
# ax.set_title("decision space")
ax.set_title(type(f).__name__)
ax.text2D(0.95, 0.5, "$x_3$", transform=ax.transAxes, fontsize=15)
ax.text2D(-0.1, 0.4, "decision space", transform=ax.transAxes, rotation=90, fontsize=15)

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(45, 45)

# plot the reference set
ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "r.", ms=5, alpha=0.4)
# plot the final Pareto approximation set
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g*", ms=5, alpha=0.4)
ax.plot3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "gray", alpha=0.4)

ax.set_xlabel("$f_1$")
ax.set_ylabel("$f_2$")
# ax.set_zlabel("f3")
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.set_zlim([0, 1])
# ax.set_title("objective space")
ax.text2D(0.1, 0.55, "$f_3$", transform=ax.transAxes, fontsize=15)
ax.text2D(-0.1, 0.4, "objective space", transform=ax.transAxes, rotation=90, fontsize=15)

ax = fig.add_subplot(1, 3, 3)
ax_ = ax.twinx()
ax.semilogy(range(1, len(opt.hist_GD) + 1), opt.hist_GD, "b-", label="GD")
ax.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax_.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
ax_.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax.set_title("Performance")
ax.set_xlabel("iteration")
ax.legend()

plt.tight_layout()
plt.savefig(f"{type(f).__name__}-{len(x0)}.pdf", dpi=700)

data = np.concatenate([np.c_[[0] * N, y0], np.c_[[max_iters] * N, opt.history_Y[-1]]], axis=0)
df = pd.DataFrame(data, columns=["iteration", "f1", "f2"])
df.to_csv(f"{f.__class__.__name__}_example.csv")
