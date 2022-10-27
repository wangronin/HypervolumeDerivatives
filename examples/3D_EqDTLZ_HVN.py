import matplotlib.pyplot as plt
import numpy as np
from hvd.algorithm import HVN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3, Eq1DTLZ4, Eq1IDTLZ1, Eq1IDTLZ2, Eq1IDTLZ3, Eq1IDTLZ4
from matplotlib import rcParams


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
    "Eq1IDTLZ3": 66,
    "Eq1IDTLZ4": 42,
}
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

# NOTE: on Eq1DTLZ4 and Eq1IDTLZ4 problems, we face a numerical issue as the Newton step is
# very tiny when the decision points are on the Pareto front, which is even below the numerical precision
# of Python :) -> Maybe utilize numerical libraries that allows for arbitrary precisions
f = Eq1DTLZ2()
dim = 11
max_iters = 5
ref = refs[type(f).__name__]
seed = seeds[type(f).__name__]

np.random.seed(seed)
pareto_set = f.get_pareto_set(500)
pareto_front = f.get_pareto_front(500)

x0 = f.get_pareto_set(60, kind="uniform")
x0[:, 0:2] += 0.02 * np.random.rand(len(x0), 2)  # perturb the initial solution a bit
y0 = np.array([f.objective(x) for x in x0])
mu = len(x0)

opt = HVN(
    dim=dim,
    n_objective=3,
    ref=ref,
    func=f.objective,
    jac=f.objective_jacobian,
    hessian=f.objective_hessian,
    h=f.constraint,
    h_jac=f.constraint_jacobian,
    h_hessian=f.constraint_hessian,
    mu=mu,
    lower_bounds=0,
    upper_bounds=1,
    minimization=True,
    x0=x0,
    max_iters=max_iters,
    verbose=True,
    problem_name=type(f).__name__,
)
X, Y, stop = opt.run()

fig = plt.figure(figsize=plt.figaspect(2))
ax = fig.add_subplot(2, 1, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(50, -20)

# plot the initial and final approximation set
ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "k+", ms=5, alpha=0.5)
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g.", ms=7, alpha=0.5)

# plot the constraint boundary
ax.plot3D(pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2], "gray", alpha=0.4)

# trajectory = np.atleast_3d([x0] + opt.hist_X)
# for i in range(len(trajectory[0])):
#     x, y, z = trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2]
#     ax.quiver(
#         x[:-1],
#         y[:-1],
#         z[:-1],
#         x[1:] - x[:-1],
#         y[1:] - y[:-1],
#         z[1:] - z[:-1],
#         color="k",
#         arrow_length_ratio=0.2,
#         alpha=1,
#     )

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
# ax.set_zlabel("$x_3$")
ax.set_xlim([0.1, 0.9])
ax.set_ylim([0.1, 0.9])
ax.set_zlim([0.1, 0.9])
# ax.set_title("decision space")
ax.set_title(type(f).__name__)
ax.text2D(0.95, 0.5, "$x_3$", transform=ax.transAxes, fontsize=15)
ax.text2D(-0.05, 0.4, "decision space", transform=ax.transAxes, rotation=90, fontsize=15)

ax = fig.add_subplot(2, 1, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(45, 45)
# plot the initial and final approximation set
# y0 = bilog(y0)
# Y = bilog(Y)
# pareto_front = bilog(pareto_front)
# ax.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k+", ms=5, alpha=0.5)
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g.", ms=5, alpha=0.4)
ax.plot3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "gray", alpha=0.4)

ax.set_xlabel("$f_1$")
ax.set_ylabel("$f_2$")
# ax.set_zlabel("f3")
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.set_zlim([0, 1])
# ax.set_title("objective space")
ax.text2D(0.1, 0.55, "$f_3$", transform=ax.transAxes, fontsize=15)
ax.text2D(-0.05, 0.4, "objective space", transform=ax.transAxes, rotation=90, fontsize=15)

# ax = fig.add_subplot(1, 3, 3)
# x = list(range(1, len(opt.hist_G_norm) + 1))
# ax.semilogy(x, opt.hist_G_norm, "g--")
# ax.set_title(r"$||G(\mathbf{X})||$", color="g")
# ax.set_xlabel("iteration")
# ax.set_xticks(x)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig(f"{type(f).__name__}-{mu}.pdf", dpi=100)
