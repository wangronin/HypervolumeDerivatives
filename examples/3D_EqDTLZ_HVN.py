import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd
from hvd.algorithm import HVN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3
from matplotlib.patches import Circle

np.random.seed(42)
plt.style.use("ggplot")

f = Eq1DTLZ1()

pareto_set = f.get_pareto_set(50)
pareto_front = f.get_pareto_front(50)

dim = 11
ref = np.array([500, 500, 700])
max_iters = 5

# a = np.mgrid[0.1:0.9:5j, 0.1:0.9:3j]
# a = np.array(list(zip(a[0].ravel(), a[1].ravel())))
# a += 0.1 * np.random.rand(len(a), 2)
# x0 = np.c_[a, 0.5 + 0.02 * np.random.rand(len(a), 1)]
# x0 = f.get_pareto_set(mu) + 0.05 * np.random.rand(mu, 3)

x0 = pd.read_csv("~/Downloads/EqDTLZ1.txt", header=None, sep=",").values
x0[:, 0:2] += 0.01 * np.random.rand(len(x0), 2)
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
)
X, Y, stop = opt.run()

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(50, -25)

# plot the initial and final approximation set
ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "r.", ms=2)
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g.", ms=2)

# plot the constraint boundary
# p = Circle((0.5, 0.5), 0.4, fc="None", ec="gray")
# ax.add_patch(p)
# art3d.pathpatch_2d_to_3d(p, z=0.5, zdir="z")
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

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(45, 45)
# plot the initial and final approximation set
ax.plot(y0[:, 0], y0[:, 1], y0[:, 2], "r.", ms=4)
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g.", ms=4)
ax.plot3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "gray", alpha=0.4)

ax.set_xlabel("f1")
ax.set_ylabel("f2")
ax.set_zlabel("f3")
ax.set_xlim([0, 0.3])
ax.set_ylim([0, 0.3])
ax.set_zlim([0, 0.5])

ax = fig.add_subplot(1, 3, 3)
ax.semilogy(range(1, len(opt.hist_G_norm) + 1), opt.hist_G_norm, "g--")
ax.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax.set_title("Performance")
ax.set_xlabel("iteration")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig(f"{type(f).__name__}-{mu}.pdf", dpi=100)

plt.show()
