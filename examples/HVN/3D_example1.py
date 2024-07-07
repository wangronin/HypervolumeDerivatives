import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from hvd.newton import HVN

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000)
plt.style.use("ggplot")

c1 = np.array([1, 1, 0])
c2 = np.array([1, -1, 0])
c3 = np.array([-1, 0, 0])


def MOP1(x):
    x = np.array(x)
    return np.array(
        [
            np.sum((x - c1) ** 2),
            np.sum((x - c2) ** 2),
            np.sum((x - c3) ** 2),
        ]
    )


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - c1), 2 * (x - c2), 2 * (x - c3)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(3), 2 * np.eye(3), 2 * np.eye(3)])


center = np.array([2 / np.sqrt(3) - 1, 0, -1.5])


def h(x):
    x = np.array(x)
    return np.sum((x - center) ** 2) - 1


def h_Jacobian(x):
    x = np.array(x)
    return 2 * (x - center)


def h_Hessian(_):
    return 2 * np.eye(3)


N = 200
point_set = []
for i, x in enumerate(np.linspace(-1, 1, int(np.sqrt(N)))):
    n = 1 + 2 * i
    y_ = -0.5 * x - 0.5
    _y = 0.5 * x + 0.5
    point_set.append(
        np.stack(
            [
                np.array([x] * n),
                np.linspace(y_, _y, n),
                np.array([0] * n),
            ]
        ).T
    )
point_set = np.concatenate(point_set, axis=0)
point_set -= center
pareto_set = point_set / np.linalg.norm(point_set, axis=1).reshape(-1, 1) + center
pareto_front = np.array([MOP1(x) for x in pareto_set])

max_iters = 30
mu = 40
ref = np.array([20, 20, 20])
w = np.abs(np.random.rand(mu, 3))
w /= np.sum(w, axis=1).reshape(-1, 1)
x0 = w @ np.vstack([c1, c2, c3])
# x0[0, 2] = 1
y0 = np.array([MOP1(_) for _ in x0])

# NOTE: the reference points affects the performance: how to set reference properly
if mu == 20:
    ref = np.array([20, 20, 20])
elif mu == 40:
    ref = np.array([38, 38, 38])
elif mu == 60:
    ref = np.array([100, 100, 100])

opt = HVN(
    n_var=3,
    n_obj=3,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    h=h,
    h_jac=h_Jacobian,
    h_hessian=h_Hessian,
    N=mu,
    X0=x0,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y, stop = opt.run()

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(50, -25)

u, v = np.mgrid[0 : 2 * np.pi : 18j, 0 : np.pi : 16j]
r = 1
x = r * np.cos(u) * np.sin(v)
y = r * np.sin(u) * np.sin(v)
z = r * np.cos(v)
x += 2 / np.sqrt(3) - 1
z -= 1.5
ax.plot_wireframe(x, y, z, alpha=0.3, color="r")

x, y, z = pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2]
triang = mtri.Triangulation(x, y)
ax.plot_trisurf(triang, z, color="k", alpha=0.3)

# plot the initial decision points
# ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "g.", ms=8)
# plot the final decision points
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g*", ms=6)
ax.set_title("decision space")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_xlim([-1.3, 1.3])
ax.set_ylim([-1.3, 1.3])
ax.set_zlim([-2.5, 0.3])

trajectory = np.atleast_3d([x0] + opt.history_X)
for i in range(len(x0)):
    x, y, z = trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2]
    ax.quiver(
        x[:-1],
        y[:-1],
        z[:-1],
        x[1:] - x[:-1],
        y[1:] - y[:-1],
        z[1:] - z[:-1],
        color="k",
        arrow_length_ratio=0.1,
        alpha=0.3,
    )

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(-12, -135)

x, y, z = pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2]
triang = mtri.Triangulation(x, y)
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
zmid = z[triang.triangles].mean(axis=1)

p = np.c_[xmid, ymid, zmid]
mask = np.array([np.any(np.all(pp > p, axis=1)) for pp in p])
triang.set_mask(mask)

# plot the Pareto front
ax.plot_trisurf(triang, z, color="k", alpha=0.2)
# plot the final Pareton approximation set
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g*", ms=8)

# trajectory = np.atleast_3d([y0] + opt.history_Y)
# for i in range(len(x0)):
#     x, y, z = trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2]
#     ax.quiver(
#         x[:-1],
#         y[:-1],
#         z[:-1],
#         x[1:] - x[:-1],
#         y[1:] - y[:-1],
#         z[1:] - z[:-1],
#         color="k",
#         arrow_length_ratio=0.05,
#         alpha=0.35,
#     )

ax.set_title("objective space")
ax.set_xlabel(r"$f_1$")
ax.set_ylabel(r"$f_2$")
ax.set_zlabel(r"$f_3$")

ax = fig.add_subplot(1, 3, 3)
# ax_ = ax.twinx()
# ax.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
ax.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
# ax.set_ylabel("HV", color="b")
ax.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax.set_title("Performance")
ax.set_xlabel("iteration")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
# plt.show()
plt.savefig(f"3D-example1-{mu}.pdf", dpi=100)

# df = pd.DataFrame(dict(iteration=range(1, len(opt.hist_HV) + 1), HV=opt.hist_HV, G_norm=opt.hist_G_norm))
# df.to_latex(buf=f"3D-example1-{mu}.tex", index=False)
