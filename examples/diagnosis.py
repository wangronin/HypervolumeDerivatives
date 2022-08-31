import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from hvd.algorithm import HVN
from hvd.hypervolume_derivatives import get_non_dominated
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import qmc

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000)


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


def h(x):
    x = np.array(x)
    return np.sum((x - np.array([2 / np.sqrt(3) - 1, 0, -1.5])) ** 2) - 1


def h_Jacobian(x):
    x = np.array(x)
    return 2 * (x - np.array([2 / np.sqrt(3) - 1, 0, -1.5]))


def h_Hessian(_):
    return 2 * np.eye(3)


N = 10
point_set = []
for i, x in enumerate(np.linspace(-0.95, 0.95, int(np.sqrt(N)))):
    n = 1 + 2 * i
    y_ = -0.5 * x - 0.5 + 0.0250
    _y = 0.5 * x + 0.5 - 0.0250
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

w = np.abs(np.random.rand(30, 3))
w /= np.sum(w, axis=1).reshape(-1, 1)
x0 = w @ np.vstack([c1, c2, c3])
# x0 -= np.array([2 / np.sqrt(3) - 1, 0, -1.5])
# x0 /= np.linalg.norm(x0, axis=1).reshape(-1, 1)
# x0 += np.array([2 / np.sqrt(3) - 1, 0, -1.5])
# x0 = point_set
y0 = np.array([MOP1(_) for _ in x0])
idx = get_non_dominated(y0, return_index=True, weakly_dominated=True)
ref = np.array([30, 30, 30])
max_iters = 35

opt = HVN(
    dim=3,
    n_objective=3,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    h=h,
    h_jac=h_Jacobian,
    h_hessian=h_Hessian,
    mu=len(x0),
    x0=x0,
    lower_bounds=-2,
    upper_bounds=2,
    minimization=True,
    max_iters=max_iters,
    verbose=True,
)

X, Y, stop = opt.run()


fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(35, -50)

ax.add_collection3d(Poly3DCollection([np.vstack([c1, c2, c3])], color="k", alpha=0.3))
u, v = np.mgrid[0 : 2 * np.pi : 16j, 0 : np.pi : 16j]
r = 1
x = r * np.cos(u) * np.sin(v)
y = r * np.sin(u) * np.sin(v)
z = r * np.cos(v)
x += 2 / np.sqrt(3) - 1
z -= 1.5
ax.plot_wireframe(x, y, z, alpha=0.4)

# plot the initial decision points
# ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "r.", ms=8)
# ax.plot(x0[-1, 0], x0[-1, 1], x0[-1, 2], "r.", ms=8)
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g.", ms=8)
ax.set_title("decision space")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

# trajectory = np.atleast_3d([x0] + opt.hist_X)
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
#         arrow_length_ratio=0.08,
#         alpha=0.35,
#     )

# for i, x in enumerate(x0):
#     ax.quiver(
#         x[0],
#         x[1],
#         x[2],
#         HVdX[i, 0],
#         HVdX[i, 1],
#         HVdX[i, 2],
#         color="k",
#         arrow_length_ratio=0.05,
#         alpha=0.35,
#     )

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(-7, -135)

# ax.plot(y0[:, 0], y0[:, 1], y0[:, 2], "g.", ms=8)
# ax.plot(y0[-1, 0], y0[-1, 1], y0[-1, 2], "r.", ms=8)
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g*", ms=8)

# for i, y in enumerate(y0):
#     ax.quiver(
#         y[0],
#         y[1],
#         y[2],
#         HVdY[i, 0],
#         HVdY[i, 1],
#         HVdY[i, 2],
#         color="k",
#         arrow_length_ratio=0.05,
#         alpha=0.35,
#     )

# trajectory = np.atleast_3d([y0] + opt.hist_Y)
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
ax_ = ax.twinx()
ax.plot(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
# ax_.plot(range(1, len(opt.hist_HV) + 1), opt._hist_inner, "g--")
ax_.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_G_norm, "g--")
ax.set_ylabel("HV", color="b")
ax_.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax.set_title("Performance")
ax.set_xlabel("iteration")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()
# plt.savefig(f"3D-example1-{len(x0)}.pdf", dpi=100)
