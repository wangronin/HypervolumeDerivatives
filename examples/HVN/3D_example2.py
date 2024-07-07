import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hvd.newton import HVN

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000)

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


c1 = np.array([-1, -1, -1])
c2 = np.array([-1, 0, 0])
c3 = np.array([-2, -2, 4])


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
    return np.array(
        [
            2 * (x - c1),
            2 * (x - c2),
            2 * (x - c3),
        ]
    )


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(3), 2 * np.eye(3), 2 * np.eye(3)])


# tol = 1e-10


def h(x):
    return x[0]


def h_Jacobian(x):
    v = np.zeros(len(x))
    v[0] = 1
    return v
    # if x[0] < tol else np.zeros(len(x))


def h_Hessian(x):
    return np.zeros((3, 3))


pareto_set = np.array([[0, -1, -1], [0, 0, 0], [0, -2, 4]])
point_set = [np.array([[0, -1, -1], [0, 0, 0], [0, -2, 4]])]
for z in np.linspace(-1.01, 3.99, 50):
    _y = z / -2 if z >= 0 else z
    y_ = (z + 6) / -5
    N = int(10 * (_y - y_))
    point_set.append(np.stack([np.array([0] * N), np.linspace(y_ - 0.01, _y + 0.01, N), np.array([z] * N)]).T)
point_set = np.concatenate(point_set, axis=0)
pareto_front = np.array([MOP1(x) for x in point_set])

dim = 3
ref = np.array([90, 90, 90])
max_iters = 30

# x0 = np.array(
#     [
#         [0.7, -1.8, -1],
#         [0.65, 0, -1.2],
#         [1.01, -2, 0.5],
#         [1, 0.5, 1],
#         [0.7, 1.5, -1.5],
#         [1.2, 1.5, 2],
#     ]
# )
mu = 20
# a = np.mgrid[-2.5:-0.5:6j, 0:3.5:10j]
# a = np.array(list(zip(a[0].ravel(), a[1].ravel())))
w = np.abs(np.random.randn(mu, 3))
w /= np.sum(w, axis=1).reshape(-1, 1)
x0 = w @ np.vstack([c1, c2, c3])
x0[:, 0] = 0.5
x0[0, 0] = 3.5

# x0 = np.c_[np.tile(0.5, (len(a), 1)), a]
y0 = np.array([MOP1(_) for _ in x0])

opt = HVN(
    n_var=dim,
    n_obj=3,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    g=h,
    g_jac=h_Jacobian,
    g_hessian=h_Hessian,
    N=len(x0),
    X0=x0,
    xl=-4,
    xu=4,
    max_iters=max_iters,
    verbose=True,
    preconditioning=False,
)
X, Y, stop = opt.run()

fig = plt.figure(figsize=plt.figaspect(1 / 3))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(25, -55)

# plot the constraint boundary
yy, zz = np.mgrid[-2.5:2:5j, -1.5:4.5:5j]
xx = np.zeros(yy.shape)
ax.plot_surface(xx, yy, zz, alpha=0.2)
# plot the efficient set
ax.add_collection3d(Poly3DCollection([pareto_set], color="k", alpha=0.3))
# plot the initial points
# ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "g.", ms=10)
# plot the final decision points
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g*", ms=6)
# ax.plot(point_set[:, 0], point_set[:, 1], point_set[:, 2], "r.", ms=8)
ax.set_title("decision space")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_ylim([-2.5, 2])
ax.set_xlim([-1.5, 1.5])

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

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(30, 13)

x, y, z = pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2]
triang = mtri.Triangulation(x, y)
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
zmid = z[triang.triangles].mean(axis=1)

p = np.c_[xmid, ymid, zmid]
mask = np.array([np.any(np.all(pp > p, axis=1)) for pp in p])
triang.set_mask(mask)

# ax.plot(x, y, z, "r.", ms=8)
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g*", ms=8)
# plot the initial points
# ax.plot(y0[:, 0], y0[:, 1], y0[:, 2], "g.", ms=10)
ax.plot_trisurf(triang, z, color="k", alpha=0.2)

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
# ax_ = ax.twinx()
# ax.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
# ax.plot(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
ax.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
# ax.set_ylabel("HV", color="b")
# ax.set_ylabel("N_Nondominated", color="b")
ax.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax.set_title("Performance")
ax.set_xlabel("iteration")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig(f"3D-example2-{mu}.pdf", dpi=100)

# df = pd.DataFrame(dict(iteration=range(1, len(opt.hist_HV) + 1), HV=opt.hist_HV, G_norm=opt.hist_G_norm))
# df.to_latex(buf=f"3D-example2-{mu}.tex", index=False)
