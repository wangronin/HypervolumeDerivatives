import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from hvd.algorithm import HVN
from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.random.seed(42)

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


dim = 3
ref = np.array([90, 90, 90])
max_iters = 50
mu = 100

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
    return np.array([2 * np.eye(dim), 2 * np.eye(dim), 2 * np.eye(dim)])


tol = 1e-3


def h(x):
    return x[0] if x[0] < tol else 0.0


def h_Jacobian(x):
    v = np.zeros(len(x))
    v[0] = 1
    return v if x[0] < tol else np.zeros(len(x))


def h_Hessian(x):
    dim = len(x)
    return np.zeros((dim, dim))


pareto_set = np.array([[0, -1, -1], [0, 0, 0], [0, -2, 4]])
point_set = [np.array([[0, -1, -1], [0, 0, 0], [0, -2, 4]])]
for z in np.linspace(-1.01, 3.99, 50):
    _y = z / -2 if z >= 0 else z
    y_ = (z + 6) / -5
    N = int(10 * (_y - y_))
    point_set.append(np.stack([np.array([0] * N), np.linspace(y_ - 0.01, _y + 0.01, N), np.array([z] * N)]).T)
point_set = np.concatenate(point_set, axis=0)
pareto_front = np.array([MOP1(x) for x in point_set])

x0 = np.c_[np.random.rand(mu, 1), np.random.rand(mu, dim - 1) * 3 - 1.5]
opt = HVN(
    dim=dim,
    n_objective=3,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    h=h,
    h_jac=h_Jacobian,
    h_hessian=h_Hessian,
    mu=mu,
    x0=x0,
    lower_bounds=-4,
    upper_bounds=4,
    minimization=True,
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(25, -55)

# plot the constraint boundary
yy, zz = np.mgrid[-2.5:2:5j, -1.5:4.5:5j]
xx = np.zeros(yy.shape)
ax.plot_surface(xx, yy, zz, alpha=0.2)
# plot the efficient set
ax.add_collection3d(Poly3DCollection([pareto_set], color="k", alpha=0.3))
# plot the final decision points
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g*", ms=6)
ax.set_title("decision space")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_ylim([-2.5, 2])
ax.set_xlim([-1.5, 1.5])

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

ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g*", ms=8)
# plot the initial points
ax.plot_trisurf(triang, z, color="k", alpha=0.2)

ax.set_title("objective space")
ax.set_xlabel(r"$f_1$")
ax.set_ylabel(r"$f_2$")
ax.set_zlabel(r"$f_3$")

ax = fig.add_subplot(1, 3, 3)
ax_ = ax.twinx()
ax.plot(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
ax_.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_G_norm, "g--")
ax.set_ylabel("HV", color="b")
ax_.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax.set_title("Performance")
ax.set_xlabel("iteration")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig(f"3D-example2-{mu}.pdf", dpi=100)

df = pd.DataFrame(dict(iteration=range(1, len(opt.hist_HV) + 1), HV=opt.hist_HV, G_norm=opt.hist_G_norm))
df.to_latex(buf=f"3D-example2-{mu}.tex", index=False)
