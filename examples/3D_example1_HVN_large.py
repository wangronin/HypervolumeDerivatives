import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from hvd.algorithm import HVN
from matplotlib import rcParams

np.random.seed(66)

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
ref = np.array([24, 24, 24])
max_iters = 50
mu = 100

c1 = np.array([1.5, 0, np.sqrt(3) / 3])
c2 = np.array([1.5, 0.5, -1 * np.sqrt(3) / 6])
c3 = np.array([1.5, -0.5, -1 * np.sqrt(3) / 6])
c1_n = c1 / np.linalg.norm(c1)
c2_n = c2 / np.linalg.norm(c2)
c3_n = c3 / np.linalg.norm(c3)


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


def h(x):
    x = np.array(x)
    return np.sum(x**2) - 1


def h_Jacobian(x):
    x = np.array(x)
    return 2 * x


def h_Hessian(x):
    return 2 * np.eye(dim)


pareto_set = [np.atleast_2d(c1)]
for s in np.arange(1, 13):
    v = s * np.sqrt(3) / 2 / 12
    y = v * np.tan(np.pi / 6)
    N = s * 2
    pareto_set.append(
        np.stack([np.array([1.5] * N), np.linspace(-y, y, N), np.array([np.sqrt(3) / 3 - v] * N)]).T
    )
pareto_set = np.concatenate(pareto_set, axis=0)
pareto_set /= np.linalg.norm(pareto_set, axis=1).reshape(-1, 1)
pareto_front = np.array([MOP1(x) for x in pareto_set])


# x0 = np.random.rand(mu, dim) * 4 - 2
x0 = np.c_[np.random.rand(mu, 1) * 0.5 + 0.8, np.random.rand(mu, dim - 1) - 0.5]
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
ax.view_init(25, -35)

u, v = np.mgrid[0 : 2 * np.pi : 16j, 0 : np.pi : 16j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, alpha=0.4)
ax.scatter(
    pareto_set[:, 0],
    pareto_set[:, 1],
    pareto_set[:, 2],
    color="k",
    marker="o",
    edgecolors="none",
    alpha=0.2,
    linewidths=1.5,
)

# plot the final decision points
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g*", ms=6)
ax.set_title("decision space")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_xlim([-1.3, 1.3])
ax.set_ylim([-1.3, 1.3])
ax.set_zlim([-1.3, 1.3])

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(20, -110)

x, y, z = pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2]
triang = mtri.Triangulation(x, y)
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
zmid = z[triang.triangles].mean(axis=1)

p = np.c_[xmid, ymid, zmid]
mask = np.array([np.any(np.all(pp > p, axis=1)) for pp in p])
mask[np.nonzero(mask)[0][8]] = False
triang.set_mask(mask)

ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g*", ms=8)
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
plt.savefig(f"3D-example1-{mu}.pdf", dpi=100)

df = pd.DataFrame(dict(iteration=range(1, len(opt.hist_HV) + 1), HV=opt.hist_HV, G_norm=opt.hist_G_norm))
df.to_latex(buf=f"3D-example1-{mu}.tex", index=False)
