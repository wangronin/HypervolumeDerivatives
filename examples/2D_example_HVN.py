import matplotlib.pyplot as plt
import numpy as np
from hvd.algorithm import HVN
from matplotlib import rcParams

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

np.random.seed(66)

dim = 2
ref = np.array([20, 20])
mu = 5
max_iters = 10


def MOP1(x):
    x = np.array(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(dim), 2 * np.eye(dim)])


def h(x):
    x = np.array(x)
    return np.sum(x**2) - 1


def h_Jacobian(x):
    x = np.array(x)
    sign = 1 if np.sum(x**2) - 1 >= 0 else -1
    return 2 * x
    #  * sign


def h_Hessian(x):
    x = np.array(x)
    sign = 1 if np.sum(x**2) - 1 >= 0 else -1
    return 2 * np.eye(dim)
    # * sign


x0 = np.random.rand(mu, dim) * 2 - 1
idx = x0[:, 0] < x0[:, 1]
a = x0[idx, 0]
x0[idx, 0] = x0[idx, 1]
x0[idx, 1] = a

x0 = np.array(
    [
        [0.5, -1.5],
        [0.75, -1.25],
        [1, -1],
        [1.25, -0.75],
        [1.5, -0.5],
        # [-0.5, -0.75],
        # [-0.25, -0.5],
        # [0, -0.25],
    ]
)
y0 = np.array([MOP1(_) for _ in x0])

opt = HVN(
    dim=dim,
    n_objective=2,
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

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6.5))
plt.subplots_adjust(right=0.95, left=0.05)
ciricle = plt.Circle((0, 0), 1, color="r", fill=False, ls="--", lw=1.5)

ax0.plot(opt.X[:, 0], opt.X[:, 1], "g*")
ax0.plot(x0[:, 0], x0[:, 1], "g.", ms=8)
ax0.add_patch(ciricle)
ax0.set_xlim([-1.55, 1.55])
ax0.set_ylim([-1.55, 1.55])
ax0.set_title("decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

n_per_axis = 30
x = np.linspace(-1.55, 1.55, n_per_axis)
y = np.linspace(-1.55, 1.55, n_per_axis)
X, Y = np.meshgrid(x, y)
Z = np.array([MOP1(p) for p in np.array([X.flatten(), Y.flatten()]).T])
Z1 = Z[:, 0].reshape(-1, len(x))
Z2 = Z[:, 1].reshape(-1, len(x))
CS1 = ax0.contour(X, Y, Z1, 10, cmap=plt.cm.gray, linewidths=0.8, alpha=0.6)
CS2 = ax0.contour(X, Y, Z2, 10, cmap=plt.cm.gray, linewidths=0.8, linestyles="--", alpha=0.6)

trajectory = np.array([x0] + opt.hist_X)
for i in range(mu):
    x, y = trajectory[:, i, 0], trajectory[:, i, 1]
    ax0.quiver(
        x[:-1],
        y[:-1],
        x[1:] - x[:-1],
        y[1:] - y[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="k",
        width=0.005,
        headlength=4.7,
        headwidth=2.7,
    )

ax1.plot(opt.Y[:, 0], opt.Y[:, 1], "g*")
ax1.plot(y0[:, 0], y0[:, 1], "g.", ms=8)
trajectory = np.array([y0] + opt.hist_Y)
for i in range(mu):
    x, y = trajectory[:, i, 0], trajectory[:, i, 1]
    ax1.quiver(
        x[:-1],
        y[:-1],
        x[1:] - x[:-1],
        y[1:] - y[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="k",
        width=0.005,
        headlength=4.7,
        headwidth=2.7,
    )

x_vals = np.array([0, 6])
y_vals = 6 - x_vals
ax1.plot(x_vals, y_vals, "r--")
ax1.set_title("objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax22 = ax2.twinx()
ax2.plot(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
ax22.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_G_norm, "g--")
ax2.set_ylabel("HV", color="b")
ax22.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))

plt.savefig("2D-example.pdf", dpi=100)
