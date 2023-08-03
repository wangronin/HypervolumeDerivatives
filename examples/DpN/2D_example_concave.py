import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import hessian, jacobian
from matplotlib import rcParams

from hvd.newton import DpN

plt.style.use("ggplot")
rcParams["font.size"] = 17
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 17
rcParams["ytick.labelsize"] = 17
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)


def F(x):
    x = np.array(x)
    return np.array([1 - np.exp(-np.sum((x - 1) ** 2)), 1 - np.exp(-np.sum((x + 1) ** 2))])


# TODO: the objective Hessian can be indefinite; find a systematic way to handle it
# Also, on concave Pareto front, the overall IGD Hessian can be indefinite.
Jacobian = jacobian(F)
Hessian = hessian(F)

p = np.linspace(-1, 1, 100)
ref_x = np.c_[p, p]
ref = np.array([F(_) for _ in ref_x])
ref -= np.array([0.01, 0.01])

max_iters = 20
mu = 10

x0 = np.linspace(-0.8, 0.8, mu)
x0 = np.c_[x0, x0] + np.array([0.5, -0.5])
y0 = np.array([F(_) for _ in x0])
opt = DpN(
    dim=2,
    n_objective=2,
    ref=ref,
    func=F,
    jac=Jacobian,
    hessian=Hessian,
    mu=len(x0),
    x0=x0,
    lower_bounds=-4,
    upper_bounds=4,
    max_iters=max_iters,
    type="igd",
    verbose=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)

ax0.plot(X[:, 0], X[:, 1], "g*")
ax0.plot(x0[:, 0], x0[:, 1], "g.", ms=8, clip_on=False)
ax0.set_xlim([-2, 2])
ax0.set_ylim([-2, 2])
ax0.set_title("Decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

ax1.plot(ref[:, 0], ref[:, 1], "k.", alpha=0.4)
ax0.plot(ref_x[:, 0], ref_x[:, 1], "k.", alpha=0.4)

n_per_axis = 65
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
Z = np.array([F(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z1 = Z[:, 0].reshape(-1, len(x))
Z2 = Z[:, 1].reshape(-1, len(x))
CS1 = ax0.contour(X1, X2, Z1, 15, cmap=plt.cm.gray, linewidths=0.8, alpha=0.6)
CS2 = ax0.contour(X1, X2, Z2, 15, cmap=plt.cm.gray, linewidths=0.8, linestyles="--", alpha=0.6)

if 1 < 2:
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
            alpha=0.5,
            headlength=4.7,
            headwidth=2.7,
        )
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
            width=0.003,
            alpha=0.3,
            headlength=4.7,
            headwidth=2.7,
        )

# x_vals = np.array([0, 6])
# y_vals = 6 - x_vals
ax1.plot(Y[:, 0], Y[:, 1], "g*")
ax1.plot(y0[:, 0], y0[:, 1], "g.", ms=8)
# ax1.plot(x_vals, y_vals, "r--")
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax22 = ax2.twinx()
ax2.semilogy(range(1, len(opt.hist_GD) + 1), opt.hist_GD, "b-", label="GD")
ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax22.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
ax2.legend()

plt.savefig(f"2D-example_concave-{mu}.pdf", dpi=1000)
