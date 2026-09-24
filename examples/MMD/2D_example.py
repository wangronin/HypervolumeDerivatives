import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd import MMD
from hvd.mmd.kernels import RBF
from hvd.mmd_newton import MMDN
from hvd.reference_set import ReferenceSet

plt.style.use("ggplot")
rcParams["font.size"] = 13
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 13
rcParams["ytick.labelsize"] = 13
rcParams["xtick.major.size"] = 5
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 5
rcParams["ytick.major.width"] = 1

np.random.seed(66)


def MOP1(x):
    x = np.asarray(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.asarray(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    return np.array([2 * np.eye(2), 2 * np.eye(2)])


def h(x):
    x = np.asarray(x)
    return np.sum(x**2) - 1


def h_Jacobian(x):
    return 2 * np.asarray(x)


def h_Hessian(x):
    return 2 * np.eye(2)


max_iters = 30
mu = 20

# Same linear initialization as examples/HVN/2D_example.py.
p = np.linspace(0, 2, mu)
x0 = np.c_[p, p - 2]
y0 = np.array([MOP1(x) for x in x0])

# On the unit circle, f1 + f2 = 6 and f1 ranges over this interval.
f1_min, f1_max = 3 - 2 * np.sqrt(2), 3 + 2 * np.sqrt(2)
pareto_f1 = np.linspace(f1_min, f1_max, 1_000)
pareto_front = np.c_[pareto_f1, 6 - pareto_f1]

# Concentrate the reference points near both endpoints. The optimized points can
# then demonstrate MMD's tendency to diversify a non-uniform target set.
reference_parameter = 1 / (1 + np.exp(-np.linspace(-4, 4, mu)))
reference_parameter = (reference_parameter - reference_parameter[0]) / (
    reference_parameter[-1] - reference_parameter[0]
)
reference_f1 = f1_min + (f1_max - f1_min) * reference_parameter
reference = np.c_[reference_f1, 6 - reference_f1]
shift_direction = {0: -np.ones(2) / np.sqrt(2)}
metrics = {
    "GD": GenerationalDistance(ref=pareto_front),
    "IGD": InvertedGenerationalDistance(ref=pareto_front),
}

indicator = MMD(
    n_var=2,
    n_obj=2,
    ref=ReferenceSet(reference, eta=shift_direction),
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    kernel=RBF(theta=5.0),
)
optimizer = MMDN(
    n_var=2,
    n_obj=2,
    indicator=indicator,
    func=MOP1,
    jac=MOP1_Jacobian,
    h=h,
    h_jac=h_Jacobian,
    h_hessian=h_Hessian,
    N=mu,
    X0=x0,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    verbose=True,
    regularization=True,
    metrics=metrics,
)
X, Y, _ = optimizer.run()

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(right=0.93, left=0.05)

circle = plt.Circle((0, 0), 1, color="r", fill=False, ls="--", lw=1.5)
ax0.plot(X[:, 0], X[:, 1], "g*", label=r"$X_{\mathrm{MMD}}$")
ax0.plot(x0[:, 0], x0[:, 1], "g.", ms=8, clip_on=False, label=r"$X_0$")
ax0.add_patch(circle)
ax0.set_xlim([-2, 2])
ax0.set_ylim([-2, 2])
ax0.set_aspect("equal")
ax0.set_title("Decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

n_per_axis = 30
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
Z = np.array([MOP1(point) for point in np.c_[X1.ravel(), X2.ravel()]])
Z1 = Z[:, 0].reshape(X1.shape)
Z2 = Z[:, 1].reshape(X2.shape)
ax0.contour(X1, X2, Z1, 10, cmap=plt.cm.Blues, linewidths=0.8, alpha=0.6)
ax0.contour(X1, X2, Z2, 10, cmap=plt.cm.Reds, linewidths=0.8, alpha=0.6)

if 1 < 2:
    trajectory = np.array([x0] + optimizer.history_X)
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
ax0.legend()

ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "r--", label="Pareto front")
ax1.plot(y0[:, 0], y0[:, 1], "g.", ms=10, label=r"$Y_0$")
ax1.plot(
    optimizer.indicator.ref.reference_set[:, 0],
    optimizer.indicator.ref.reference_set[:, 1],
    "^",
    color="tab:orange",
    ms=7,
    label="reference",
)
ax1.plot(Y[:, 0], Y[:, 1], "g*", label=r"$Y_{\mathrm{MMD}}$")

if 1 < 2:
    trajectory = np.array([y0] + optimizer.history_Y)
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
            alpha=0.5,
            headlength=4.7,
            headwidth=2.7,
        )
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax1.legend()

iterations = range(1, len(optimizer.history_indicator_value) + 1)
averaged_hausdorff = np.maximum(optimizer.history_metrics["GD"], optimizer.history_metrics["IGD"])
ax22 = ax2.twinx()
ax2.semilogy(iterations, averaged_hausdorff, "b-")
ax22.semilogy(iterations, optimizer.history_R_norm, "g--")
ax2.set_ylabel("Averaged Hausdorff distance", color="b")
ax22.set_ylabel(r"$\|R(\mathbf{X})\|$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))

plt.tight_layout()
plt.savefig(f"MMD-2D-example-{mu}.pdf", dpi=1000)
