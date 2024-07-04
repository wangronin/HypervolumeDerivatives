import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.linalg import solve

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
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


from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd import MMD, rational_quadratic, rbf
from hvd.utils import precondition_hessian

np.random.seed(42)


def MOP1(x):
    x = np.array(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(2), 2 * np.eye(2)])


best_from_angel = pd.read_csv("MOP1_n=30.csv", header=None).values
# generate the reference set
mu = len(best_from_angel)
p = np.linspace(-1, 1, mu)
ref_X = np.c_[p, p]
ref = np.array([MOP1(_) for _ in ref_X])
# the reference set
# ref = ref - 0.3 * np.ones(2)
# the initial population
p = np.linspace(-1, 0.5, mu)
X = np.c_[p, p + 0.5]
Y = np.array([MOP1(_) for _ in X])
dim = Y.shape[1]

theta = 1 / mu
kernel = rational_quadratic
mmd = MMD(2, 2, ref=ref, func=MOP1, jac=MOP1_Jacobian, kernel=kernel, theta=theta)
# generate a fine grained Pareto front for measuring the final metrics
p = np.linspace(-1, 1, 500)
pareto_set = np.c_[p, p]
pareto_front = np.array([MOP1(_) for _ in pareto_set])
# performance indicator
mmd_metric = MMD(2, 2, ref=pareto_front, func=MOP1, jac=MOP1_Jacobian, kernel=kernel, theta=theta)
gd_metric = GenerationalDistance(ref=pareto_front)
igd_metric = InvertedGenerationalDistance(ref=pareto_front, cluster_matching=False)
angel_best = mmd_metric.compute(best_from_angel)
# performance metrics
hist_value = [mmd_metric.compute(Y)]
hist_deltap = [igd_metric.compute(Y=Y)]
hist_norm = []
max_iters = 5

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5.5))
plt.subplots_adjust(right=0.95, left=0.05)

ax0_handles = []
ax0_handles += ax0.plot(pareto_set[:, 0], pareto_set[:, 1], "m-", alpha=0.5)
ax0_handles += ax0.plot(X[:, 0], X[:, 1], "g.", ms=8, clip_on=False)
ax0.set_title("decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

n_per_axis = 30
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
Z = np.array([MOP1(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z1 = Z[:, 0].reshape(-1, len(x))
Z2 = Z[:, 1].reshape(-1, len(x))
CS1 = ax0.contour(X1, X2, Z1, 10, cmap=plt.cm.gray, linewidths=0.8, alpha=0.6)
CS2 = ax0.contour(X1, X2, Z2, 10, cmap=plt.cm.gray, linewidths=0.8, linestyles="--", alpha=0.6)

ax1_handles = []
ax1_handles += ax1.plot(ref[:, 0], ref[:, 1], "r+")
ax1_handles += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "m-", alpha=0.5)
ax1_handles += ax1.plot(Y[:, 0], Y[:, 1], "g.", ms=5)
ax1.set_title("objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

for i in range(max_iters):
    out = mmd.compute_hessian(X)
    H_, g = out["MMDdX2"], out["MMDdX"]
    H = np.zeros((2 * mu, 2 * mu))
    # H = H_
    for i in range(mu):
        H[i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2] = H_[i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2]
    # H = precondition_hessian(H)
    # breakpoint()
    g_ = g.reshape(-1, 1)
    hist_norm.append(np.linalg.norm(g_))
    newton_step = 1 * solve(H, g_).reshape(mu, -1)
    step_size = np.ones((mu, 1))
    # Armijo–Goldstein backtracking line search
    for i in range(mu):
        X_ = X.copy()
        for _ in range(5):
            X_[i] += step_size[i] * newton_step[i]
            f0 = mmd.compute(np.array([MOP1(_) for _ in X]))
            f1 = mmd.compute(np.array([MOP1(_) for _ in X_]))
            # when R norm is close to machine precision, it makes no sense to perform the line search
            g_ = mmd.compute_gradient(X_)["MMDdX"][i]
            armijo = f0 - f1 >= 1e-4 * step_size[i] * np.inner(g[i], newton_step[i])
            # curvature = np.abs(np.inner(g_, newton_step[i])) <= np.abs(0.9 * np.inner(g[i], newton_step[i]))
            if armijo:
                break
            else:
                step_size[i] *= 0.5

    print(step_size.ravel())
    for i in range(mu):
        ax0.quiver(
            X[i, 0],
            X[i, 1],
            newton_step[i, 0],
            newton_step[i, 1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color="k",
            width=0.005,
            alpha=0.5,
            headlength=3,
            headwidth=1.7,
        )
    X += step_size * newton_step
    Y_ = np.array([MOP1(x) for x in X])

    # for i in range(mu):
    #     ax1.quiver(
    #         Y[i, 0],
    #         Y[i, 1],
    #         Y_[i, 0] - Y[i, 0],
    #         Y_[i, 1] - Y[i, 1],
    #         scale_units="xy",
    #         angles="xy",
    #         scale=1,
    #         color="k",
    #         width=0.005,
    #         alpha=0.5,
    #         headlength=4.7,
    #         headwidth=2.7,
    #     )
    Y = Y_
    hist_value.append(mmd_metric.compute(Y))
    hist_deltap.append(igd_metric.compute(Y=Y))

ax0_handles += ax0.plot(X[:, 0], X[:, 1], "kx", ms=5)
ax1_handles += ax1.plot(Y[:, 0], Y[:, 1], "kx", ms=5)
ax1_handles += ax1.plot(best_from_angel[:, 0], best_from_angel[:, 1], "r.", ms=5)
ax0.legend(ax0_handles, ["efficient set", r"$Y_0$", r"$Y_{\text{final}}$"])
ax1.legend(ax1_handles, ["reference set", "Pareto front", r"$Y_0$", r"$Y_{\text{final}}$", "Angel's set"])

ax2.semilogy(range(1, len(hist_value) + 1), hist_value, "b-")
ax2.semilogy(range(1, len(hist_norm) + 1), hist_norm, "r--")
ax2.hlines(angel_best, 1, len(hist_norm) + 1, colors="k", linestyles="dashed")
ax2.set_title(rf"{kernel.__name__}, $\sigma^2 = {0.5/theta}$")
ax2.set_xlabel("iteration")
ax2.legend(["MMD", r"$||\nabla \text{MMD}||_2$", "Angel's best set"])
ax2.set_xticks(range(1, max_iters + 1))
plt.savefig(f"MOP1_Newton_{kernel.__name__}_sigma2={0.5/theta:.2f}_{mu}points.pdf")

Y_ = Y[Y[:, 0].argsort()]
Y_ = np.concatenate([best_from_angel[0:-2], Y_[[0]], Y_[[-1]]], axis=0)
value = mmd_metric.compute(Y)
print(f"MMD (Angel): {angel_best}")
print(f"MMD (final): {hist_value[-1]}")
print(f"MMD (Angel's with extremel points replaced): {value}")
