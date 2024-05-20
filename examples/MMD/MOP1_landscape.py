import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

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


from hvd.mmd import MMD, rational_quadratic, rbf

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


# the index of the point that will vary
index = 15


def func(x):
    y = MOP1(x)
    Y[index] = y
    return mmd.compute(Y)


def func2(x):
    X[index] = x
    return mmd.compute_gradient(X)["MMDdX"][index] * 20


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
# X = np.c_[p, p]
Y = np.array([MOP1(_) for _ in X])
dim = Y.shape[1]

theta = 1 / mu
kernel = rbf
mmd = MMD(2, 2, ref=ref, func=MOP1, jac=MOP1_Jacobian, kernel=kernel, theta=theta)
g = mmd.compute_gradient(X)["MMDdX"][index]
# generate a fine grained Pareto front for measuring the final metrics
p = np.linspace(-1, 1, 500)
pareto_set = np.c_[p, p]
pareto_front = np.array([MOP1(_) for _ in pareto_set])

fig, ax0 = plt.subplots(1, 1, figsize=(8, 8))
plt.subplots_adjust(right=0.96, left=0.1)

ax0_handles = []
ax0_handles += ax0.plot(pareto_set[:, 0], pareto_set[:, 1], "m-", alpha=0.5)
ax0_handles += ax0.plot(X[:, 0], X[:, 1], "g.", ms=5, clip_on=False)
ax0.plot(X[index, 0], X[index, 1], "r.", ms=5, clip_on=False)
ax0.quiver(
    X[index, 0],
    X[index, 1],
    g[0],
    g[1],
    scale_units="xy",
    angles="xy",
    scale=1,
    color="k",
    width=0.005,
    alpha=0.5,
    headlength=3,
    headwidth=1.7,
)
ax0.set_title("w/o reference set shifting")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

n_per_axis = 40
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
# plot the contour line of MMD
Z = np.array([func(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z = Z.reshape(-1, len(x))
CS = ax0.contour(X1, X2, Z, 30, cmap=plt.cm.jet, linewidths=0.8, alpha=0.6)
fig.colorbar(CS)
ax0.clabel(CS, inline=True, fontsize=10)
# plot the gradient field of MMD
# G = np.array([func2(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
# U = G[:, 0].reshape(-1, len(x))
# V = G[:, 1].reshape(-1, len(x))
# Q = ax0.quiver(X1, X2, U, V, scale=1, color="k", width=0.005, alpha=0.5, headlength=2, headwidth=1.7)

ax0.legend(ax0_handles, ["efficient set", r"$Y_0$"])
plt.savefig(f"MOP1_Newton_{kernel.__name__}_sigma2={0.5/theta:.2f}_landscape_{index}_no_shifting.pdf")
