import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
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


from hvd.mmd import MMD, rbf

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


N = 5
# p = np.linspace(-1, 0, N)
p = 2 / (1 + np.exp(-np.linspace(-3, 3, N))) - 1
ref_X = np.c_[p, p]
ref_unshifted = np.array([MOP1(_) for _ in ref_X])
# the reference set
ref = ref_unshifted - 0.5 * np.ones(2)
theta = 1
kernel = rbf
mmd = MMD(2, 2, ref=ref, func=MOP1, kernel=kernel, theta=theta)
# generate a fine grained Pareto front for measuring the final metrics
p = np.linspace(-1, 1, 500)
pareto_set = np.c_[p, p]
pareto_front = np.array([MOP1(_) for _ in pareto_set])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(right=0.96, left=0.1)

ax0.set_aspect("equal")
ax0_handles = []
ax0_handles += ax0.plot(pareto_set[:, 0], pareto_set[:, 1], "m-", alpha=0.5)
ax0.set_title(rf"MMD with $\theta={theta}$")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")

n_per_axis = 60
x = np.linspace(-2, 2, n_per_axis)
X1, X2 = np.meshgrid(x, x)
# plot the contour line of MMD
Y = np.array([MOP1(x) for x in np.array([X1.flatten(), X2.flatten()]).T])
Z = np.array([mmd.compute(Y=y.reshape(1, -1)) for y in Y])
Z = Z.reshape(-1, len(x))
CS = ax0.contour(X1, X2, Z, 30, cmap=plt.cm.jet, linewidths=0.8, alpha=0.6)
fig.colorbar(CS)
ax0.clabel(CS, inline=True, fontsize=10)
ax0.legend(ax0_handles, ["efficient set"])

p = np.linspace(-1, 1, 1000)
X = np.c_[p, p]
Y = np.array([MOP1(x) for x in X])
v = np.array([mmd.compute(Y=y.reshape(1, -1)) for y in Y])
ax1.plot(p, v, "k-")
ax1.set_title("Along the Pareto set")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel("MMD")

plt.tight_layout()
plt.savefig(f"MOP1_{kernel.__name__}_theta={theta:.2f}_landscape.pdf")
