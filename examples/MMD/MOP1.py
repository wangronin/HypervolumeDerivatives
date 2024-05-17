import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

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

from hvd.mmd import MMD

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


N = 15
p = np.linspace(-1, 0.2, N)
X = np.c_[p, p + 1]
Y = np.array([MOP1(_) for _ in X])

p = np.r_[np.linspace(-1, 0, 33), np.linspace(0, 1, 17)]
ref_X = np.c_[p, p]
ref = np.array([MOP1(_) for _ in ref_X])
PF = ref.copy()
ref -= 0.4 * np.ones(2)
dim = Y.shape[1]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6.5))
plt.subplots_adjust(right=0.97, left=0.1)

ax0.plot(ref_X[:, 0], ref_X[:, 1], "g*")
ax0.plot(X[:, 0], X[:, 1], "g.", ms=8, clip_on=False)
# ax0.set_xlim([-2, 2])
# ax0.set_ylim([-2, 2])
ax0.set_title("Decision space")
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

ax1.plot(ref[:, 0], ref[:, 1], "r*")
ax1.plot(PF[:, 0], PF[:, 1], "g.")
ax1.plot(Y[:, 0], Y[:, 1], "g.", ms=8)
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")


mmd = MMD(2, 2, ref=ref, func=MOP1, jac=MOP1_Jacobian, theta=0.5)
theta = np.logspace(-4, 1, 100)

for i in range(30):
    grad = mmd.compute_gradient(X)["MMDdX"]
    grad /= np.sqrt(np.sum(grad**2, axis=1)).reshape(-1, 1)
    step_size = 0.1 * np.exp(-0.1 * i) * np.ones((N, 1))
    # for i in range(N):
    #     X_ = X.copy()
    #     for _ in range(5):
    #         X_[i] -= step_size[i] * grad[i]
    #         f0 = mmd.compute(np.array([MOP1(_) for _ in X]))
    #         f1 = mmd.compute(np.array([MOP1(_) for _ in X_]))
    #         # Armijo–Goldstein condition
    #         # when R norm is close to machine precision, it makes no sense to perform the line search
    #         success = f0 - f1 >= 1e-3 * step_size[i] * np.inner(grad[i], grad[i])
    #         if success:
    #             break
    #         else:
    #             step_size[i] *= 0.5
    print(step_size.ravel())
    step = -step_size * grad
    for i in range(N):
        ax0.quiver(
            X[i, 0],
            X[i, 1],
            step[i, 0],
            step[i, 1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color="k",
            width=0.005,
            alpha=0.5,
            headlength=3,
            headwidth=1.7,
        )
    X += step
    Y_ = np.array([MOP1(x) for x in X])

    for i in range(N):
        ax1.quiver(
            Y[i, 0],
            Y[i, 1],
            Y_[i, 0] - Y[i, 0],
            Y_[i, 1] - Y[i, 1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color="k",
            width=0.005,
            alpha=0.5,
            headlength=4.7,
            headwidth=2.7,
        )
    Y = Y_

plt.show()
