import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
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


def kernel(x, y):
    return jnp.exp(-jnp.sum((x - y) ** 2))


N = 30
p = np.linspace(1, 6, N)
Y = X = np.c_[p, 7 - p]

p = np.linspace(0, 6, 50)
ref = np.c_[p, 6 - p]
dim = Y.shape[1]


def U(theta: float) -> float:
    mmd = MMD(ref=ref, theta=theta)
    return np.max(-1 * mmd.compute_gradient(Y))


theta = np.logspace(-4, 1, 100)
v = np.array([U(t) for t in theta])
idx = np.argmin(v)
theta_ = theta[idx]

mmd = MMD(ref=ref, theta=theta_)
grad = -1 * mmd.compute_gradient(Y)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.5))
plt.subplots_adjust(right=0.95, left=0.15)

ax0.plot(theta, v, "k--")
ax0.plot(theta, v, "k.")
ax0.set_xscale("log")
ax0.set_xlabel(r"$\theta$")
ax0.set_ylabel(r"$\max_{l, i} \partial \textrm{MMD} /\partial y^l_i$")

ax1.set_aspect("equal")
ax1.plot(Y[:, 0], Y[:, 1], "g.")
ax1.plot(ref[:, 0], ref[:, 1], "r+", ms=8, clip_on=False, alpha=0.4)
ax1.set_title(rf"$\theta^* = {theta_}$")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
grad /= np.sqrt(np.sum(grad**2, axis=1)).reshape(-1, 1)
for i in range(N):
    ax1.quiver(
        Y[i, 0],
        Y[i, 1],
        grad[i, 0],
        grad[i, 1],
        # newton[i, 0],
        # newton[i, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="k",
        width=0.005,
        alpha=0.7,
        headlength=4.7,
        headwidth=2.7,
    )
plt.show()
