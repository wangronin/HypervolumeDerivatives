import autograd.numpy as npa
import matplotlib.pyplot as plt
import numpy as np
from autograd import hessian, jacobian
from matplotlib import rcParams
from scipy.linalg import cho_solve, cholesky

from hvd.delta_p import InvertedGenerationalDistance

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
    x = npa.array(x)
    offset = npa.array([1] * 2)
    return npa.array([1 - npa.exp(-npa.sum((x - offset) ** 2)), 1 - npa.exp(-npa.sum((x + offset) ** 2))])


Jacobian = jacobian(F)
Hessian = hessian(F)


def _precondition_hessian(H: np.ndarray) -> np.ndarray:
    """Precondition the Hessian matrix to make sure it is positive definite

    Args:
        H (np.ndarray): the Hessian matrix

    Returns:
        np.ndarray: the preconditioned Hessian
    """
    # pre-condition the Hessian
    beta = 1e-6
    v = np.min(np.diag(H))
    tau = 0 if v > 0 else -v + beta
    I = np.eye(H.shape[0])
    for _ in range(35):
        try:
            L = cholesky(H + tau * I, lower=True)
            break
        except:
            tau = max(2 * tau, beta)
    return L


# def MOP1(x):
#     x = np.array(x)
#     return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


# def MOP1_Jacobian(x):
#     x = np.array(x)
#     return np.array([2 * (x - 1), 2 * (x + 1)])


# def MOP1_Hessian(x):
#     x = np.array(x)
#     return np.array([2 * np.eye(2), 2 * np.eye(2)])


# def h(x):
#     x = np.array(x)
#     return np.sum(x**2) - 1


# def h_Jacobian(x):
#     x = np.array(x)
#     return 2 * x


# theta = np.linspace(-np.pi * 3 / 4, np.pi / 4, 100)
# ref_x = np.array([[np.cos(a) * 0.99, np.sin(a) * 0.99] for a in theta])
# ref = np.array([MOP1(_) for _ in ref_x])
p = np.linspace(-1, 1, 100)
ref_x = np.c_[p, p]
ref = np.array([F(_) for _ in ref_x])
mu = 3


# different initializations of the first population
if 1 < 2:
    # option1: linearly spacing
    p = np.linspace(0, 2, mu)
    x0 = np.c_[p, p - 2]
elif 11 < 2:
    # option2: logistic spacing/denser on two tails
    p = 2 / (1 + np.exp(-np.linspace(-3, 3, mu)))
    x0 = np.c_[p, p - 2]
elif 11 < 2:
    # option3: logit spacing/denser in the middle
    p = np.log(1 / (1 - np.linspace(0.09485175, 1.90514825, mu) / 2) - 1)
    p = 2 * (p - np.min(p)) / (np.max(p) - np.min(p))
    x0 = np.c_[p, p - 2]
elif 11 < 2:
    # option4: spherical initialization
    theta = np.linspace(-np.pi * 5 / 8, np.pi / 8, 50)
    x0 = np.array([[2 * np.cos(a), 2 * np.sin(a)] for a in theta])

x0 = np.linspace(-0.2, 0.2, mu)
x0 = np.c_[x0, x0] + np.array([0.5, -0.5])
y0 = np.array([F(_) for _ in x0])

igd = InvertedGenerationalDistance(ref, F, Jacobian, Hessian, cluster_matching=True)
igd._match(y0)
medroids = igd._medroids
medroids_x = ref_x[igd._idx]
medroids_x = medroids_x[igd._medoids_idx]

grad, Hessian = igd.compute_derivatives(x0, Y=y0)
L = [_precondition_hessian(H) for H in Hessian]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 6))
plt.subplots_adjust(right=0.93, left=0.05)

ax0.plot(x0[:, 0], x0[:, 1], "g.", ms=8)
ax0.plot(medroids_x[:, 0], medroids_x[:, 1], "r*", ms=8)
ax0.plot(ref_x[:, 0], ref_x[:, 1], "g.", ms=5, alpha=0.3)

ax1.plot(y0[:, 0], y0[:, 1], "g.", ms=8)
ax1.plot(medroids[:, 0], medroids[:, 1], "r*", ms=8)
ax1.plot(ref[:, 0], ref[:, 1], "g.", ms=5, alpha=0.3)

for i in range(mu):
    t = medroids_x[igd._medoids_idx[i]]
    ax0.quiver(
        x0[i, 0],
        x0[i, 1],
        t[0] - x0[i, 0],
        t[1] - x0[i, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="k",
        width=0.005,
        alpha=0.5,
        headlength=4.7,
        headwidth=2.7,
    )

for i in range(mu):
    # diff = -1 * solve(Hessian[i], grad[i].reshape(-1, 1)).ravel()
    diff = -1 * cho_solve((L[i], True), grad[i].reshape(-1, 1)).ravel()
    ax0.quiver(
        x0[i, 0],
        x0[i, 1],
        diff[0],
        diff[1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="r",
        width=0.005,
        alpha=0.5,
        headlength=4.7,
        headwidth=2.7,
    )

for i in range(mu):
    t = medroids[igd._medoids_idx[i]]
    ax1.quiver(
        y0[i, 0],
        y0[i, 1],
        t[0] - y0[i, 0],
        t[1] - y0[i, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="k",
        width=0.005,
        alpha=0.5,
        headlength=4.7,
        headwidth=2.7,
    )

ax0.set_title("Decision space")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")
ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

plt.tight_layout()
plt.show()
# plt.savefig(f"2D-example-{mu}.pdf", dpi=1000)
