import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

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


def MOP1(x):
    x = np.array(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(2), 2 * np.eye(2)])


def h(x):
    x = np.array(x)
    return np.sum(x**2) - 1


def h_Jacobian(x):
    x = np.array(x)
    return 2 * x


theta = np.linspace(-np.pi * 3 / 4, np.pi / 4, 100)
ref_x = np.array([[np.cos(a) * 0.99, np.sin(a) * 0.99] for a in theta])
ref = np.array([MOP1(_) for _ in ref_x])

mu = 10


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

y0 = np.array([MOP1(_) for _ in x0])

igd = InvertedGenerationalDistance(ref, MOP1, MOP1_Jacobian, MOP1_Hessian)
igd._clustering_and_matching(y0)
medroids = igd._medroids

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.subplots_adjust(right=0.93, left=0.05)

x_vals = np.array([0, 6])
y_vals = 6 - x_vals
# ax1.plot(Y[:, 0], Y[:, 1], "g*")
ax.plot(y0[:, 0], y0[:, 1], "g.", ms=8)
ax.plot(medroids[:, 0], medroids[:, 1], "r*", ms=8)
ax.plot(ref[:, 0], ref[:, 1], "g.", ms=5, alpha=0.3)
# ax.plot(x_vals, y_vals, "k--")

for i in range(mu):
    t = medroids[igd._medoids_idx[i]]
    ax.quiver(
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
ax.set_title("Objective space")
ax.set_xlabel(r"$f_1$")
ax.set_ylabel(r"$f_2$")

plt.show()
# plt.savefig(f"2D-example-{mu}.pdf", dpi=1000)
