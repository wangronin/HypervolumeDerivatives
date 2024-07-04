import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.newton import HVN

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


def convex(x):
    return np.array(x)


def convex_Jacobian(x):
    return np.eye(2)


def convex_Hessian(x):
    return np.zeros((2, 2))


def g(x):
    v = x[0] ** 2 - 2 * x[0] + 1
    if v >= 1e-4:
        return 0
    else:
        return v


def g_Jacobian(x):
    v = x[0] ** 2 - 2 * x[0] + 1
    if v >= 1e-4:
        return np.array([0, 0])
    else:
        return np.r_[2 * x[0] - 2, 0]


def g_Hessian(x):
    v = x[0] ** 2 - 2 * x[0] + 1
    if v >= 1e-4:
        return np.zeros((2, 2))
    else:
        h = np.zeros((2, 2))
        h[0, 0] = 2
        return h


ref = np.array([1, 1])
max_iters = 3

N = 10
x1 = np.linspace(0, 1, N)
x2 = x1**2 - 2 * x1 + 1
x0 = np.c_[x1, x2]
y0 = x0.copy()

opt = HVN(
    dim=2,
    n_objective=2,
    ref=ref,
    func=convex,
    jac=convex_Jacobian,
    hessian=convex_Hessian,
    h=g,
    h_jac=g_Jacobian,
    h_hessian=g_Hessian,
    mu=len(x0),
    x0=x0,
    lower_bounds=0,
    upper_bounds=1,
    minimization=True,
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ciricle = plt.Circle((0, 0), 1, color="r", fill=False, ls="--", lw=1.5)

ax0.plot(Y[:, 0], Y[:, 1], "g*")
ax0.plot(y0[:, 0], y0[:, 1], "g.", ms=8)
trajectory = np.array([y0] + opt.hist_Y)

for i in range(N):
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

x_vals = np.linspace(0, 1, 100)
y_vals = x_vals**2 - 2 * x_vals + 1
ax0.plot(x_vals, y_vals, "r--")
ax0.set_title("Objective space")
ax0.set_xlabel(r"$f_1$")
ax0.set_ylabel(r"$f_2$")

ax22 = ax1.twinx()
ax1.plot(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "b-")
# ax22.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_G_norm, "g--")
ax1.set_ylabel("HV", color="b")
ax22.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax1.set_title("Performance")
ax1.set_xlabel("iteration")
ax1.set_xticks(range(1, max_iters + 1))

plt.savefig(f"convex-example-{N}.pdf", dpi=1000)

# df = pd.DataFrame(dict(iteration=range(1, len(opt.hist_HV) + 1), HV=opt.hist_HV, G_norm=opt.hist_G_norm))
# df.to_latex(buf=f"2D-example-{mu}.tex", index=False)
