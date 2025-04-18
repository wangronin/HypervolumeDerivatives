import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import integrate
from scipy.stats import norm

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
from hvd.mmd import MMD
from hvd.mmd_newton import MMDNewton
from hvd.newton import DpN
from hvd.reference_set import ReferenceSet

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


N = 4
# p = norm.ppf(np.linspace(0.05, 0.95, 30), scale=1 / 2)
p = np.linspace(-1, -0.2, N)
# p = 2 / (1 + np.exp(-np.linspace(-3, 3, N))) - 1
# p = np.r_[np.linspace(-1, 0, 5), np.linspace(0, 1, 25)]
ref_X = np.c_[p, p]
ref_unshifted = np.array([MOP1(_) for _ in ref_X])
# the reference set
ref = ref_unshifted - 0.5 * np.ones(2)
# the initial population
# p = np.linspace(-1, 1, N)
p = -0.4
X = np.c_[p, p]
Y = np.array([MOP1(_) for _ in X])
dim = Y.shape[1]
# theta = 1 / (10 * N)
theta = 0.01

# generate a fine grained Pareto front for measuring the final metrics
p = np.linspace(-1, 1, 500)
pareto_set = np.c_[p, p]
pareto_front = np.array([MOP1(_) for _ in pareto_set])
# performance indicator
max_iters = 10
metrics = dict(
    GD=GenerationalDistance(pareto_front),
    IGD=InvertedGenerationalDistance(pareto_front),
    MMD=MMD(2, 2, ref=ref, func=MOP1, theta=theta),
)
mmd = MMD(2, 2, ref=ref, func=MOP1, theta=theta)
opt = MMDNewton(
    n_var=2,
    n_obj=2,
    ref=ReferenceSet(ref=ref, eta=np.zeros(2)),
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    N=len(X),
    X0=X,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    verbose=True,
    metrics=metrics,
    preconditioning=False,
    theta=theta,
)
X_opt, Y_opt, _ = opt.run()

# compare to DpN
opt_dpn = DpN(
    dim=2,
    n_obj=2,
    ref=ReferenceSet(ref=ref, eta=np.zeros(2)),
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    N=len(X),
    x0=X,
    xl=-2,
    xu=2,
    max_iters=max_iters,
    verbose=True,
    type="igd",
    metrics=metrics,
)
X_DpN, Y_DpN, _ = opt_dpn.run()

pd.DataFrame(ref).to_csv("MMD_MOP1_ref.csv", index=False)
pd.DataFrame(Y).to_csv("MMD_MOP1_Y0.csv", index=False)
pd.DataFrame(Y_opt).to_csv(f"MMD_MOP1_Y_MMD_theta{theta}.csv", index=False)
pd.DataFrame(Y_DpN).to_csv(f"MMD_MOP1_Y_DpN.csv", index=False)

slope_func = lambda f1: -1 / (((f1 / 2) ** 0.5 - 2) * (f1 / 2) ** (-0.5))
c1 = 0.5
slope = slope_func(Y_opt[0, 0])
k1 = Y_opt[0] + np.array([c1, slope * c1])
k2 = Y_opt[0] - np.array([c1, slope * c1])
c2 = 1.5
k3 = Y_opt[0] + np.array([c2, -c2 / slope])
k4 = Y_opt[0] - np.array([c2, -c2 / slope])
d = (np.sum(ref - Y_opt, axis=0) + Y_opt)[0]


r = ref - Y_opt
func = lambda w1, w2: np.sum(np.sin(np.sqrt(2 * theta) * r @ np.array([[w1], [w2]])))
integrad1 = lambda w1, w2: w1 * func(w1, w2) * norm.pdf(w1) * norm.pdf(w2)
integrad2 = lambda w1, w2: w2 * func(w1, w2) * norm.pdf(w1) * norm.pdf(w2)
A = integrate.nquad(integrad1, [(-np.inf, np.inf), (-np.inf, np.inf)])[0]
B = integrate.nquad(integrad2, [(-np.inf, np.inf), (-np.inf, np.inf)])[0]
print(-A / B)
print(-1 / slope)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6))
plt.subplots_adjust(right=0.95, left=0.05)

ax0_handles = []
ax0_handles += ax0.plot(pareto_set[:, 0], pareto_set[:, 1], "m-", alpha=0.5)
ax0_handles += ax0.plot(X[:, 0], X[:, 1], "g.", ms=8, clip_on=False)
ax0_handles += ax0.plot(X_opt[:, 0], X_opt[:, 1], "kx", ms=5)
ax0_handles += ax0.plot(X_DpN[:, 0], X_DpN[:, 1], "r+", ms=5)
ax0.set_aspect("equal")
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

ax1_handles = []
ax1_handles += ax1.plot(ref[:, 0], ref[:, 1], "k.", ms=5)
ax1_handles += ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "m-", alpha=0.5)
ax1_handles += ax1.plot(Y[:, 0], Y[:, 1], "g.", ms=5)
ax1_handles += ax1.plot(Y_opt[:, 0], Y_opt[:, 1], "kx", ms=5)
ax1_handles += ax1.plot(Y_DpN[:, 0], Y_DpN[:, 1], "r+", ms=5)
ax1_handles += ax1.plot((k1[0], k2[0]), (k1[1], k2[1]), "g--", lw=1.3)
ax1_handles += ax1.plot((k3[0], k4[0]), (k3[1], k4[1]), "k--", lw=1.3)
ax1_handles += ax1.plot((Y_opt[0, 0], d[0]), (Y_opt[0, 1], d[1]), "r--", lw=1.3)

ax1.set_aspect("equal")
ax1.set_title(rf"$\theta={theta}$")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax0.legend(ax0_handles, ["efficient set", r"$X_0$", r"$X_{\text{MMD}}$", r"$X_{\text{DpN}}$"])
ax1.legend(
    ax1_handles,
    [
        "reference set",
        "Pareto front",
        r"$Y_0$",
        r"$Y_{\text{MMD}}$",
        r"$Y_{\text{DpN}}$",
        "Normal space",
        "Tangent space",
        r"$\sum_i \vec{r}^{\,i} - \vec{y}$",
    ],
    fontsize=12,
)

plt.tight_layout()
plt.savefig(f"MMD-MOP1_theta{theta}.pdf", dpi=1000)
