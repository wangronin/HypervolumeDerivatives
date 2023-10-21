import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from hvd.newton import DpN
from hvd.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, PymooProblemWithAD

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

np.random.seed(42)

# NOTE: issues found so far on ZDTs
# ZDT2: some points converges to the wrong decision boundary; the convergence rate is quadratic still.
# ZDT3: stagnation in local efficient points, which is natural on this problem
# ZDT6: indefinite DpN Hessians; Hessian modification method is needed

N = 10
max_iters = 5
problem = PymooProblemWithAD(ZDT6(n_var=2))
pareto_front = problem.get_pareto_front(500)
pareto_set = problem.get_pareto_set(500)

ref = problem.get_pareto_front(100)
ref -= 0.02
# get the re-image of the reference set
# f_x2 = (
#     lambda x1, c: 0.111111111111111 * c
#     + 0.0555555555555556 * x1
#     + 0.111111111111111 * np.sqrt(c * x1 + 0.25 * x1**2)
#     - 0.111111111111111
# )
# x1 = ref[:, 0]
# x2 = [f_x2(*_) for _ in zip(x1, ref[:, 1])]
# ref_x = np.c_[x1, x2]
# the initial approximation set
x0 = problem.get_pareto_set(N, kind="uniform")
x0[:, 1:] += 0.01 * np.random.rand(N, problem.n_var - 1)
y0 = np.array([problem.objective(x) for x in x0])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax1.plot(y0[:, 0], y0[:, 1], "r.", ms=7, alpha=0.5)
ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=5, alpha=0.35)
ax1.plot(ref[:, 0], ref[:, 1], "b.", ms=4, mec="none", alpha=0.35)

ax2.plot(x0[:, 0], x0[:, 1], "r.", ms=7, alpha=0.5)
ax2.plot(pareto_set[:, 0], pareto_set[:, 1], "g.", mec="none", ms=5, alpha=0.4)
# ax2.plot(ref_x[:, 0], ref_x[:, 1], "b.", ms=4, mec="none", alpha=0.3)

x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 0.1, 30)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([problem.objective(p) for p in np.array([X1.flatten(), X2.flatten()]).T])
Z1 = Z[:, 0].reshape(len(x2), -1)
Z2 = Z[:, 1].reshape(len(x2), -1)
CS1 = ax2.contourf(X1, X2, Z1, 10, cmap=plt.cm.gray, linestyles="none", alpha=0.6)
CS2 = ax2.contourf(X1, X2, Z2, 10, cmap=plt.cm.gray, linestyles="None", alpha=0.6)

ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")
ax2.set_title("Decision space")
ax2.set_xlabel(r"$x_1$")
ax2.set_ylabel(r"$x_2$")

opt = DpN(
    dim=problem.n_var,
    n_objective=problem.n_obj,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    mu=N,
    x0=x0,
    lower_bounds=problem.xl,
    upper_bounds=problem.xu,
    max_iters=max_iters,
    type="igd",
    verbose=True,
)

while not opt.terminate():
    opt.reference_set = ref
    opt.newton_iteration()
    opt.log()

X = opt._get_primal_dual(opt.X)[0]
Y = opt.Y
M = opt.active_indicator._medroids
# x1 = M[:, 0]
# x2 = [f_x2(*_) for _ in zip(x1, M[:, 1])]
# M_x = np.c_[x1, x2]

ax1.plot(M[:, 0], M[:, 1], "r^", ms=7, alpha=0.5, clip_on=False)
# ax2.plot(M_x[:, 0], M_x[:, 1], "r^", ms=7, alpha=0.5)
ax1.plot(Y[:, 0], Y[:, 1], "r*", ms=7, alpha=0.5, clip_on=False)
ax2.plot(X[:, 0], X[:, 1], "r*", ms=7, alpha=0.5, clip_on=False)
# plot the trajectory
trajectory = np.array([x0] + opt.hist_X)
for i in range(N):
    x, y = trajectory[:, i, 0], trajectory[:, i, 1]
    ax2.quiver(
        x[:-1],
        y[:-1],
        x[1:] - x[:-1],
        y[1:] - y[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="k",
        width=0.004,
        alpha=0.8,
        headlength=4.7,
        headwidth=2.7,
        clip_on=False,
    )
trajectory = np.array([y0] + opt.hist_Y)
for i in range(N):
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
        width=0.004,
        alpha=0.8,
        headlength=4.7,
        headwidth=2.7,
    )
ax3_ = ax3.twinx()
ax3.semilogy(range(1, len(opt.hist_GD) + 1), opt.hist_GD, "b-", label="GD")
ax3.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax3_.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
ax3_.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax3.set_title("Performance")
ax3.set_xlabel("iteration")
ax3.set_xticks(range(1, max_iters + 1))
ax3.legend()
plt.savefig(f"{problem._problem.__class__.__name__}-2D-decision_space.pdf", dpi=1000)
