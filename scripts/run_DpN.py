import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.io import loadmat

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


# def pymoo_problem_wrapper(problem: Problem):
#     out = {}

#     def evaluate_func(x):
#         x = np.atleast_2d(x)
#         problem._evaluate(x, out)
#         return out["F"].ravel()

#     return evaluate_func


np.random.seed(66)

N = 10
max_iters = 10
problem = PymooProblemWithAD(ZDT1())
pareto_front = problem.get_pareto_front(1000)

if 11 < 2:
    # load the reference set
    ref = pd.read_csv("./data-reference-set/ZDT1_NSGA-II_ref.csv", header=0).values
    # ref -= 0.01
    # the load the final population from an EMOA
    data = loadmat("./data/ZDT1_NSGA-II.mat")
    df = pd.DataFrame(data["data"], columns=[c[0] for c in data["columns"][0]])
    df = df[(df.run == 2) & (df.iteration == 800)]
    x0 = df.loc[:, "x1":f"x{problem.n_var}"].iloc[:10, :].values
    y0 = df.loc[:, "f1":f"f{problem.n_obj}"].iloc[:10, :].values
else:
    ref = problem.get_pareto_front(100) - 0.02
    x0 = problem.get_pareto_set(N, kind="uniform")
    x0[:, 1:] += 0.1 * np.random.rand(N, problem.n_var - 1)
    y0 = np.array([problem.objective(x) for x in x0])

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

delta = 0.03
while not opt.terminate():
    # ref -= delta
    # delta *= 0.5  # exponential decay of the shift
    opt.reference_set = ref
    opt.newton_iteration()
    opt.log()

X = opt._get_primal_dual(opt.X)[0]
Y = opt.Y
M = opt.active_indicator._medroids

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
plt.subplots_adjust(right=0.93, left=0.05)
ax1.plot(y0[:, 0], y0[:, 1], "r.", ms=7, alpha=0.5)
ax1.plot(pareto_front[:, 0], pareto_front[:, 1], "g.", mec="none", ms=4, alpha=0.2)
ax1.plot(ref[:, 0], ref[:, 1], "b.", ms=4, mec="none", alpha=0.3)
ax1.plot(M[:, 0], M[:, 1], "r^", ms=7, alpha=0.5)
ax1.plot(Y[:, 0], Y[:, 1], "r*", ms=7, alpha=0.5)

if 1 < 2:
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
            width=0.003,
            alpha=0.5,
            headlength=4.5,
            headwidth=2.5,
        )

ax1.set_title("Objective space")
ax1.set_xlabel(r"$f_1$")
ax1.set_ylabel(r"$f_2$")

ax22 = ax2.twinx()
ax2.semilogy(range(1, len(opt.hist_GD) + 1), opt.hist_GD, "b-", label="GD")
ax2.semilogy(range(1, len(opt.hist_IGD) + 1), opt.hist_IGD, "r-", label="IGD")
ax22.semilogy(range(1, len(opt.hist_R_norm) + 1), opt.hist_R_norm, "g--")
ax22.set_ylabel(r"$||R(\mathbf{X})||$", color="g")
ax2.set_title("Performance")
ax2.set_xlabel("iteration")
ax2.set_xticks(range(1, max_iters + 1))
ax2.legend()
plt.tight_layout()
plt.savefig(f"ZDT1-{N}points.pdf", dpi=1000)
