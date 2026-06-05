import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems import IDTLZ2
from hvd.reference_set import ReferenceSet

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 20
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

f = IDTLZ2(n_var=12)
max_iters = 10
pareto_front = f.get_pareto_front()
x0 = pd.read_csv("./data/IDTLZ2_X0.csv", header=None).values
y0 = np.array([f.objective(x) for x in x0])
ref = pd.read_csv("./data/IDTLZ2_reference.csv", header=None).values
N = len(x0)
metrics = dict(GD=GenerationalDistance(ref=ref), IGD=InvertedGenerationalDistance(ref=ref))
opt = DpN(
    dim=12,
    n_obj=3,
    ref=ReferenceSet(ref=ref),
    func=f.objective,
    jac=f.objective_jacobian,
    hessian=f.objective_hessian,
    N=N,
    x0=x0,
    xl=0,
    xu=1,
    max_iters=max_iters,
    verbose=True,
    metrics=metrics,
    type="igd",
    preconditioning=True,
)
X, Y, stop = opt.run()
# plotting
reference_set = opt.ref.reference_set
x, y, z = pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2]
triang = mtri.Triangulation(x, y)
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
zmid = z[triang.triangles].mean(axis=1)
p = np.c_[xmid, ymid, zmid]
mask = np.array([np.any(np.all(pp > p, axis=1)) for pp in p])
triang.set_mask(mask)

fig = plt.figure(figsize=(8, 22), layout="compressed")
ax = fig.add_subplot(3, 1, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(35, 45)
lines = []
lines += ax.plot(y0[:, 0], y0[:, 1], y0[:, 2], "k.", ms=13)
lines += ax.plot(reference_set[:, 0], reference_set[:, 1], reference_set[:, 2], "g+", ms=13)
lines.append(ax.plot_trisurf(triang, z, color="k", alpha=0.2))
ax.legend(lines, [r"$Y_0$", "reference set", r"Pareto front"], loc="upper left")
ax.set_xlabel(r"$f_1$")
ax.set_ylabel(r"$f_2$")
ax.set_zlabel(r"$f_3$")

ax = fig.add_subplot(3, 1, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(35, 45)
lines = []
lines += ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r.", ms=13)
lines.append(ax.plot_trisurf(triang, z, color="k", alpha=0.2))
ax.legend(lines, [r"$Y_{\text{final}}$", r"Pareto front"], loc="upper left")
ax.set_xlabel(r"$f_1$")
ax.set_ylabel(r"$f_2$")
ax.set_zlabel(r"$f_3$")

ax = fig.add_subplot(3, 1, 3)
ax.set_box_aspect(1)
ax_ = ax.twinx()
ax_.set_box_aspect(1)
ax.semilogy(range(1, len(opt.history_indicator_value) + 1), opt.history_indicator_value, "b-")
ax_.semilogy(range(1, len(opt.history_R_norm) + 1), opt.history_R_norm, "g--")
ax.set_ylabel("")
ax_.set_ylabel("")
ax.text(0, 1.01, r"$\Delta_p$", transform=ax.transAxes, color="b", ha="right", va="bottom")
ax_.text(
    1.12, 1.01, r"$||R(\mathbf{X}, \lambda)||$", transform=ax.transAxes, color="g", ha="right", va="bottom"
)
ax.set_xlabel("iteration")
plt.savefig(f"IDTLZ2-DpN-example-{N}.pdf", dpi=100)
