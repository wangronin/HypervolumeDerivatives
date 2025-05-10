import sys

sys.path.insert(0, "./")
import numpy as np
import pandas as pd

from hvd.newton import DpN
from hvd.problems import ZDT1, PymooProblemWithAD
from scripts.utils import plot_2d

np.random.seed(66)

max_iters = 13
problem = PymooProblemWithAD(ZDT1(n_var=3))
pareto_front = problem.get_pareto_front(1000)

# load the reference set
ref = pd.read_csv("./ZDT1/ZDT1_REF_Filling.csv", header=None).values
medoids = pd.read_csv("./ZDT1/ZDT1_REF_Match_30points.csv", header=None).values
# the load the final population from an EMOA
x0 = pd.read_csv("./ZDT1/ZDT1_Pop_x.csv", header=None).values
y0 = pd.read_csv("./ZDT1/ZDT1_Pop_y.csv", header=None).values
eta = {0: pd.read_csv("./ZDT1/ZDT1_eta.csv", header=None).values.ravel()}
N = len(x0)

opt = DpN(
    dim=problem.n_var,
    n_obj=problem.n_obj,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    g=problem.ieq_constraint,
    g_jac=problem.ieq_jacobian,
    N=N,
    x0=x0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    type="igd",
    verbose=True,
    eta=eta,
    pareto_front=pareto_front,
)
X, Y, _ = opt.run()

fig_name = f"ZDT1_example.pdf"
plot_2d(
    y0, Y, ref, pareto_front, opt.history_Y, opt.history_medoids, opt.hist_IGD, opt.history_R_norm, fig_name
)
data = [np.c_[[i + 1] * N, y] for i, y in enumerate(opt.history_Y)]
df = pd.DataFrame(np.concatenate(data, axis=0), columns=["iteration", "f1", "f2"])
df.to_csv("ZDT1_example.csv")
