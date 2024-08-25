import sys

sys.path.insert(0, "./")
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from hvd.newton import DpN
from hvd.problems import CONV3
from hvd.utils import non_domin_sort
from scripts.utils import plot_3d

np.random.seed(66)

max_iters = 10
problem = CONV3()
pareto_front = problem.get_pareto_front(5000)

# load the reference set
path = "./CONV3_example/"
ref = pd.read_csv(path + f"CONV3_NSGA-II_run_1_ref.csv", header=None).values
idx = non_domin_sort(ref, only_front_indices=True)[0]
ref = ref[idx]
idx = cdist(pareto_front, ref).min(axis=0).argsort()
idx = idx[0:-10]
ref = ref[idx]
# the load the final population from an EMOA
x0 = pd.read_csv(path + f"CONV3_NSGA-II_run_1_lastpopu_x.csv", header=None).values[:100]
y0 = pd.read_csv(path + f"CONV3_NSGA-II_run_1_lastpopu_y.csv", header=None).values[:100]
N = len(x0)

opt = DpN(
    dim=problem.n_var,
    n_obj=problem.n_obj,
    ref=ref,
    func=problem.objective,
    jac=problem.objective_jacobian,
    hessian=problem.objective_hessian,
    # g=problem.ieq_constraint,
    # g_jac=problem.ieq_jacobian,
    N=N,
    X0=x0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    type="igd",
    verbose=True,
    pareto_front=pareto_front,
)
X, Y, _ = opt.run()
fig_name = f"{problem.__class__.__name__}.pdf"
plot_3d(y0, Y, ref, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
data = np.concatenate([np.c_[[0] * N, y0], np.c_[[max_iters] * N, opt.hist_Y[-1]]], axis=0)
df = pd.DataFrame(data, columns=["iteration", "f1", "f2", "f3"])
df.to_csv("CONV3_example.csv")
