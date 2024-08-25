import sys

sys.path.insert(0, "./")
import numpy as np
import pandas as pd

from hvd.newton import DpN
from hvd.problems import ZDT6, PymooProblemWithAD
from scripts.utils import plot_2d

np.random.seed(66)

max_iters = 5
run = 2
f = ZDT6()
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front(500)

# load the reference set
path = "./Gen1510/"
emoa = "NSGA-II"
gen = 110
problem_name = "ZDT6"
ref_label = pd.read_csv(
    f"{path}/{problem_name}_{emoa}_run_{run}_component_id_gen{gen}.csv", header=None
).values[0]
n_cluster = len(np.unique(ref_label))
ref = dict()
eta = dict()
# load the reference set
for i in range(n_cluster):
    ref[i] = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_filling_comp{i+1}_gen{gen}.csv", header=None
    ).values
    eta[i] = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_eta_{i+1}_gen{gen}.csv", header=None
    ).values.ravel()
all_ref = np.concatenate([v for v in ref.values()], axis=0)

# the load the final population from an EMOA
x0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_x_gen{gen}.csv", header=None).values
y0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_y_gen{gen}.csv", header=None).values
Y_label = pd.read_csv(
    f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_labels_gen{gen}.csv", header=None
).values.ravel()
Y_label = Y_label - 1
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
    N=len(x0),
    X0=x0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    type="igd",
    verbose=True,
    pareto_front=pareto_front,
    eta=eta,
    Y_label=Y_label,
)
X, Y, _ = opt.run()

fig_name = f"ZDT6_example.pdf"
plot_2d(y0, Y, ref, pareto_front, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)

data = np.concatenate([np.c_[[0] * N, y0], np.c_[[max_iters] * N, opt.hist_Y[-1]]], axis=0)
df = pd.DataFrame(data, columns=["iteration", "f1", "f2"])
df.to_csv("ZDT6_example.csv")
