import sys

sys.path.insert(0, "./")
import numpy as np
import pandas as pd

from hvd.newton import DpN
from hvd.problems import DTLZ7, PymooProblemWithAD
from scripts.utils import plot_3d

np.random.seed(66)

max_iters = 8
run = 1
problem = PymooProblemWithAD(DTLZ7())
pareto_front = problem.get_pareto_front()

# load the reference set
path = "./DTLZ/"
ref_label = pd.read_csv(path + f"DTLZ7_NSGA-III_run_{run}_component_id.csv", header=None).values[0]
n_cluster = len(np.unique(ref_label))
ref = dict()
for i in range(n_cluster):
    ref[i] = pd.read_csv(path + f"DTLZ7_NSGA-III_run_{run}_filling_comp{i+1}.csv", header=None).values

all_ref = np.concatenate([v for v in ref.values()], axis=0)
# the load the final population from an EMOA
x0 = pd.read_csv(path + f"DTLZ7_NSGA-III_run_{run}_lastpopu_x.csv", header=None).values[0:200]
y0 = pd.read_csv(path + f"DTLZ7_NSGA-III_run_{run}_lastpopu_y.csv", header=None).values[0:200]
Y_label = pd.read_csv(path + f"DTLZ7_NSGA-III_run_{run}_lastpopu_labels.csv", header=None).values.ravel()
Y_label = Y_label[0:200] - 1
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
    X0=x0,
    xl=problem.xl,
    xu=problem.xu,
    max_iters=max_iters,
    type="igd",
    verbose=True,
    pareto_front=pareto_front,
    Y_label=Y_label,
)
X, Y, _ = opt.run()

fig_name = "DTLZ7_example.pdf"
plot_3d(y0, Y, all_ref, opt.history_Y, opt.history_medoids, opt.hist_IGD, opt.history_R_norm, fig_name)
data = np.concatenate([np.c_[[0] * N, y0], np.c_[[max_iters] * N, opt.history_Y[-1]]], axis=0)
df = pd.DataFrame(data, columns=["iteration", "f1", "f2", "f3"])
df.to_csv("DTLZ7_example.csv")
