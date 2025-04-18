import sys

sys.path.insert(0, "./")
import numpy as np
import pandas as pd

from hvd.newton import DpN
from hvd.problems import ZDT3, PymooProblemWithAD
from scripts.utils import plot_2d

np.random.seed(66)

max_iters = 8
run = 1
problem = PymooProblemWithAD(ZDT3())
pareto_front = problem.get_pareto_front(500)

# load the reference set
path = "./ZDT/ZDT3/"
ref_label = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_component_id.csv", header=None).values[0]
n_cluster = len(np.unique(ref_label))
ref = dict()
eta = dict()
for i in range(n_cluster):
    ref[i] = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_filling_comp{i+1}.csv", header=None).values
    eta[i] = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_eta_{i+1}.csv", header=None).values.ravel()

all_ref = np.concatenate([v for v in ref.values()], axis=0)
# the load the final population from an EMOA
x0 = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_lastpopu_x.csv", header=None).values[0:50]
y0 = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_lastpopu_y.csv", header=None).values[0:50]
Y_label = pd.read_csv(path + f"ZDT3_NSGA-II_run_{run}_lastpopu_labels.csv", header=None).values.ravel()[0:50]
Y_label -= 1
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
    eta=eta,
    Y_label=Y_label,
)
X, Y, _ = opt.run()

fig_name = f"ZDT3_example.pdf"
plot_2d(
    y0, Y, ref, pareto_front, opt.history_Y, opt.history_medoids, opt.hist_IGD, opt.history_R_norm, fig_name
)
data = np.concatenate([np.c_[[0] * N, y0], np.c_[[max_iters] * N, opt.history_Y[-1]]], axis=0)
df = pd.DataFrame(data, columns=["iteration", "f1", "f2"])
df.to_csv("ZDT3_example.csv")
