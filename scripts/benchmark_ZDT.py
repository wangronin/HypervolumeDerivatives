import sys

sys.path.insert(0, "./")
import re
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, PymooProblemWithAD
from hvd.utils import get_non_dominated
from scripts.utils import plot_2d, read_reference_set_data

np.random.seed(66)

max_iters = 5
n_jobs = 30
problem_name = sys.argv[1]
print(problem_name)
problem = PymooProblemWithAD(locals()[problem_name]())
pf = problem.get_pareto_front(1000)
path = "./data-reference/ZDT/"
emoa = "SMS-EMOA"
gen = 300


def execute(run: int) -> np.ndarray:
    # read the reference set
    ref, x0, y0, Y_label, eta = read_reference_set_data(path, problem_name, emoa, run, gen)
    ref_ = np.vstack([r for r in ref.values()])
    # create the algorithm
    # TODO: create an alternative interface to directly take `problem` as input
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
        pareto_front=pf,
        eta=eta,
        Y_label=Y_label,
    )
    # remove the dominated ones in the final solutions
    Y = opt.run()[1]
    Y = get_non_dominated(Y)
    # plotting the final approximation set
    if 1 < 2:
        fig_name = f"./plots/{problem_name}_DpN_{emoa}_run{run}_{gen}"
        plot_2d(y0, Y, ref_, pf, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    # save the final approximation set
    if 1 < 2:
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y.csv", index=False)
        df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
        df_y0.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y0.csv", index=False)
    gd_value = GenerationalDistance(pf).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pf).compute(Y=Y)
    return np.array([igd_value, gd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]
if problem_name == "ZDT2" and emoa == "NSGA-III":
    run_id = list(set(run_id) - set([24]))

if 11 < 2:
    for i in run_id:
        execute(i)
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
df.to_csv(f"results/{problem_name}-DpN-{emoa}-{gen}.csv", index=False)
