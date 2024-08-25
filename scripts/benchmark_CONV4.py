import sys
from typing import Dict

sys.path.insert(0, "./")
import re
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import directed_hausdorff

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems import CONV4_2F
from hvd.utils import get_non_dominated
from scripts.utils import plot_4d, read_reference_set_data

np.random.seed(66)

max_iters = 5
n_jobs = 30
gen = 400
emoa = "NSGA-III"
problem_name = "CONV4_2F"
problem = CONV4_2F(boundry_constraints=False)
pareto_front = problem.get_pareto_front(5000)
reference_point = np.array([2, 2, 2, 10])  # for hypervolume
path = "./data-reference/CONV4_2F/"


def match_cluster(Y: np.ndarray, ref: Dict) -> np.ndarray:
    idx = []
    for y in Y:
        idx.append(np.argmin([directed_hausdorff(ref[i], np.atleast_2d(y))[0] for i in range(len(ref))]))
    return np.array(idx)


def execute(run: int):
    # read the reference set
    ref, x0, y0, Y_label, eta = read_reference_set_data(path, problem_name, emoa, run, gen)
    all_ref = np.vstack([r for r in ref.values()])
    # initialize the optimizer
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
    # remove the dominated solution in Y
    Y = opt.run()[1]
    Y = get_non_dominated(Y)
    # plotting the final approximation set
    if 1 < 2:
        fig_name = f"./plots/{problem_name}_DpN_{emoa}_run{run}_{gen}.pdf"
        plot_4d(y0, Y, all_ref, opt.hist_Y, opt.history_medoids, opt.hist_IGD, opt.hist_R_norm, fig_name)
    # save the final approximation set
    if 1 < 2:
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y.csv", index=False)
        df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
        df_y0.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}_y0.csv", index=False)

    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    return np.array([igd_value, gd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0]) for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x.csv")
]
if 1 < 2:
    data = []
    for i in run_id:
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
df.to_csv(f"results/{problem_name}-DpN-{emoa}-{gen}.csv", index=False)
