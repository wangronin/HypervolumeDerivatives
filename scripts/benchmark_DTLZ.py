import sys

sys.path.insert(0, "./")
import re
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.core.problem import Problem as PymooProblem

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.newton import DpN
from hvd.problems import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    IDTLZ1,
    IDTLZ2,
    IDTLZ3,
    IDTLZ4,
    PymooProblemWithAD,
)
from hvd.utils import get_non_dominated
from scripts.utils import plot_3d, read_reference_set_data

np.random.seed(66)

max_iters = 10
n_jobs = 30
problem_name = sys.argv[1]
gen = 150
emoa = "NSGA-II"
print(problem_name)

if problem_name.startswith("IDTLZ"):
    problem = locals()[problem_name](boundry_constraints=True)
    path = "./data-reference/IDTLZ/"
elif problem_name.startswith("DTLZ"):
    problem = locals()[problem_name](boundry_constraints=True)
    # path = "./data-reference/DTLZ/"
    path = "./data"

problem = PymooProblemWithAD(problem) if isinstance(problem, PymooProblem) else problem
pareto_front = problem.get_pareto_front()
reference_point = {  # for hypervolume
    "DTLZ[1-6]": np.array([1, 1, 1]),
    "DTLZ7": np.array([1, 1, 6]),
    "IDTLZ1[1-4]": np.array([1, 1, 1]),
}


def execute(run: int) -> np.ndarray:
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
        x0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=max_iters,
        type="igd",
        verbose=True,
        pareto_front=pareto_front,
        eta=eta,
        Y_label=Y_label,
    )
    Y = opt.run()[1]
    # remove the dominated ones in the final solutions
    Y = get_non_dominated(Y)
    # plotting the final approximation set
    if 1 < 2:
        fig_name = f"./plots/{problem_name}_DpN_{emoa}_run{run}_{gen}"
        plot_3d(
            y0, Y, all_ref, opt.history_Y, opt.history_medoids, opt.hist_IGD, opt.history_R_norm, fig_name
        )
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
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]
if problem_name == "DTLZ2" and emoa == "SMS-EMOA":
    run_id = list(set(run_id) - set([7]))
if problem_name == "DTLZ4" and emoa == "MOEAD":
    run_id = list(set(run_id) - set([3]))
if problem_name == "IDTLZ4" and emoa == "NSGA-III":
    run_id = list(set(run_id) - set([12]))

if 1 < 2:
    data = []
    # for i in [1]:
    for i in run_id:
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
df.to_csv(f"results/{problem_name}-DpN-{emoa}-{gen}.csv", index=False)
