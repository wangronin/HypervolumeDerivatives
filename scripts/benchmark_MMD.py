import sys

sys.path.insert(0, "./")
import re
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.mmd import MMD
from hvd.mmd_newton import MMDNewton
from hvd.problems import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, PymooProblemWithAD
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import plot_2d, read_reference_set_data

np.random.seed(66)

max_iters = 10
n_jobs = 30
problem_name = sys.argv[1]
print(problem_name)
problem = PymooProblemWithAD(locals()[problem_name]())
pf = problem.get_pareto_front(1000)
path = "./ZDT/"
emoa = "NSGA-II"
gen = 300
ref_point = dict(
    ZDT1=[11, 11],
    ZDT2=[11, 11],
    ZDT3=[11, 11],
    ZDT4=[11, 11],
    ZDT6=[11, 11],
    DTLZ1=[2, 2, 2],
    DTLZ2=[2, 2, 2],
    DTLZ3=[2, 2, 2],
    DTLZ4=[2, 2, 2],
)


def execute(run: int) -> np.ndarray:
    # read the reference set
    ref, x0, y0, Y_index, eta = read_reference_set_data(path, problem_name, emoa, run, gen)
    ref_ = np.vstack([r for r in ref])
    N = len(x0)
    # create the algorithm
    pareto_front = problem.get_pareto_front()
    # `theta` parameter is very important, `1/N` is empirically good
    theta = 1.0 / N
    mmd = MMD(n_var=problem.n_var, n_obj=problem.n_obj, ref=pareto_front, theta=theta)
    metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))

    # TODO: passing the hyperparameter; we do not use matching
    opt = MMDNewton(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        ref=ReferenceSet(ref=ref, eta=eta, Y_idx=Y_index),
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
        verbose=False,
        metrics=metrics,
        matching=False,
        preconditioning=False,
        theta=theta,
    )
    # remove the dominated ones in the final solutions
    Y = opt.run()[1]
    Y = get_non_dominated(Y)
    # plotting the final approximation set
    if 11 < 2:
        fig_name = f"./plots/{problem_name}_MMD_{emoa}_run{run}_{gen}"
        plot_2d(
            y0, Y, ref_, pf, opt.history_Y, opt.history_medoids, opt.hist_IGD, opt.history_R_norm, fig_name
        )
    # Please save the final approximation set
    if 1 < 2:
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(f"{problem_name}_MMD_{emoa}_run{run}_{gen}_y.csv", index=False)
        df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
        df_y0.to_csv(f"{problem_name}_MMD_{emoa}_run{run}_{gen}_y0.csv", index=False)

    hv_value = hypervolume(Y, ref=ref_point[problem_name])
    gd_value = GenerationalDistance(pf).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pf).compute(Y=Y)
    mmd_value = mmd.compute(Y=Y)
    return np.array([hv_value, igd_value, gd_value, mmd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]
if 11 < 2:
    for i in run_id:
        execute(i)
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["HV", "IGD", "GD", "MMD", "Jac_calls"])
df.to_csv(f"results/{problem_name}-MMD-{emoa}-{gen}.csv", index=False)
