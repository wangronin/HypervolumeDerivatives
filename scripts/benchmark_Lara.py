import sys

sys.path.insert(0, "./")
import re
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn_extra.cluster import KMedoids

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.gradient_based import Lara
from hvd.problems import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    ZDT1,
    ZDT2,
    ZDT3,
    ZDT4,
    ZDT6,
    PymooProblemWithAD,
)
from hvd.utils import get_non_dominated
from scripts.utils import read_reference_set_data

np.random.seed(66)

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
    DTLZ5=[2, 2, 2],
    DTLZ6=[2, 2, 2],
    DTLZ7=[2, 2, 10],
)

max_iters = 5
n_jobs = 30
path = "./MMD_data/"
problem_name = sys.argv[1]
print(problem_name)

if problem_name.startswith("DTLZ"):
    n_var = 7 if problem_name == "DTLZ1" else 10
    problem = locals()[problem_name](n_var=n_var, boundry_constraints=True)
elif problem_name.startswith("ZDT"):
    problem = PymooProblemWithAD(locals()[problem_name]())


def execute(emoa: str, run: int) -> np.ndarray:
    # read the reference set
    # gen = 200 if emoa == "MOEAD" else 300
    gen = 300
    ref, x0, y0, Y_index, eta = read_reference_set_data(path, problem_name, emoa, run, gen)
    N = len(x0)
    pareto_front = problem.get_pareto_front(1000) if problem.n_obj == 2 else problem.get_pareto_front()
    if len(pareto_front) > 1000:
        km = KMedoids(n_clusters=1000, method="alternate", random_state=0, init="k-medoids++").fit(
            pareto_front
        )
        pareto_front = pareto_front[km.medoid_indices_]
    metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
    # compute the initial performance metrics
    gd_value0 = metrics["GD"].compute(Y=y0)
    igd_value0 = metrics["IGD"].compute(Y=y0)
    print(f"initial GD: {gd_value0}")
    print(f"initial IGD: {igd_value0}")
    # create the algorithm
    opt = Lara(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        func=problem.objective,
        jac=problem.objective_jacobian,
        N=N,
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=max_iters,
        verbose=False,
        metrics=metrics,
    )
    Y = opt.run()[1]
    Y = get_non_dominated(Y)
    # plotting the final approximation set
    gd_value = metrics["GD"].compute(Y=Y)
    igd_value = metrics["IGD"].compute(Y=Y)
    # remove the dominated ones in the final solutions
    return np.array([igd_value, gd_value, opt.state.n_jac_evals])


for emoa in ["NSGA-II", "NSGA-III", "MOEAD"]:
    # get all run IDs
    # gen = 200 if emoa == "MOEAD" else 300
    gen = 300
    run_id = [
        int(re.findall(r"run_(\d+)_", s)[0])
        for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
    ]
    if 11 < 2:
        data = []
        for i in run_id:
            print(i)
            data.append(execute(emoa=emoa, run=i))
    else:
        data = Parallel(n_jobs=n_jobs)(delayed(execute)(emoa=emoa, run=i) for i in run_id)

    df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
    df.to_csv(f"{problem_name}-Lara-{emoa}-300.csv", index=False)
