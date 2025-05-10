import sys

sys.path.insert(0, "./")
import re
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import LocalOutlierFactor
from sklearn_extra.cluster import KMedoids

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.mmd import MMD, laplace, rbf
from hvd.mmd_newton import MMDNewton
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
    PymooProblemWithAD,
)
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import plot_2d, plot_3d, read_reference_set_data

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
problem_name = sys.argv[1]
print(problem_name)

if problem_name.startswith("DTLZ"):
    n_var = 7 if problem_name == "DTLZ1" else 10
    problem = locals()[problem_name](n_var=n_var, boundry_constraints=True)
elif problem_name.startswith("ZDT"):
    problem = PymooProblemWithAD(locals()[problem_name]())

path = "./MMD_data/"
# emoa = "NSGA-II"
emoa = "MOEAD"
gen = 200 if emoa == "MOEAD" else 300
# get hyperparameters
params = pd.read_csv("./scripts/benchmark_MMD_param.csv", index_col=None, header=0)
params = params[(params.algorithm == emoa) & (params.problem == problem_name)]
kernel_name, theta = params["kernel"].values[0], params["param"].values[0]
kernel = locals()[kernel_name]


def execute(run: int) -> np.ndarray:
    # read the reference set
    ref, x0, y0, Y_index, eta = read_reference_set_data(path, problem_name, emoa, run, gen)
    ref_list = np.vstack([r for r in ref.values()])
    N = len(x0)
    # create the algorithm
    pareto_front = problem.get_pareto_front(1000) if problem.n_obj == 2 else problem.get_pareto_front()
    # TODO: move this part to the problems
    if len(pareto_front) > 1000:
        km = KMedoids(n_clusters=1000, method="alternate", random_state=0, init="k-medoids++").fit(
            pareto_front
        )
        pareto_front = pareto_front[km.medoid_indices_]
    mmd = MMD(n_var=problem.n_var, n_obj=problem.n_obj, ref=pareto_front, theta=theta, kernel=kernel)
    metrics = dict(GD=GenerationalDistance(pareto_front), IGD=InvertedGenerationalDistance(pareto_front))
    # compute the initial performance metrics
    hv_value0 = hypervolume(y0, ref=ref_point[problem_name])
    gd_value0 = metrics["GD"].compute(Y=y0)
    igd_value0 = metrics["IGD"].compute(Y=y0)
    mmd_value0 = mmd.compute(Y=y0)
    print(f"initial HV: {hv_value0}")
    print(f"initial GD: {gd_value0}")
    print(f"initial IGD: {igd_value0}")
    print(f"initial MMD: {mmd_value0}")
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
        verbose=True,
        metrics=metrics,
        matching=False,
        preconditioning=True,
        theta=theta,
        kernel=kernel,
    )
    # remove the dominated ones in the final solutions
    Y = opt.run()[1]
    Y = get_non_dominated(Y)
    # if problem.n_obj == 3:
    # score = LocalOutlierFactor(n_neighbors=5).fit_predict(Y)
    # Y = Y[score != -1]
    # plotting the final approximation set
    if 11 < 2:
        fig_name = f"./plots/{problem_name}_MMD_{emoa}_run{run}_{gen}.pdf"
        if problem.n_obj == 2:
            plot_2d(
                y0,
                Y,
                ref_list,
                pareto_front,
                opt.history_Y,
                opt.history_medoids,
                opt.history_metrics,
                opt.history_R_norm,
                fig_name,
            )
        elif problem.n_obj == 3:
            plot_3d(
                y0,
                Y,
                ref_list,
                pareto_front,
                opt.history_Y,
                opt.history_medoids,
                opt.history_metrics,
                opt.history_R_norm,
                fig_name,
            )
    # save the final approximation set
    if 11 < 2:
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(f"{problem_name}_MMD_{emoa}_run{run}_{gen}_y.csv", index=False)
        df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
        df_y0.to_csv(f"{problem_name}_MMD_{emoa}_run{run}_{gen}_y0.csv", index=False)

    hv_value = hypervolume(Y, ref=ref_point[problem_name])
    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    mmd_value = mmd.compute(Y=Y)
    return np.array([hv_value, igd_value, gd_value, mmd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]
if problem_name == "DTLZ4" and emoa == "MOEAD":
    run_id = list(set(run_id) - set([3]))

if 11 < 2:
    data = []
    for i in run_id:
        print(i)
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["HV", "IGD", "GD", "MMD", "Jac_calls"])
df.to_csv(f"results/{problem_name}-MMD-{emoa}-300.csv", index=False)
