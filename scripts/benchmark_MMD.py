import sys

sys.path.insert(0, "./")
import re
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn_extra.cluster import KMedoids

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.mmd import MMD, laplace, linear, rbf
from hvd.mmd_newton import MMDNewton
from hvd.problems import *
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import plot, read_reference_set_data

np.random.seed(66)

# settings
max_iters = 10
n_jobs = 30
source_data_path = Path("./MMD_data/")
csv_path = Path("./")
plot_path = Path("./plots")
moea = "NSGA-III"
moea_gen = 300
# get problem
problem_name = sys.argv[1]
boundary_constraints = True
print(f"optimize {problem_name}")
# create the problem instance
problem_type = globals()[problem_name]
problem = problem_type(boundary_constraints=boundary_constraints)
# read the HV reference point
ref_point = pd.read_csv("./scripts/ref_point.csv", index_col="problem").loc[problem_name].values
# get hyperparameters
params = pd.read_csv("./scripts/benchmark_MMD_param.csv", index_col=None, header=0)
params = params[(params.algorithm == moea) & (params.problem == problem_name)]
kernel_name, theta = params["kernel"].values[0], params["param"].values[0]
kernel = locals()[kernel_name]


def execute(run: int) -> np.ndarray:
    # read the reference set
    ref, x0, y0, Y_index, eta = read_reference_set_data(source_data_path, problem_name, moea, run, moea_gen)
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
    hv_value0 = hypervolume(y0, ref=ref_point)
    gd_value0 = metrics["GD"].compute(Y=y0)
    igd_value0 = metrics["IGD"].compute(Y=y0)
    mmd_value0 = mmd.compute(Y=y0)
    print(f"initial HV: {hv_value0}")
    print(f"initial GD: {gd_value0}")
    print(f"initial IGD: {igd_value0}")
    print(f"initial MMD: {mmd_value0}")

    t0 = time.process_time_ns()
    opt = MMDNewton(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        ref=ReferenceSet(ref=ref, eta=eta, Y_idx=Y_index),
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
        N=N,
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=max_iters,
        verbose=True,
        metrics=metrics,
        matching=False,
        regularization=True,
        theta=theta,
        kernel=kernel,
    )
    Y = opt.run()[1]
    wall_clock_time = time.process_time_ns() - t0
    Y = get_non_dominated(Y)  # remove the dominated ones in the final solutions
    print(opt.history_R_norm)
    # plotting the final approximation set
    fig_name = plot_path / f"{problem_name}_MMD_{moea}_run{run}_{moea_gen}.pdf"
    plot(y0, Y, ref_list, pareto_front, fig_name, opt)
    # save the final approximation set
    if 11 < 2:
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(csv_path / f"{problem_name}_MMD_{moea}_run{run}_{moea_gen}_y.csv", index=False)
        df_y0 = pd.DataFrame(y0, columns=[f"f{i}" for i in range(1, y0.shape[1] + 1)])
        df_y0.to_csv(csv_path / f"{problem_name}_MMD_{moea}_run{run}_{moea_gen}_y0.csv", index=False)
    # calculate the performance values
    hv_value = hypervolume(Y, ref=ref_point)
    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    mmd_value = mmd.compute(Y=Y)
    return np.array(
        [hv_value, igd_value, gd_value, mmd_value, opt.state.n_jac_evals, wall_clock_time / 1000.0]
    )


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob(f"{source_data_path}/{problem_name}_{moea}_run_*_lastpopu_x_gen{moea_gen}.csv")
]
if problem_name == "DTLZ4" and moea == "MOEAD":
    run_id = list(set(run_id) - set([3]))

if 1 < 2:
    data = []
    for i in run_id:
        print(i)
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["HV", "IGD", "GD", "MMD", "Jac_calls", "wall_clock_time"])
df.to_csv(f"results/{problem_name}-MMD-{moea}-300.csv", index=False)
