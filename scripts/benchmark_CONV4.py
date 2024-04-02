import sys

sys.path.insert(0, "./")
import random
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import rcParams
from pymoo.core.problem import Problem as PymooProblem

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.newton import DpN
from hvd.problems import CONV4_2F
from hvd.utils import get_non_dominated

plt.style.use("ggplot")
rcParams["font.size"] = 17
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
# rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 17
rcParams["ytick.labelsize"] = 17
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

max_iters = 6
n_jobs = 30
gen = 400
emoa = "NSGA-III"
problem_name = "CONV4_2F"
problem = locals()[problem_name](boundry_constraints=True)
pareto_front = problem.get_pareto_front()
ref = {"DTLZ[1-6]": np.array([1, 1, 1]), "IDTLZ1[1-4]": np.array([1, 1, 1]), "DTLZ7": np.array([1, 1, 6])}
path = "./data-reference/CONV4_2F_NSGA-III_ref/"


def execute(run: int):
    ref_label = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_component_id.csv", header=None).values[0]
    n_cluster = len(np.unique(ref_label))
    ref = dict()
    eta = dict()
    # load the reference set
    for i in range(n_cluster):
        if n_cluster == 1:
            ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_gen{gen}.csv"
        else:
            ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_{i+1}_gen{gen}.csv"
        try:
            r = pd.read_csv(ref_file, header=None).values
        except:
            continue
        # downsample the reference; otherwise, initial clustering take too long
        ref[i] = np.array(random.sample(r.tolist(), 700)) if len(r) >= 700 else r
        try:
            eta[i] = pd.read_csv(
                f"{path}/{problem_name}_{emoa}_run_{run}_eta_{i+1}_gen{gen}.csv", header=None
            ).values.ravel()
        except:
            if i > 0 and eta[i - 1] is not None:  # copy the shift direction from the last cluster
                eta[i] = eta[i - 1]
            else:
                eta = None

    # sometimes the precomputed `eta` value can be `nan`
    if (eta is not None) and (np.any([np.any(np.isnan(_eta)) for _eta in eta.values()])):
        eta = None

    # the load the final population from an EMOA
    x0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_x.csv", header=None).values
    y0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_y.csv", header=None).values
    Y_label = pd.read_csv(
        f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_labels.csv", header=None
    ).values.ravel()
    Y_label = Y_label - 1
    # removing the outliers in `Y`
    idx = (Y_label != -2) & (Y_label != -1)
    x0 = x0[idx]
    y0 = y0[idx]
    Y_label = Y_label[idx]
    all_ref = np.vstack([r for r in ref.values()])

    # TODO: this is an ad-hoc solution. Maybe fix this special case in the `ReferenceSet` class
    # if the minimal number of points in the `ref` clusters is smaller than
    # the maximal number of points in `y0` clusters, then we merge all clusters
    min_point_ref_cluster = np.min([len(r) for r in ref.values()])
    max_point_y_cluster = np.max(np.unique(Y_label, return_counts=True)[1])
    # if the number of clusters of `Y` is more than that of the reference set
    if (len(np.unique(Y_label)) > len(ref)) or (max_point_y_cluster > min_point_ref_cluster):
        ref = np.vstack([r for r in ref.values()])
        Y_label = np.zeros(len(y0))
        eta = None
        # ensure the number of approximation points is less than the number of reference points
        if len(ref) < len(y0):
            n = len(ref)
            x0 = x0[:n]
            y0 = y0[:n]
            Y_label = Y_label[:n]

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
    X, Y, _ = opt.run()
    # remove the dominated solution in Y
    Y = get_non_dominated(Y)
    if 1 < 2:  # save the final approximation set
        df = pd.DataFrame(Y, columns=[f"f{i}" for i in range(1, Y.shape[1] + 1)])
        df.to_csv(f"{problem_name}_DpN_{emoa}_run{run}_{gen}.csv", index=False)

    gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
    # hv_value = hypervolume(Y, ref)
    return np.array([igd_value, gd_value, opt.state.n_jac_evals])


# get all run IDs
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0]) for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x.csv")
]
if 11 < 2:
    data = []
    for i in run_id[1:]:
        print(i)
        data.append(execute(i))
else:
    data = Parallel(n_jobs=n_jobs)(delayed(execute)(run=i) for i in run_id)

df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "Jac_calls"])
df.to_csv(f"results/{problem_name}-DpN-{emoa}-{gen}.csv", index=False)
