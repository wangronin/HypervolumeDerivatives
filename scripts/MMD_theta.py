import sys

sys.path.insert(0, "./")

import glob
import re

import matplotlib.pylab as plt
import numpy as np
from matplotlib import rcParams
from scipy.linalg import block_diag, solve

from hvd.mmd import MMD, laplace, rbf
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
from hvd.reference_set import ReferenceSet
from hvd.utils import precondition_hessian
from scripts.utils import read_reference_set_data

rcParams["font.size"] = 12
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

path = "./MMD_data/"
emoa = "MOEAD"
gen = 200
kernel = laplace
problem_name = sys.argv[1]
if problem_name.startswith("DTLZ"):
    n_var = 7 if problem_name == "DTLZ1" else 10
    problem = locals()[problem_name](n_var=n_var, boundry_constraints=True)
elif problem_name.startswith("ZDT"):
    problem = PymooProblemWithAD(locals()[problem_name]())

run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob.glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]

thetas = np.logspace(-1, 4, 15)
for run in [9]:
    ref, X0, Y0, Y_index, eta = read_reference_set_data(path, problem_name, emoa, run, gen)
    ref = ReferenceSet(ref=ref, eta=eta, Y_idx=Y_index)
    ref.shift(0.08)  # simulate the initial shift of MMD Newton
    sigma_min = []
    sigma_max = []
    for theta in thetas:
        print(theta)
        mmd = MMD(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            ref=ref,
            func=problem.objective,
            jac=problem.objective_jacobian,
            hessian=problem.objective_hessian,
            theta=theta,
            kernel=kernel,
        )
        res = mmd.compute_hessian(X=X0)
        H, g = res["MMDdX2"], res["MMDdX"]
        H_ = precondition_hessian(H)
        g = g.reshape(-1, 1)
        sigmas = np.linalg.eigh(H_)[0]
        sigma_min.append(sigmas[0])
        sigma_max.append(sigmas[-1])

    print(sigma_min)
    condition_number = np.array(sigma_max) / np.array(sigma_min)
    # first filter the absolute condition number into [1, 1e2]
    # then we select the theta whose eigenspectrum is the "easiest" to pre-condition
    idx = np.nonzero((condition_number > 1) & (condition_number < 1e2))[0]
    # print(thetas[idx][k])
    plt.loglog(thetas, condition_number, "k.", ls="--", ms=8)
    plt.show()
