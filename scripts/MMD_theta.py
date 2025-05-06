import sys

sys.path.insert(0, "./")

import glob
import re

import matplotlib.pylab as plt
import numpy as np
from matplotlib import rcParams

from hvd.mmd import MMD, laplace, rbf
from hvd.problems import DTLZ1, DTLZ2, DTLZ3, DTLZ4, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, PymooProblemWithAD
from hvd.reference_set import ReferenceSet
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
emoa = "NSGA-II"
gen = 300
kernel = rbf
problem_name = sys.argv[1]
problem = PymooProblemWithAD(locals()[problem_name]())
run_id = [
    int(re.findall(r"run_(\d+)_", s)[0])
    for s in glob.glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
]

thetas = np.logspace(-1, 4, 30)
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
        H = mmd.compute_hessian(X=X0)["MMDdX2"]
        sigmas = np.linalg.eigh(H)[0]
        sigma_min.append(sigmas[0])
        sigma_max.append(sigmas[-1])

    print(sigma_min)
    condition_number = np.array(sigma_max) / np.abs(sigma_min)
    # TODO: first filter the absolute condition number into [1, 1e3]
    # then we select the theta with the largest `sigma_min`
    idx = np.nonzero(condition_number > 1)[0]
    k = np.argmin(condition_number[idx])
    print(thetas[idx][k])
    plt.loglog(thetas, condition_number, "k.", ms=8)
    plt.loglog(thetas, condition_number, "k--")
    plt.show()
