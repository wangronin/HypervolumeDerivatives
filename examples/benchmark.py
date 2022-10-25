import numpy as np
import pandas as pd
from hvd.algorithm import HVN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3
from joblib import Parallel, delayed

f = Eq1DTLZ1()
dim = 11
ref = np.array([1, 1, 1])
max_iters = 25
N = 15


def run(i):
    np.random.seed(i)
    # x0 = pd.read_csv("./data/EqDTLZ1.txt", header=None, sep=",").values
    x0 = f.get_pareto_set(30)
    x0[:, 0:2] += 0.02 * np.random.rand(len(x0), 2)
    y0 = np.array([f.objective(x) for x in x0])
    mu = len(x0)
    opt = HVN(
        dim=dim,
        n_objective=3,
        ref=ref,
        func=f.objective,
        jac=f.objective_jacobian,
        hessian=f.objective_hessian,
        h=f.constraint,
        h_jac=f.constraint_jacobian,
        h_hessian=f.constraint_hessian,
        mu=mu,
        lower_bounds=0,
        upper_bounds=1,
        minimization=True,
        x0=x0,
        max_iters=max_iters,
        verbose=True,
    )
    opt.run()
    return opt.hist_G_norm


data = Parallel(n_jobs=10)(delayed(run)(i) for i in range(N))
np.savez(f"{type(f).__name__}.npz", G_norm=np.array(data))
