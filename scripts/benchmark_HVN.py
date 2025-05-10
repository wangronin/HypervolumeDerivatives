import numpy as np
from joblib import Parallel, delayed
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from hvd.hypervolume import hypervolume
from hvd.newton import HVN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3, Eq1DTLZ4, Eq1IDTLZ1, Eq1IDTLZ2, Eq1IDTLZ3, Eq1IDTLZ4
from hvd.problems.base import MOOAnalytical, PymooProblemWrapper
from hvd.utils import non_domin_sort


def hybrid(seed: int, problem: MOOAnalytical, ref: np.ndarray):
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
    termination = get_termination("n_gen", 1000)
    algorithm = AdaptiveEpsilonConstraintHandling(NSGA3(pop_size=200, ref_dirs=ref_dirs), perc_eps_until=0.5)
    # execute the optimization
    res = minimize(PymooProblemWrapper(problem), algorithm, termination, seed=seed, verbose=False)
    HV0 = hypervolume(res.F, ref)
    X_ = res.X
    X = np.array([p._X for p in res.pop])  # final approximation set of NSGA-III
    problem.CPU_time = 0  # clear the CPU_time counter since we only need to measure the time taken by HVN
    opt = HVN(
        n_var=11,
        n_obj=3,
        ref=ref,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        h=problem.constraint,
        h_jac=problem.constraint_jacobian,
        h_hessian=problem.constraint_hessian,
        N=len(X),
        xl=0,
        xu=1,
        minimization=True,
        X0=X,
        max_iters=10,
        verbose=False,
        problem_name=type(problem).__name__,
    )
    X, Y, _ = opt.run()
    idx = non_domin_sort(Y, only_front_indices=True)[0]
    HV = opt.history_HV[-1]
    CPU_time = problem.CPU_time / 1e9
    return X[idx], X_, CPU_time, HV0, HV


refs = {
    "Eq1DTLZ1": np.array([1, 1, 1]),
    "Eq1DTLZ2": np.array([1, 1, 1]),
    "Eq1DTLZ3": np.array([1, 1, 1]),
    "Eq1DTLZ4": np.array([1.2, 5e-3, 5e-4]),
    "Eq1IDTLZ1": np.array([1, 1, 1]),
    "Eq1IDTLZ2": np.array([1, 1, 1]),
    "Eq1IDTLZ3": np.array([1, 1, 1]),
    "Eq1IDTLZ4": np.array([-0.4, 0.6, 0.6]),
}

N = 15
problems = [
    Eq1DTLZ1(3, 11),
    Eq1DTLZ2(3, 11),
    Eq1DTLZ3(3, 11),
    Eq1DTLZ4(3, 11),
    Eq1IDTLZ1(3, 11),
    Eq1IDTLZ2(3, 11),
    Eq1IDTLZ3(3, 11),
    Eq1IDTLZ4(3, 11),
]

for problem in problems:
    ref = refs[type(problem).__name__]
    data = Parallel(n_jobs=N)(delayed(hybrid)(i, problem, ref) for i in range(N))
    np.savez(f"{type(problem).__name__}-hybrid.npz", data=data)
