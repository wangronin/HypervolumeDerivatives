import numpy as np
from hvd.algorithm import HVN
from hvd.hypervolume import hypervolume
from hvd.problems import (
    Eq1DTLZ1,
    Eq1DTLZ2,
    Eq1DTLZ3,
    Eq1DTLZ4,
    Eq1IDTLZ1,
    Eq1IDTLZ2,
    Eq1IDTLZ3,
    Eq1IDTLZ4,
    MOOAnalytical,
)
from hvd.utils import non_domin_sort
from joblib import Parallel, delayed
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions


class ProblemWrapper(ElementwiseProblem):
    """Wrapper for Eq-DTLZ problems"""

    def __init__(self, problem: MOOAnalytical):
        self._problem = problem
        super().__init__(
            n_var=problem.n_decision_vars, n_obj=problem.n_objectives, n_eq_constr=1, xl=0.0, xu=1.0
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        out["F"] = self._problem.objective(x)
        out["H"] = self._problem.constraint(x)


def hybrid(seed: int, problem: MOOAnalytical, ref: np.ndarray):
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
    termination = get_termination("n_gen", 1000)
    algorithm = AdaptiveEpsilonConstraintHandling(NSGA3(pop_size=200, ref_dirs=ref_dirs), perc_eps_until=0.5)
    # execute the optimization
    res = minimize(ProblemWrapper(problem), algorithm, termination, seed=seed, verbose=False)
    HV0 = hypervolume(res.F, ref)
    X_ = res.X
    X = np.array([p._X for p in res.pop])  # final approximation set of NSGA-III
    problem.CPU_time = 0  # clear the CPU_time counter since we only need to measure the time taken by HVN
    opt = HVN(
        dim=11,
        n_objective=3,
        ref=ref,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        h=problem.constraint,
        h_jac=problem.constraint_jacobian,
        h_hessian=problem.constraint_hessian,
        mu=len(X),
        lower_bounds=0,
        upper_bounds=1,
        minimization=True,
        x0=X,
        max_iters=10,
        verbose=False,
        problem_name=type(problem).__name__,
    )
    X, Y, _ = opt.run()
    idx = non_domin_sort(Y, only_front_indices=True)[0]
    HV = opt.hist_HV[-1]
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
# problems = [
#     Eq1DTLZ1(3, 11),
#     Eq1DTLZ2(3, 11),
#     Eq1DTLZ3(3, 11),
#     Eq1DTLZ4(3, 11),
#     Eq1IDTLZ1(3, 11),
#     Eq1IDTLZ2(3, 11),
#     Eq1IDTLZ4(3, 11),
# ]

problems = [Eq1IDTLZ3(3, 11)]

if 11 < 2:
    for problem in problems:
        CPU_time = []
        ND = []
        for i in range(3):
            ref = refs[type(problem).__name__]
            res = hybrid(i, problem, ref)
            CPU_time.append(res[2])
            ND.append(len(res[0]))
        print(f"{type(problem).__name__} - CPU time: {CPU_time} - #ND: {ND}")

for problem in problems:
    ref = refs[type(problem).__name__]
    data = Parallel(n_jobs=N)(delayed(hybrid)(i, problem, ref) for i in range(N))
    np.savez(f"{type(problem).__name__}-hybrid.npz", data=data)
