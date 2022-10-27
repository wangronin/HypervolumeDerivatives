import numpy as np
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
from joblib import Parallel, delayed
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter


class ProblemWrapper(ElementwiseProblem):
    def __init__(self, problem):
        self._problem = problem
        super().__init__(
            n_var=problem.n_decision_vars, n_obj=problem.n_objectives, n_eq_constr=1, xl=0.0, xu=1.0
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._problem.objective(x)
        out["H"] = self._problem.constraint(x)


def NSGAIII(seed, problem, ref):
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
    termination = get_termination("n_gen", 1000)
    problem = ProblemWrapper(problem)
    algorithm = AdaptiveEpsilonConstraintHandling(NSGA3(pop_size=200, ref_dirs=ref_dirs), perc_eps_until=0.5)
    # execute the optimization
    res = minimize(problem, algorithm, termination, seed=seed, verbose=False)
    return res.X, hypervolume(res.F, ref)


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
# problems = [Eq1DTLZ1(3, 11), Eq1DTLZ2(3, 11), Eq1DTLZ3(3, 11)]
problems = [Eq1DTLZ4(3, 11), Eq1IDTLZ1(3, 11), Eq1IDTLZ2(3, 11), Eq1IDTLZ3(3, 11)]
for problem in problems:
    ref = refs[type(problem).__name__]
    data = Parallel(n_jobs=N)(delayed(NSGAIII)(i, problem, ref) for i in range(N))
    np.savez(f"{type(problem).__name__}-NSGA3-2.npz", data=data)
