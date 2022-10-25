import numpy as np
from hvd.hypervolume import hypervolume
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3, Eq1DTLZ4
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


def NSGAIII(seed, problem):
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
    termination = get_termination("n_gen", 2400)
    problem = ProblemWrapper(problem)
    algorithm = AdaptiveEpsilonConstraintHandling(NSGA3(pop_size=200, ref_dirs=ref_dirs), perc_eps_until=0.5)
    # execute the optimization
    res = minimize(problem, algorithm, termination, seed=seed, verbose=True)
    return res.X, hypervolume(res.F, np.ones(3))


# f = Eq1DTLZ4(3, 11)
# Y = f.get_pareto_front(500)
# Scatter(angle=(45, 45)).add(Y).show()
# breakpoint()

N = 15
problems = [Eq1DTLZ1(3, 11), Eq1DTLZ2(3, 11), Eq1DTLZ3(3, 11)]
for problem in problems:
    problem = Eq1DTLZ1(n_objectives=3, n_decision_vars=11)
    data = Parallel(n_jobs=N)(delayed(NSGAIII)(i, problem) for i in range(N))
    np.savez(f"{type(problem).__name__}-NSGA3.npz", data=data)
