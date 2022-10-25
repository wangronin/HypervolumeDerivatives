import numpy as np
from hvd.algorithm import HVN
from hvd.problems import Eq1DTLZ1
from joblib import Parallel, delayed
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions


class ProblemWrapper(ElementwiseProblem):
    def __init__(self, problem):
        self._problem = problem
        super().__init__(
            n_var=problem.n_decision_vars, n_obj=problem.n_objectives, n_eq_constr=1, xl=0.0, xu=1.0
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._problem.objective(x)
        out["H"] = self._problem.constraint(x)


def hybrid(seed):
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
    termination = get_termination("n_gen", 1000)

    f = Eq1DTLZ1(n_objectives=3, n_decision_vars=11)
    problem = ProblemWrapper(f)
    algorithm = AdaptiveEpsilonConstraintHandling(NSGA3(pop_size=200, ref_dirs=ref_dirs), perc_eps_until=0.5)
    # execute the optimization
    res = minimize(problem, algorithm, termination, seed=seed, verbose=False)
    X = np.array([p._X for p in res.pop])

    opt = HVN(
        dim=11,
        n_objective=3,
        ref=np.array([1, 1, 1]),
        func=f.objective,
        jac=f.objective_jacobian,
        hessian=f.objective_hessian,
        h=f.constraint,
        h_jac=f.constraint_jacobian,
        h_hessian=f.constraint_hessian,
        mu=len(X),
        lower_bounds=0,
        upper_bounds=1,
        minimization=True,
        x0=X,
        max_iters=10,
        verbose=False,
    )
    X = opt.run()[0]
    CPU_time = np.sum(opt.hist_CPU_time_FE) / 1e9
    return X, CPU_time


N = 15
# create the algorithm object
data = Parallel(n_jobs=N)(delayed(hybrid)(i) for i in range(N))
np.savez(f"Eq1DTLZ1.npz", data=data)
