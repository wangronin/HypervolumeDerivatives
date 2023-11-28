import copy
import sys

sys.path.insert(0, "./")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from scipy.io import savemat

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.problems import CF9, CONV3, CONV4, UF7, UF8, Eq1DTLZ2, Eq1DTLZ3, MOOAnalytical

pop_to_numpy = lambda pop: np.array([np.r_[ind.X, ind.F, ind.H, ind.G] for ind in pop])
ref_point = np.array([11, 11])


class ProblemWrapper(ElementwiseProblem):
    def __init__(self, problem: MOOAnalytical) -> None:
        self._problem = problem
        super().__init__(
            n_var=problem.n_decision_vars,
            n_obj=problem.n_objectives,
            xl=problem.lower_bounds,
            xu=problem.upper_bounds,
            n_ieq_constr=self._problem.n_ieq_constr if hasattr(self._problem, "n_ieq_constr") else 0,
            n_eq_constr=self._problem.n_eq_constr if hasattr(self._problem, "n_eq_constr") else 0,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        out["F"] = self._problem.objective(x)  # objective value
        if hasattr(self._problem, "constraint"):
            out["H"] = self._problem.constraint(x)  # equality constraint value


class ModifiedObjective(Problem):
    """Modified objective function based on the following paper:

    Ishibuchi, H.; Matsumoto, T.; Masuyama, N.; Nojima, Y.
    Effects of dominance resistant solutions on the performance of evolutionary multi-objective
    and many-objective algorithms. In Proceedings of the Genetic and Evolutionary Computation
    Conference (GECCO '20), Cancún, Mexico, 8-12 July 2020.
    """

    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self._alpha = 0.02
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            xl=problem.xl,
            xu=problem.xu,
            n_ieq_constr=self._problem.n_ieq_constr if hasattr(self._problem, "n_ieq_constr") else 0,
            n_eq_constr=self._problem.n_eq_constr if hasattr(self._problem, "n_eq_constr") else 0,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        self._problem._evaluate(x, out, *args, **kwargs)
        F = out["F"]
        out["F"] = (1 - self._alpha) * F + self._alpha * np.tile(
            F.sum(axis=1).reshape(-1, 1), (1, self.n_obj)
        ) / self.n_obj

    # def pareto_front(self, *args, **kwargs):
    # return self._problem.pareto_front(*args, **kwargs)


def minimize(
    problem, algorithm, termination=None, copy_algorithm=True, copy_termination=True, run_id=None, **kwargs
):
    data = []
    columns = (
        [f"x{i}" for i in range(1, problem.n_var + 1)]
        + [f"f{i}" for i in range(1, problem.n_obj + 1)]
        + [f"h{i}" for i in range(1, problem.n_eq_constr + 1)]
        + [f"g{i}" for i in range(1, problem.n_ieq_constr + 1)]
    )

    # create a copy of the algorithm object to ensure no side-effects
    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        if termination is not None:
            if copy_termination:
                termination = copy.deepcopy(termination)

            kwargs["termination"] = termination

        algorithm.setup(problem, **kwargs)

    # actually execute the algorithm
    k = 1
    while algorithm.has_next():
        algorithm.next()
        pop = copy.deepcopy(algorithm.pop)
        if algorithm.n_gen == k + 1:
            df = pd.DataFrame(pop_to_numpy(pop), columns=columns)
            df.insert(0, "iteration", k)
            data.append(df)
            k += 1
    res = algorithm.result()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm
    data = pd.concat(data, axis=0)
    if run_id is not None:
        data.insert(0, "run", run_id)

    pareto_front = problem.pareto_front(1000)
    # gd_func = GenerationalDistance(pareto_front)
    # igd_func = InvertedGenerationalDistance(pareto_front)
    return data
    # return np.array([igd_func.compute(Y=res.F), gd_func.compute(Y=res.F), hypervolume(res.F, ref_point)])


def get_algorithm(n_objective: int, algorithm_name: str):
    if algorithm_name == "NSGA-II":
        algorithm = NSGA2(pop_size=50)
    elif algorithm_name == "NSGA-III":
        # create the reference directions to be used for the optimization
        if n_objective == 2:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        elif n_objective == 3:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
        elif n_objective == 4:
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=11)
        algorithm = NSGA3(pop_size=400, ref_dirs=ref_dirs)

    elif algorithm_name == "MOEAD":
        # the reference points are set to make the population size ~100
        if n_objective == 2:
            ref_dirs = get_reference_directions("uniform", 2, n_partitions=99)
        elif n_objective == 3:
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=13)
        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
        )
    elif algorithm_name == "SMS-EMOA":
        algorithm = SMSEMOA(pop_size=100)
    if 11 < 2:
        algorithm = AdaptiveEpsilonConstraintHandling(algorithm, perc_eps_until=0.8)
    return algorithm


N = 30
problem = sys.argv[1]
for problem_name in [
    problem,
]:  # "zdt2", "zdt3", "zdt4", "zdt6"]:
    # for problem_name in [f"dtlz{i}" for i in range(1, 8)]:
    # problem_name = problem.__class__.__name__
    print(problem_name)
    # problem = ModifiedObjective(get_problem(problem_name))
    # problem = ModifiedObjective(ProblemWrapper(problem))
    # problem = ProblemWrapper(problem)
    problem = get_problem(problem_name)
    termination = get_termination("n_gen", 3000)

    for algorithm_name in ("NSGA-II",):
        algorithm = get_algorithm(problem.n_obj, algorithm_name)
        # data = minimize(problem, algorithm, termination, run_id=1, seed=1, verbose=True)
        data = Parallel(n_jobs=N)(
            delayed(minimize)(problem, algorithm, termination, run_id=i + 1, seed=i + 1, verbose=False)
            for i in range(N)
        )
        df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV"])
        df.to_csv(f"{problem_name}-NSGA-II.csv", index=False)
        # data = pd.concat(data, axis=0)
        # data.to_csv(f"./data/{problem_name.upper()}_{algorithm_name}.csv", index=False)
        # data.to_csv(f"./data/CONV4_{algorithm_name}.csv", index=False)
        # data.to_csv(f"./data/{problem_name}_{algorithm_name}.csv", index=False)
        # save to Matlab's data format
        # mdic = {"data": data.values, "columns": data.columns.values}
        # savemat(f"./data/{problem_name.upper()}_{algorithm_name}.mat", mdic)
