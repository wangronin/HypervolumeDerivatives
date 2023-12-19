import copy
import sys

sys.path.insert(0, "./")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.problems.base import MOOAnalytical

# ref_point = np.array([11, 11])


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
    res = algorithm.result()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm
    if problem.n_obj == 3:
        if problem_name not in ["DTLZ4", "DTLZ6", "DTLZ7"]:
            ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=30).do()
            pareto_front = problem.pareto_front(ref_dirs)
        else:
            pareto_front = problem.pareto_front()
    else:
        pareto_front = problem.pareto_front(1000)
    gd_value = GenerationalDistance(pareto_front).compute(Y=res.F)
    igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=res.F)
    # return np.array([igd_value, gd_value, hypervolume(res.F, ref_point)])
    # return np.array([igd_value, gd_value, np.sum(np.bitwise_or(res.X < 0, res.X > 1))])
    return np.array([igd_value, gd_value])


def get_algorithm(
    n_objective: int, algorithm_name: str, pop_size: int, constrained: bool
) -> GeneticAlgorithm:
    if algorithm_name == "NSGA-II":
        algorithm = NSGA2(pop_size=pop_size)
    elif algorithm_name == "NSGA-III":
        # create the reference directions to be used for the optimization
        if n_objective == 2:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        elif n_objective == 3:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
        elif n_objective == 4:
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=11)
        algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    elif algorithm_name == "MOEAD":
        # the reference points are set to make the population size ~100
        if n_objective == 2:
            ref_dirs = get_reference_directions("uniform", 2, n_partitions=99)
        elif n_objective == 3:
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=23)
        algorithm = MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    elif algorithm_name == "SMS-EMOA":
        algorithm = SMSEMOA(pop_size=pop_size)

    if constrained:
        if algorithm_name != "MOEAD":
            algorithm = AdaptiveEpsilonConstraintHandling(algorithm, perc_eps_until=0.8)
    return algorithm


def get_Jacobian_calls(path, problem_name, algorithm_name, gen):
    return int(np.median(pd.read_csv(f"{path}/{problem_name}-DpN-{algorithm_name}-{gen}.csv").Jac_calls))


n_iter_newton = 5
gen = 300
gen_func = lambda n_var, scale: 4 * scale + 10 * n_var
N = 30
problem = sys.argv[1]

for problem_name in [problem]:
    print(problem_name)
    problem = get_problem(problem_name)
    pop_size = 100 if problem.n_obj == 2 else 300
    constrained = problem.n_eq_constr > 0 or problem.n_ieq_constr > 0

    for algorithm_name in ("NSGA-II",):
        scale = int(
            get_Jacobian_calls("./results", problem_name, algorithm_name, gen) / pop_size / n_iter_newton
        )
        termination = get_termination("n_gen", gen + n_iter_newton * gen_func(problem.n_var, scale))
        algorithm = get_algorithm(problem.n_obj, algorithm_name, pop_size, constrained)
        data = Parallel(n_jobs=N)(
            delayed(minimize)(problem, algorithm, termination, run_id=i + 1, seed=i + 1, verbose=False)
            for i in range(N)
        )
        # df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV"])
        df = pd.DataFrame(np.array(data), columns=["IGD", "GD"])
        df.to_csv(f"{problem_name}-{algorithm_name}-{gen}.csv", index=False)
        # data = pd.concat(data, axis=0)
        # data.to_csv(f"./data/{problem_name.upper()}_{algorithm_name}.csv", index=False)
        # data.to_csv(f"./data/CONV4_{algorithm_name}.csv", index=False)
        # data.to_csv(f"./data/{problem_name}_{algorithm_name}.csv", index=False)
        # save to Matlab's data format
        # mdic = {"data": data.values, "columns": data.columns.values}
        # savemat(f"./data/{problem_name.upper()}_{algorithm_name}.mat", mdic)
