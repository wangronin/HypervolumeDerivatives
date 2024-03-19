import copy
import sys

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import Problem as PymooProblem
from pymoo.problems.many import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from pymoo.problems.multi import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.problems import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10, IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4
from hvd.problems.base import CONV42F, MOOAnalytical, PymooProblemWrapper
from hvd.sms_emoa import SMSEMOA

pop_to_numpy = lambda pop: np.array([ind.F for ind in pop])


def minimize(
    problem,
    algorithm,
    algorithm_name,
    termination=None,
    copy_algorithm=True,
    copy_termination=True,
    run_id=None,
    **kwargs,
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
    if problem.name().startswith("DTLZ"):
        if problem_name in ["DTLZ5", "DTLZ6", "DTLZ7"]:
            pareto_front = problem.pareto_front()
        else:
            ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=30).do()
            pareto_front = problem.pareto_front(ref_dirs)
    else:
        pareto_front = problem.pareto_front(1000)
    # TODO: ad-hoc solution for `res.F` being `None`. Figure out why..
    if res.F is None:
        igd_value, gd_value = np.nan, np.nan
    else:
        gd_value = GenerationalDistance(pareto_front).compute(Y=res.F)
        igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=res.F)
    return np.array([igd_value, gd_value])


def get_algorithm(
    n_objective: int, algorithm_name: str, pop_size: int, constrained: bool
) -> GeneticAlgorithm:
    if algorithm_name == "NSGA-II":
        algorithm = NSGA2(pop_size=pop_size)
    elif algorithm_name == "NSGA-III":
        # create the reference directions to be used for the optimization
        if n_objective == 2:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 20)
        elif n_objective == 3:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
        elif n_objective == 4:
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=11)
        algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    elif algorithm_name == "MOEAD":
        # the reference points are set to make the population size ~100
        if n_objective == 2:
            ref_dirs = get_reference_directions("uniform", 2, n_partitions=80)
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


n_iter_newton = 6
gen = 300
# NOTE: the following running budget is estimated with upper bounds of AD's theory
# gen_func = lambda n_var, scale: 4 * scale + 10 * n_var
# NOTE: 1.836 is obtained on ZDTs
gen_func = lambda n_var, scale: int(1.836 * scale + 3)
N = 30
problem_names = sys.argv[1]

for problem_name in [problem_names]:
    print(problem_name)
    problem = locals()[problem_name]()
    problem = problem if isinstance(problem, PymooProblem) else PymooProblemWrapper(problem)
    pop_size = 100 if problem.n_obj == 2 else 300
    constrained = (hasattr(problem, "n_eq_constr") and problem.n_eq_constr > 0) or (
        hasattr(problem, "n_ieq_constr") and problem.n_ieq_constr > 0
    )
    for algorithm_name in ("MOEAD",):
        scale = int(
            get_Jacobian_calls("./results", problem_name, algorithm_name, gen) / pop_size / n_iter_newton
        )
        termination = get_termination("n_gen", gen + n_iter_newton * gen_func(problem.n_var, scale))
        algorithm = get_algorithm(problem.n_obj, algorithm_name, pop_size, constrained)
        # minimize(problem, algorithm, algorithm_name, termination, run_id=1, seed=1, verbose=True)
        data = Parallel(n_jobs=N)(
            delayed(minimize)(
                problem, algorithm, algorithm_name, termination, run_id=i + 1, seed=i + 1, verbose=False
            )
            for i in range(N)
        )
        # df = pd.DataFrame(np.array(data), columns=["IGD", "GD", "HV"])
        data = np.array(data)
        data = data[~np.any(np.isnan(data), axis=1)]
        df = pd.DataFrame(data, columns=["IGD", "GD"])
        df.to_csv(f"results/{problem_name}-{algorithm_name}-{gen}.csv", index=False)
        # data = pd.concat(data, axis=0)
        # data.to_csv(f"./data/{problem_name.upper()}_{algorithm_name}.csv", index=False)
        # data.to_csv(f"./data/CONV4_{algorithm_name}.csv", index=False)
        # data.to_csv(f"./data/{problem_name}_{algorithm_name}.csv", index=False)
        # save to Matlab's data format
        # mdic = {"data": data.values, "columns": data.columns.values}
        # savemat(f"./data/{problem_name.upper()}_{algorithm_name}.mat", mdic)
