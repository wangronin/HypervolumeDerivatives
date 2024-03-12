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
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import Problem as PymooProblem
from pymoo.problems.many import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from pymoo.problems.multi import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from scipy.io import savemat

from hvd.problems import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10, IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4
from hvd.problems.base import PymooProblemWrapper

# NOTE: this is a slightly faster implementation of SMS-EMOA
from hvd.sms_emoa import SMSEMOA

pop_to_numpy = lambda pop: np.array([np.r_[ind.X, ind.F, ind.H, ind.G] for ind in pop])
data_path = "./"


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
    return data


def get_algorithm(n_objective: int, algorithm_name: str, constrained: bool) -> GeneticAlgorithm:
    if n_objective == 2:
        pop_size = 100
    elif n_objective == 3:
        pop_size = 300
    elif n_objective == 4:
        pop_size = 600

    if algorithm_name == "NSGA-II":
        algorithm = NSGA2(pop_size=pop_size)
    elif algorithm_name == "NSGA-III":
        # create the reference directions to be used for the optimization
        if n_objective == 2:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1)
        elif n_objective == 3:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=23)
        elif n_objective == 4:
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=13)
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


N = 30
problems = [
    # CF1(),
    # CF2(),
    # CF3(),
    # CF4(),
    # CF5(),
    # CF6(),
    # CF7(),
    # CF8(),
    # CF9(),
    # CF10(),
    # ZDT1(),
    # ZDT2(),
    # ZDT3(),
    # ZDT4(),
    # ZDT6(),
    # DTLZ1(),
    # DTLZ2(),
    # DTLZ3(),
    # DTLZ4(),
    # DTLZ5(),
    # DTLZ6(),
    # DTLZ7(),
    # Eq1IDTLZ1(),
    # Eq1IDTLZ2(),
    # Eq1IDTLZ3(),
    # Eq1IDTLZ4(),
    IDTLZ1(),
    IDTLZ2(),
    IDTLZ3(),
    IDTLZ4(),
]

# idx = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
# for problem in [problems[idx]]:
for problem in problems:
    problem_name = problem.__class__.__name__
    print(problem_name)
    problem = problem if isinstance(problem, PymooProblem) else PymooProblemWrapper(problem)
    termination = get_termination("n_gen", 600)
    constrained = (hasattr(problem, "n_eq_constr") and problem.n_eq_constr > 0) or (
        hasattr(problem, "n_ieq_constr") and problem.n_ieq_constr > 0
    )
    for algorithm_name in ["NSGA-II"]:
        algorithm = get_algorithm(problem.n_obj, algorithm_name, constrained)
        # data = minimize(problem, algorithm, termination, run_id=1, seed=1, verbose=True)
        data = Parallel(n_jobs=N)(
            delayed(minimize)(problem, algorithm, termination, run_id=i + 1, seed=i + 1, verbose=False)
            for i in range(N)
        )
        data = pd.concat(data, axis=0)
        # save to Matlab's data format
        mdic = {"data": data.values, "columns": data.columns.values}
        savemat(f"{data_path}/{problem_name.upper()}_{algorithm_name}.mat", mdic)
        # save to CSV
        # data.to_csv(f"./data/{problem_name.upper()}_{algorithm_name}.csv", index=False)
