import copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

pop_to_numpy = lambda pop: np.array([np.r_[ind.X, ind.F] for ind in pop])


def minimize(
    problem, algorithm, termination=None, copy_algorithm=True, copy_termination=True, run_id=None, **kwargs
):
    data = []
    columns = [f"x{i}" for i in range(1, problem.n_var + 1)] + [f"f{i}" for i in range(1, problem.n_obj + 1)]
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


def get_algorithm(n_objective: int, algorithm_name: str):
    if algorithm_name == "NSGA-III":
        # create the reference directions to be used for the optimization
        if n_objective == 2:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        elif n_objective == 3:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)

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
    return algorithm


N = 15
for problem_name in ["zdt1", "zdt2"]:
    problem = get_problem(problem_name)
    termination = get_termination("n_gen", 500)

    for algorithm_name in ("SMS-EMOA", "MOEAD"):
        algorithm = get_algorithm(problem.n_obj, algorithm_name)

        data = Parallel(n_jobs=N)(
            delayed(minimize)(problem, algorithm, termination, run_id=i, seed=i, verbose=False)
            for i in range(N)
        )
        data = pd.concat(data, axis=0)
        data.to_csv(f"./data/{problem_name.upper()}_{algorithm_name}.csv", index=False)
