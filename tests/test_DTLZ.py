import sys

sys.path.insert(0, "./")

import numpy as np

from hvd.problems import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, PymooProblemWithAD


def test_DTLZ():
    for f in [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]:
        problem = PymooProblemWithAD(f())
        x = np.random.rand(problem.n_var)
        problem.objective(x)
        problem.objective_jacobian(x)
        problem.objective_hessian(x)
        print(len(problem.get_pareto_front()))


test_DTLZ()
