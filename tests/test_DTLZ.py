import sys

sys.path.insert(0, "./")

import numpy as np
import pytest

from hvd.problems import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    get_pymoo_problem,
)

np.random.seed(42)


@pytest.mark.parametrize("F", [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7])
def test_DTLZ(F):
    problem = F()
    x = np.random.rand(problem.n_var)
    problem.objective(x)
    problem.objective_jacobian(x)
    problem.objective_hessian(x)
    problem.get_pareto_front()
    assert problem.eq_constraint(x) is None
    assert problem.ieq_constraint(x) is None


@pytest.mark.parametrize("F", [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7])
def test_DTLZ_with_boundary_constraints(F):
    problem = F(boundary_constraints=True)
    x = np.random.rand(problem.n_var)
    problem.objective(x)
    problem.objective_jacobian(x)
    problem.objective_hessian(x)
    assert problem.eq_constraint(x) is None
    assert problem.n_eq_constr == 0
    assert problem.n_ieq_constr == problem.n_var * 2
    problem.ieq_constraint(x)
    problem.ieq_jacobian(x)
    problem.ieq_hessian(x)
    problem.get_pareto_front()


@pytest.mark.parametrize("F", [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7])
def test_DTLZ_function_value(F):
    problem = F()
    problem_pymoo = get_pymoo_problem(
        F.__name__,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
    )
    for index in range(10):
        x = np.random.rand(problem.n_var)
        np.testing.assert_allclose(problem.objective(x), problem_pymoo.objective(x))
        if index == 0:
            np.testing.assert_allclose(
                problem.objective_jacobian(x),
                problem_pymoo.objective_jacobian(x),
            )
            np.testing.assert_allclose(
                problem.objective_hessian(x),
                problem_pymoo.objective_hessian(x),
                rtol=1e-6,
                atol=1e-8,
            )
