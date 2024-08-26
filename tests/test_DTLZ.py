import sys

sys.path.insert(0, "./")

import numpy as np
import pandas as pd
import pymoo.problems.many as many
import pytest

from hvd.problems import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    IDTLZ1,
    IDTLZ2,
    IDTLZ3,
    IDTLZ4,
    Eq1DTLZ1,
    Eq1DTLZ2,
    Eq1DTLZ3,
    Eq1DTLZ4,
    Eq1IDTLZ1,
    Eq1IDTLZ2,
    Eq1IDTLZ3,
    Eq1IDTLZ4,
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
    with pytest.raises(Exception):
        problem.eq_constraint(x)
    with pytest.raises(Exception):
        problem.ieq_constraint(x)


@pytest.mark.parametrize("F", [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7])
def test_DTLZ_with_boundary_constraints(F):
    problem = F(boundry_constraints=True)
    x = np.random.rand(problem.n_var)
    problem.objective(x)
    problem.objective_jacobian(x)
    problem.objective_hessian(x)
    with pytest.raises(Exception):
        problem.eq_constraint(x)
    assert problem.n_eq_constr == 0
    assert problem.n_ieq_constr == problem.n_var * 2
    problem.ieq_constraint(x)
    problem.ieq_jacobian(x)
    problem.ieq_hessian(x)
    problem.get_pareto_front()


@pytest.mark.parametrize("F", [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7])
def test_DTLZ_function_value(F):
    problem = F()
    F_pymoo = getattr(many, F.__name__)
    problem_pymoo = F_pymoo(n_var=problem.n_var, n_obj=problem.n_obj)
    for _ in range(10):
        x = np.random.rand(problem.n_var)
        assert all(np.isclose(problem.objective(x), problem_pymoo.evaluate(x)))


@pytest.mark.parametrize("F", [IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4])
def test_IDTLZ(F):
    problem = F(boundry_constraints=False)
    x = np.random.rand(problem.n_var)
    problem.objective(x)
    problem.objective_jacobian(x)
    problem.objective_hessian(x)
    with pytest.raises(Exception):
        problem.eq_constraint(x)
        problem.ieq_constraint(x)
    assert problem.n_eq_constr == 0
    assert problem.n_ieq_constr == 0
    problem.get_pareto_front()


input = pd.read_csv("tests/IDTLZ_test_input.csv", index_col=None, header=None).values
expected = [
    [409.45442476, 394.24468018, 317.53653453],
    [-0.02013554, 0.33894782, -0.51014466],
    [382.79525969, 241.31844776, -483.31935897],
    [-0.85517519, 0.85517181, 0.85517519],
]


@pytest.mark.parametrize("F, input, expected", zip([IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4], input, expected))
def test_IDTLZ_function_value(F, input, expected):
    problem = F()
    assert all(np.isclose(problem.objective(np.array(input)), expected))


@pytest.mark.parametrize(
    "F, input, expected, cstr",
    zip(
        [Eq1IDTLZ1, Eq1IDTLZ2, Eq1IDTLZ3, Eq1IDTLZ4],
        input,
        expected,
        [0.15491684, 0.12304637, 0.03234342, 0.01905571],
    ),
)
def test_Eq1IDTLZ_function_value(F, input, expected, cstr):
    problem = F()
    assert all(np.isclose(problem.objective(np.array(input)), expected))
    assert np.isclose(problem.eq_constraint(np.array(input)), cstr)


@pytest.mark.parametrize(
    "problem, expected, cstr",
    [
        (Eq1DTLZ1, [39.2583506542203, 363.3913996454754, 234.0513096302470], 0.01943601),
        (Eq1DTLZ2, [1.092253942326640, 0.168601869538367, 1.696393580953341], 0.01943601),
        (Eq1DTLZ3, [686.973245563538, 106.042165687341, 1066.946942373372], 0.01943601),
        (Eq1DTLZ4, [2.024647240000000, 0, 0], 0.01943601),
    ],
)
def test_Eq1DTLZ_function_value(problem, expected, cstr):
    f = problem(n_var=10)
    x = np.array([0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649, 0.1576, 0.9706, 0.9572, 0.4854])
    assert np.all(np.isclose(f.objective(x), expected))
    assert np.isclose(f.eq_constraint(x), cstr)
