import numpy as np
import pandas as pd
import pytest

from hvd import problems
from hvd.problems import (
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

# @pytest.mark.parametrize("F", [IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4])
# def test_IDTLZ(F):
#     problem = F()
#     x = np.random.rand(problem.n_var)
#     problem.objective(x)
#     problem.objective_jacobian(x)
#     problem.objective_hessian(x)
#     assert problem.eq_constraint(x) is None
#     assert problem.n_eq_constr == 0
#     assert problem.n_ieq_constr == 0
#     assert problem.ieq_constraint(x) is None
#     problem.get_pareto_front()


input = np.array(
    [
        [
            0.56640464,
            0.4760504,
            0.66367117,
            0.93682974,
            0.7325721,
            0.21494038,
            0.03118314,
            0.26226404,
            0.59507793,
            0.05142581,
            0.49636625,
        ],
        [
            0.59684285,
            0.33424389,
            0.7709122,
            0.10659825,
            0.07513778,
            0.72818876,
            0.49549132,
            0.6884024,
            0.43482734,
            0.24640203,
            0.81910232,
        ],
        [
            0.79941588,
            0.69469647,
            0.27214514,
            0.59023067,
            0.3609739,
            0.09158207,
            0.91731358,
            0.13681863,
            0.95023735,
            0.44600577,
            0.18513293,
        ],
        [
            0.54190095,
            0.87294584,
            0.73222489,
            0.80656115,
            0.65878337,
            0.69227656,
            0.84919565,
            0.24966801,
            0.48942496,
            0.22120944,
            0.98766801,
        ],
    ]
)
expected = [
    [409.45442476, 394.24468018, 317.53653453],
    [-0.02013554, 0.33894782, -0.51014466],
    [382.79525969, 241.31844776, -483.31935897],
    [-0.85517519, 0.85517181, 0.85517519],
]


# @pytest.mark.parametrize("F, input, expected", zip([IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4], input, expected))
# def test_IDTLZ_function_value(F, input, expected):
#     problem = F()
#     assert all(np.isclose(problem.objective(np.array(input)), expected))


# @pytest.mark.parametrize(
#     "F, input, expected, cstr",
#     zip(
#         [Eq1IDTLZ1, Eq1IDTLZ2, Eq1IDTLZ3, Eq1IDTLZ4],
#         input,
#         expected,
#         [0.15501684, 0.12314637, 0.03244342, 0.01915571],
#     ),
# )
# def test_Eq1IDTLZ_function_value(F, input, expected, cstr):
#     problem = F()
#     assert np.all(np.isclose(problem.objective(np.array(input)), expected))
#     assert np.isclose(problem.eq_constraint(np.array(input)), cstr)


@pytest.mark.parametrize(
    "problem, expected, cstr",
    [
        (Eq1DTLZ1, [39.2583506542203, 363.3913996454754, 234.0513096302470], 0.01953601),
        (Eq1DTLZ2, [1.092253942326640, 0.168601869538367, 1.696393580953341], 0.01953601),
        (Eq1DTLZ3, [686.973245563538, 106.042165687341, 1066.946942373372], 0.01953601),
        (Eq1DTLZ4, [2.024647240000000, 0, 0], 0.01953601),
    ],
)
def test_Eq1DTLZ_function_value(problem, expected, cstr):
    f = problem(n_var=10)
    x = np.array([0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649, 0.1576, 0.9706, 0.9572, 0.4854])
    assert np.all(np.isclose(f.objective(x), expected))
    assert np.isclose(f.eq_constraint(x), cstr)


def test_batched_equality_derivatives_match_single_evaluations() -> None:
    problem = Eq1DTLZ1(n_var=5)
    population = np.stack((np.full(5, 0.3), np.full(5, 0.6)))

    np.testing.assert_allclose(
        problem.eq_jacobian(population),
        np.stack([problem.eq_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.eq_hessian(population),
        np.stack([problem.eq_hessian(row) for row in population]),
    )


@pytest.mark.parametrize("family", ["DTLZ", "IDTLZ"])
@pytest.mark.parametrize("number", range(1, 5))
@pytest.mark.parametrize("n_obj", [3, 4])
def test_eq1_pareto_set_intersects_parent_set_and_equality(family, number, n_obj):
    parent = getattr(problems, f"{family}{number}")(n_var=8, n_obj=n_obj)
    problem = getattr(problems, f"Eq1{family}{number}")(n_var=8, n_obj=n_obj)
    population = problem.get_pareto_set(32)

    assert population.shape == (32, problem.n_var)
    assert np.all((population >= problem.xl) & (population <= problem.xu))
    np.testing.assert_array_equal(population[:, n_obj - 1 :], 0.5)
    np.testing.assert_allclose(problem.eq_constraint(population), 0.0, atol=2e-15)

    # Moving the constraint mixin first must preserve the selected objective,
    # including inversion, scale, and the power-100 transform in variant 4.
    np.testing.assert_allclose(problem.objective(population), parent.objective(population))
    np.testing.assert_allclose(problem.get_pareto_front(32), parent.objective(population))
    assert problem.alpha == parent.alpha
    assert problem.scale == parent.scale


@pytest.mark.parametrize("n_obj", [3, 4])
def test_eq1_pareto_sampling_is_reproducible_without_changing_global_rng(n_obj):
    problem = problems.Eq1DTLZ2(n_obj=n_obj)
    state = np.random.get_state()
    expected = problem.get_pareto_set(33)
    actual = problem.get_pareto_set(33)
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == 33
    for before, after in zip(state, np.random.get_state()):
        np.testing.assert_equal(before, after)
