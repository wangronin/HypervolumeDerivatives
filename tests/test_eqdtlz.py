import numpy as np
import pytest

from hvd import problems


@pytest.mark.parametrize("family", ["DTLZ", "IDTLZ"])
@pytest.mark.parametrize("number", range(1, 5))
@pytest.mark.parametrize("n_obj", [2, 3, 4])
def test_eq1_pareto_set_intersects_parent_set_and_equality(family, number, n_obj):
    parent = getattr(problems, f"{family}{number}")(n_var=8, n_obj=n_obj)
    problem = getattr(problems, f"Eq1{family}{number}")(n_var=8, n_obj=n_obj)
    population = problem.get_pareto_set(32)

    assert population.shape == (2 if n_obj == 2 else 32, problem.n_var)
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
