import numpy as np
import pytest
import jax
import jax.numpy as jnp

from hvd.problems import CMOP, WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9


WFG_PROBLEMS = (WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9)


@pytest.mark.parametrize("problem_type", WFG_PROBLEMS)
def test_wfg_objectives_match_pymoo(problem_type: type[CMOP]) -> None:
    import pymoo.problems.many.wfg as pymoo_wfg

    problem = problem_type(n_var=6, n_obj=3)
    pymoo_problem_type = getattr(pymoo_wfg, problem_type.__name__)
    pymoo_problem = pymoo_problem_type(n_var=6, n_obj=3)
    x = np.linspace(0.17, 0.73, problem.n_var) * problem.xu

    np.testing.assert_allclose(
        problem.objective(x),
        pymoo_problem.evaluate(x, return_values_of=["F"]),
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.parametrize("problem_type", WFG_PROBLEMS)
def test_wfg_supports_ad_and_batched_derivatives(problem_type: type[CMOP]) -> None:
    problem = problem_type(n_var=6, n_obj=3)
    x = np.linspace(0.17, 0.73, problem.n_var) * problem.xu
    population = np.stack((x, 0.8 * x))

    assert isinstance(problem, CMOP)
    assert problem.n_eq_constr == 0
    assert problem.n_ieq_constr == 0
    assert np.all(np.isfinite(problem.objective_jacobian(x)))
    assert np.all(np.isfinite(problem.objective_hessian(x)))
    np.testing.assert_allclose(
        problem.objective(population),
        np.stack([problem.objective(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.objective_jacobian(population),
        np.stack([problem.objective_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.objective_hessian(population),
        np.stack([problem.objective_hessian(row) for row in population]),
    )


def test_wfg_can_add_box_constraints() -> None:
    problem = WFG1(n_var=6, n_obj=3, boundary_constraints=True)
    x = (problem.xl + problem.xu) / 2

    assert problem.n_ieq_constr == 2 * problem.n_var
    assert np.all(problem.ieq_constraint(x) <= 0)


@pytest.mark.parametrize("problem_type", WFG_PROBLEMS)
def test_wfg_population_and_pymoo_adapter_match(problem_type: type[CMOP]) -> None:
    import pymoo.problems.many.wfg as pymoo_wfg

    problem = problem_type(n_var=10, n_obj=3)
    population = np.random.default_rng(66).uniform(0.05, 0.95, (32, 10)) * problem.xu
    independent = getattr(pymoo_wfg, problem_type.__name__)(n_var=10, n_obj=3)
    values = problem.objective(population)

    np.testing.assert_allclose(values, independent.evaluate(population), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(values, problem.as_pymoo_problem().evaluate(population))
    np.testing.assert_allclose(values, np.stack([problem.objective(x) for x in population]), atol=1e-12)


def test_wfg1_pareto_values_and_derivatives_at_zero_distance() -> None:
    # Powers of two make normalization exact, isolating the mathematical
    # zero-distance front from pymoo's rounding residual with bounds 2, 4, ... .
    problem = WFG1(n_var=10, n_obj=3, xu=2.**np.arange(1, 11))
    # Equal transformed position coordinates within each group make the
    # weighted reductions p and q, giving an independently known front point.
    pq = np.array([[0., 0.], [0.25, 0.7], [0.6, 0.2], [1., 1.]])
    positions = np.repeat(pq**50, 2, axis=1)
    population = np.column_stack((positions, np.full((4, 6), 0.35))) * problem.xu
    p, q = pq.T
    expected = np.column_stack((
        2 * (1 - np.cos(np.pi * p / 2)) * (1 - np.cos(np.pi * q / 2)),
        4 * (1 - np.cos(np.pi * p / 2)) * (1 - np.sin(np.pi * q / 2)),
        6 * (1 - p + np.sin(10 * np.pi * p) / (10 * np.pi)),
    ))

    np.testing.assert_allclose(problem.objective(population), expected, atol=1e-14)
    jacobian = problem.objective_jacobian(population)
    hessian = problem.objective_hessian(population)
    assert np.isfinite(jacobian).all()
    assert np.isfinite(hessian).all()
    np.testing.assert_array_equal(jacobian[:, :, problem.k:], 0.)
    np.testing.assert_array_equal(hessian[:, :, problem.k:, :], 0.)
    np.testing.assert_array_equal(hessian[:, :, :, problem.k:], 0.)


def test_wfg1_matches_pymoo_at_represented_pareto_points() -> None:
    from pymoo.problems.many.wfg import WFG1 as PymooWFG1

    problem = WFG1(n_var=10, n_obj=3)
    population = problem.get_pareto_set(32)
    expected = PymooWFG1(n_var=10, n_obj=3).evaluate(population)

    np.testing.assert_allclose(problem.objective(population), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(problem.objective_jacobian(population)[:, :, problem.k:], 0.)
    assert np.isfinite(problem.objective_hessian(population)).all()


def test_wfg1_jit_is_stable_near_pareto_set() -> None:
    problem = WFG1(n_var=10, n_obj=3)
    # NSGA-III run 1, row 233: the old JIT expression changed each objective
    # by about 0.109 through roundoff before the fractional power.
    x = np.array([
        6.867216947826e-4, 1.5177170622393e-3, 2.144198863719e-4, 1.54733241045317e-12,
        3.50000009323521, 4.20000000146688, 4.9, 5.6, 6.30000000001432, 7.,
    ])
    with jax.disable_jit():
        expected = problem.objective(x)

    np.testing.assert_allclose(problem.objective(x), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(problem.objective(np.stack((x, x))), [expected, expected], atol=1e-14)
    assert np.isfinite(problem.objective_jacobian(x)).all()
    assert np.isfinite(problem.objective_hessian(x)).all()


def test_wfg1_power_derivative_bound_and_second_derivative() -> None:
    from hvd.problems.wfg import _bias_poly

    limit = 1e6
    power = lambda y: _bias_poly(y, 0.02, limit)
    values = jnp.array([0., 1e-250, 1e-14, 1e-6, 0.1, 1.])
    gradient = jax.jit(jax.vmap(jax.grad(power)))(values)
    hessian = jax.jit(jax.vmap(jax.grad(jax.grad(power))))(values)

    np.testing.assert_allclose(jax.vmap(power)(values), np.asarray(values)**0.02)
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    assert gradient[0] == hessian[0] == 0
    np.testing.assert_allclose(gradient[1:3], limit)
    np.testing.assert_array_equal(hessian[1:3], 0.)
    np.testing.assert_allclose(gradient[3:], 0.02 * values[3:]**(-0.98))
    np.testing.assert_allclose(hessian[3:], -0.0196 * values[3:]**(-1.98))


def test_wfg1_derivative_bound_does_not_change_objective_values() -> None:
    bounded = WFG1(n_var=10, power_gradient_limit=100.)
    exact = WFG1(n_var=10, power_gradient_limit=None)
    population = bounded.get_pareto_set(32)

    np.testing.assert_array_equal(bounded.objective(population), exact.objective(population))
    assert np.isfinite(bounded.objective_jacobian(population)).all()
    assert not np.isfinite(exact.objective_jacobian(population)).all()


@pytest.mark.parametrize("limit", [-1., 0., np.inf, np.nan])
def test_wfg1_rejects_invalid_derivative_bound(limit: float) -> None:
    with pytest.raises(ValueError, match="power_gradient_limit"):
        WFG1(power_gradient_limit=limit)
