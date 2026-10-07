import numpy as np
import pytest

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
