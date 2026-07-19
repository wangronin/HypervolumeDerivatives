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
        problem.objective_batch(population),
        np.stack([problem.objective(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.objective_jacobian_batch(population),
        np.stack([problem.objective_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.objective_hessian_batch(population),
        np.stack([problem.objective_hessian(row) for row in population]),
    )


def test_wfg_can_add_box_constraints() -> None:
    problem = WFG1(n_var=6, n_obj=3, boundary_constraints=True)
    x = (problem.xl + problem.xu) / 2

    assert problem.n_ieq_constr == 2 * problem.n_var
    assert np.all(problem.ieq_constraint(x) <= 0)
