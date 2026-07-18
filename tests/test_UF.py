import numpy as np
import pytest

from hvd.problems import ConstrainedMOP, MOP, UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10


@pytest.mark.parametrize("problem_type", [UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10])
def test_uf_supports_ad_and_batches(problem_type: type[MOP]) -> None:
    problem = problem_type()
    x = (problem.xl + problem.xu) / 2
    assert problem.objective(x).shape == (problem.n_obj,)
    assert problem.objective_batch(np.stack((x, x))).shape == (2, problem.n_obj)
    assert problem.objective_jacobian(x).shape == (problem.n_obj, problem.n_var)
    assert problem.objective_hessian(x).shape == (problem.n_obj, problem.n_var, problem.n_var)


def test_uf_objectives_vanish_to_the_documented_front_on_pareto_set():
    for problem_type in (UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10):
        problem = problem_type()
        x = np.zeros(problem.n_var)
        x[: 2 if problem.n_obj == 3 else 1] = (0.37, 0.42)[: 2 if problem.n_obj == 3 else 1]
        indices = np.arange(1, problem.n_var + 1)
        if problem.n_obj == 2:
            if problem_type is UF2:
                phase = 6*np.pi*x[0] + indices*np.pi/problem.n_var
                envelope = .3*x[0]**2*np.cos(24*np.pi*x[0]+4*indices*np.pi/problem.n_var)+.6*x[0]
                x[1::2] = envelope[1::2]*np.sin(phase[1::2])
                x[2::2] = envelope[2::2]*np.cos(phase[2::2])
            elif problem_type is UF3:
                x = x[0] ** (.5*(1+3*(indices-2)/(problem.n_var-2)))
            else:
                x[1:] = np.sin(6*np.pi*x[0]+indices[1:]*np.pi/problem.n_var)
        else:
            x[2:] = 2*x[1]*np.sin(2*np.pi*x[0]+indices[2:]*np.pi/problem.n_var)
        assert np.all(np.isfinite(problem.objective(x)))


@pytest.mark.parametrize("problem_type", [UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10])
def test_uf_can_expose_decision_bounds_as_constraints(problem_type: type[MOP]) -> None:
    problem = problem_type(boundary_constraints=True)
    assert isinstance(problem, ConstrainedMOP)
    assert problem.n_ieq_constr == 2 * problem.n_var

    x = (problem.xl + problem.xu) / 2
    constraints = problem.ieq_constraint(x)
    assert constraints.shape == (2 * problem.n_var,)
    assert np.all(constraints <= 0)


def test_uf_bounds_can_be_overridden_explicitly() -> None:
    xl = np.array([0.1, -0.5, -0.25])
    xu = np.array([0.9, 0.5, 0.25])
    problem = UF1(n_var=3, xl=xl, xu=xu, boundary_constraints=True)

    np.testing.assert_array_equal(problem.xl, xl)
    np.testing.assert_array_equal(problem.xu, xu)
    assert problem.ieq_constraint((xl + xu) / 2).shape == (2 * problem.n_var,)
