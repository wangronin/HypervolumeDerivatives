import numpy as np
import pytest

from hvd.problems import CMOP, MOP, UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10


UF_PROBLEMS = (UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10)

# Values obtained by evaluating the CalObj formulas in the corresponding
# MATLAB UF1.m--UF10.m files at
# x = xl + linspace(0.05, 0.95, 30) * (xu - xl).
MATLAB_REFERENCE_VALUES = (
    (2.650111248344718, 3.618773252838321),
    (0.5762281539765921, 1.392457255841188),
    (2.797656514314896, 3.596210455539544),
    (0.19175350104478323, 1.124134142281501),
    (7.301778228241268, 8.83606018812196),
    (11.147604661281672, 12.997599953823043),
    (3.149391519997777, 3.293099778935241),
    (3.0165883415703116, 2.2064939119526366, 2.5993136960390135),
    (2.0317880625140448, 2.1569227845152876, 3.4398201175525482),
    (11.074591601532871, 10.41406115514193, 12.400931077741717),
)


@pytest.mark.parametrize("problem_type", UF_PROBLEMS)
def test_uf_supports_ad_and_batches(problem_type: type[MOP]) -> None:
    problem = problem_type(n_var=8)
    x = problem.xl + 0.37 * (problem.xu - problem.xl)
    second_x = problem.xl + 0.63 * (problem.xu - problem.xl)
    population = np.stack((x, second_x))

    assert problem.objective(x).shape == (problem.n_obj,)
    assert problem.objective_jacobian(x).shape == (problem.n_obj, problem.n_var)
    assert problem.objective_hessian(x).shape == (problem.n_obj, problem.n_var, problem.n_var)
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


@pytest.mark.parametrize(("problem_type", "expected"), zip(UF_PROBLEMS, MATLAB_REFERENCE_VALUES))
def test_uf_matches_matlab_reference_values(
    problem_type: type[MOP], expected: tuple[float, ...]
) -> None:
    problem = problem_type()
    fractions = np.linspace(0.05, 0.95, problem.n_var)
    x = problem.xl + fractions * (problem.xu - problem.xl)

    np.testing.assert_allclose(problem.objective(x), expected, rtol=1e-7, atol=1e-8)


def test_uf_objectives_vanish_to_the_documented_front_on_pareto_set():
    for problem_type in UF_PROBLEMS:
        problem = problem_type()
        x = np.zeros(problem.n_var)
        x[: 2 if problem.n_obj == 3 else 1] = (0.37, 0.42)[: 2 if problem.n_obj == 3 else 1]
        indices = np.arange(1, problem.n_var + 1)
        if problem.n_obj == 2:
            if problem_type is UF2:
                phase = 6 * np.pi * x[0] + indices * np.pi / problem.n_var
                envelope = (
                    0.3 * x[0] ** 2 * np.cos(24 * np.pi * x[0] + 4 * indices * np.pi / problem.n_var)
                    + 0.6 * x[0]
                )
                x[1::2] = envelope[1::2] * np.sin(phase[1::2])
                x[2::2] = envelope[2::2] * np.cos(phase[2::2])
            elif problem_type is UF3:
                x = x[0] ** (0.5 * (1 + 3 * (indices - 2) / (problem.n_var - 2)))
            else:
                x[1:] = np.sin(6 * np.pi * x[0] + indices[1:] * np.pi / problem.n_var)
        else:
            x[2:] = 2 * x[1] * np.sin(2 * np.pi * x[0] + indices[2:] * np.pi / problem.n_var)
        assert np.all(np.isfinite(problem.objective(x)))


@pytest.mark.parametrize("problem_type", UF_PROBLEMS)
def test_uf_can_expose_decision_bounds_as_constraints(problem_type: type[MOP]) -> None:
    problem = problem_type(n_var=8, boundary_constraints=True)
    assert isinstance(problem, CMOP)
    assert problem.n_ieq_constr == 2 * problem.n_var

    x = (problem.xl + problem.xu) / 2
    constraints = problem.ieq_constraint(x)
    assert constraints.shape == (2 * problem.n_var,)
    assert np.all(constraints <= 0)
    population = np.stack((x, problem.xl))
    np.testing.assert_allclose(
        problem.ieq_constraint_batch(population),
        np.stack([problem.ieq_constraint(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.ieq_jacobian_batch(population),
        np.stack([problem.ieq_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.ieq_hessian_batch(population),
        np.stack([problem.ieq_hessian(row) for row in population]),
    )


def test_uf_bounds_can_be_overridden_explicitly() -> None:
    xl = np.array([0.1, -0.5, -0.25])
    xu = np.array([0.9, 0.5, 0.25])
    problem = UF1(n_var=3, xl=xl, xu=xu, boundary_constraints=True)

    np.testing.assert_array_equal(problem.xl, xl)
    np.testing.assert_array_equal(problem.xu, xu)
    assert problem.ieq_constraint((xl + xu) / 2).shape == (2 * problem.n_var,)
