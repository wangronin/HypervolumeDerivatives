import numpy as np
import pytest

from hvd.problems import CF1, Eq1DTLZ1, ZDT1


@pytest.mark.parametrize("problem_type", [ZDT1, CF1, Eq1DTLZ1])
@pytest.mark.parametrize("population_size", [0, 1])
def test_unified_evaluations_preserve_empty_and_singleton_axes(problem_type, population_size) -> None:
    problem = problem_type(n_var=5)
    point = np.full(problem.n_var, 0.3)
    population = np.tile(point, (population_size, 1))
    families = [("objective", problem.n_obj), ("eq", problem.n_eq_constr), ("ieq", problem.n_ieq_constr)]

    for family, count in families:
        for order, suffix in enumerate(("", "_jacobian", "_hessian")):
            name = f"{family}{suffix}" if family == "objective" or suffix else f"{family}_constraint"
            evaluate = getattr(problem, name)
            result = evaluate(population)
            if count == 0:
                assert result is None
                continue
            shape = (count,) + (problem.n_var,) * order
            assert result.shape == (population_size, *shape)
            assert evaluate(point).shape == shape
            if population_size:
                np.testing.assert_allclose(result[0], evaluate(point))


def test_batched_objective_derivatives_match_single_evaluations() -> None:
    problem = ZDT1(n_var=5)
    x = np.linspace(0.1, 0.9, problem.n_var)
    population = np.stack((x, 0.9 * x))

    np.testing.assert_allclose(
        problem.objective_jacobian(population),
        np.stack([problem.objective_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.objective_hessian(population),
        np.stack([problem.objective_hessian(row) for row in population]),
    )


def test_batched_inequality_derivatives_match_single_evaluations() -> None:
    problem = CF1(n_var=5)
    population = np.stack((np.full(5, 0.3), np.full(5, 0.6)))

    np.testing.assert_allclose(
        problem.ieq_jacobian(population),
        np.stack([problem.ieq_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.ieq_hessian(population),
        np.stack([problem.ieq_hessian(row) for row in population]),
    )


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
