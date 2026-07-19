import numpy as np

from hvd.problems import CF1, Eq1DTLZ1, ZDT1


def test_batched_objective_derivatives_match_single_evaluations() -> None:
    problem = ZDT1(n_var=5)
    x = np.linspace(0.1, 0.9, problem.n_var)
    population = np.stack((x, 0.9 * x))

    np.testing.assert_allclose(
        problem.objective_jacobian_batch(population),
        np.stack([problem.objective_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.objective_hessian_batch(population),
        np.stack([problem.objective_hessian(row) for row in population]),
    )


def test_batched_inequality_derivatives_match_single_evaluations() -> None:
    problem = CF1(n_var=5)
    population = np.stack((np.full(5, 0.3), np.full(5, 0.6)))

    np.testing.assert_allclose(
        problem.ieq_jacobian_batch(population),
        np.stack([problem.ieq_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.ieq_hessian_batch(population),
        np.stack([problem.ieq_hessian(row) for row in population]),
    )


def test_batched_equality_derivatives_match_single_evaluations() -> None:
    problem = Eq1DTLZ1(n_var=5)
    population = np.stack((np.full(5, 0.3), np.full(5, 0.6)))

    np.testing.assert_allclose(
        problem.eq_jacobian_batch(population),
        np.stack([problem.eq_jacobian(row) for row in population]),
    )
    np.testing.assert_allclose(
        problem.eq_hessian_batch(population),
        np.stack([problem.eq_hessian(row) for row in population]),
    )
