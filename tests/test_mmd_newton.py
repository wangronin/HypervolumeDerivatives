import numpy as np
import pytest

from hvd.mmd_newton import MMDNewton
from hvd.mmd_vectorized import laplace
from hvd.problems import IDTLZ1
from hvd.reference_set import ReferenceSet


@pytest.mark.parametrize("boundary_constraints", [False, True])
def test_idtlz1_runs_with_optional_boundary_constraints(boundary_constraints: bool) -> None:
    problem = IDTLZ1(boundary_constraints=boundary_constraints)
    x0 = np.vstack([np.zeros(problem.n_var), np.full(problem.n_var, 0.5)])

    optimizer = MMDNewton(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        ref=ReferenceSet(problem.get_pareto_front()),
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
        N=len(x0),
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=1,
        verbose=False,
        matching=False,
        regularization=True,
        theta=500.0,
        kernel=laplace,
    )

    X, Y, _ = optimizer.run()

    expected_constraint_count = 2 * problem.n_var if boundary_constraints else 0
    assert optimizer.n_ieq == expected_constraint_count
    assert np.all(optimizer.step_size > 0)
    assert X.shape == x0.shape
    assert Y.shape == (len(x0), problem.n_obj)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(Y))
