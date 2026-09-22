import logging
from types import SimpleNamespace

import numpy as np
import pytest

from hvd.mmd_newton import MMDNewton
from hvd.mmd_vectorized import rational_quadratic
from hvd.problems import IDTLZ1
from hvd.reference_set import ReferenceSet
from hvd.utils import project_box_step


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
        matching=True,
        beta=0.37,
        regularization=True,
        project_box_constraints=True,
        theta=1.0,
        kernel=rational_quadratic,
    )

    X, Y, _ = optimizer.run()

    expected_constraint_count = 2 * problem.n_var if boundary_constraints else 0
    assert optimizer.n_ieq == expected_constraint_count
    assert optimizer.indicator.beta == pytest.approx(0.37)
    assert np.all(optimizer.step_size >= 0)
    assert optimizer.step_size.shape == (len(x0),)
    assert X.shape == x0.shape
    assert Y.shape == (len(x0), problem.n_obj)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(Y))
    assert np.all(X >= problem.xl)
    assert np.all(X <= problem.xu)


def test_fraction_to_boundary_is_computed_per_point() -> None:
    problem = IDTLZ1(boundary_constraints=False)
    x0 = np.vstack([np.zeros(problem.n_var), np.full(problem.n_var, 0.5)])
    step = np.zeros_like(x0)
    step[0, 0] = -2.0
    step[0, 1] = 2.0
    step[1, 0] = 2.0

    feasible_step, maximum = project_box_step(step, x0, problem.xl, problem.xu)

    assert feasible_step[0, 0] == 0.0
    assert maximum[0] == pytest.approx(0.4975)
    assert maximum[1] == pytest.approx(0.24875)
    candidate = x0 + maximum[:, None] * feasible_step[:, : problem.n_var]
    assert np.all(candidate >= problem.xl)
    assert np.all(candidate <= problem.xu)


def test_individual_line_search_does_not_take_an_untested_step() -> None:
    class TrialState:
        def __init__(self):
            self.X = np.zeros((2, 1))
            self.n_jac_evals = 0

        def update_one(self, x, i):
            self.X[i] = x

    optimizer = MMDNewton.__new__(MMDNewton)
    optimizer.N = 2
    optimizer.state = TrialState()
    optimizer.indicator = SimpleNamespace(re_match=True)
    optimizer.logger = logging.getLogger(__name__)
    tested = []

    def residual(state):
        tested.append(state.X.copy())
        first = 2.0 if state.X[0, 0] > 0.5 else 0.1
        return (np.array([[first], [2.0]]),)

    optimizer._compute_R = residual
    step_size = optimizer._backtracking_line_search_individual(
        np.ones((2, 1)), np.ones((2, 1))
    )

    assert step_size == pytest.approx([0.5, 0.0])
    assert len(tested) == 8  # 2 trials for point 0, 6 for point 1
    assert optimizer.indicator.re_match
