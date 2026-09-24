import logging
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hvd.mmd import MMD, MMDMatching
from hvd.mmd.kernels import RationalQuadratic
from hvd.mmd_newton import MMDN
from hvd.problems import IDTLZ1
from hvd.reference_set import ReferenceSet
from hvd.utils import project_box_step


@pytest.fixture
def newton_args() -> dict:
    return dict(
        n_var=2,
        n_obj=2,
        func=lambda x: x,
        jac=lambda x: np.broadcast_to(np.eye(2), (*x.shape[:-1], 2, 2)),
        xl=np.zeros(2),
        xu=np.ones(2),
        X0=np.array([[0.2, 0.8], [0.8, 0.2]]),
        max_iters=1,
        regularization=True,
        verbose=False,
    )


def test_reference_is_owned_by_injected_indicator(newton_args: dict) -> None:
    reference = ReferenceSet(np.array([[0.0, 1.0], [1.0, 0.0]]))
    indicator = MMDMatching(2, 2, reference, beta=0.37)
    callbacks = (indicator.func, indicator.jac, indicator.hessian)
    optimizer = MMDN(**newton_args, indicator=indicator)

    assert optimizer.indicator is indicator
    assert (indicator.func, indicator.jac, indicator.hessian) == callbacks
    assert not hasattr(optimizer, "ref")
    assert not hasattr(optimizer, "history_medoids")
    replacement = ReferenceSet(np.array([[0.1, 0.9], [0.9, 0.1]]))
    expected = replacement.reference_set.copy() + 0.04 * replacement.eta[0]
    indicator.ref = replacement
    optimizer.run()
    assert indicator.ref is replacement
    np.testing.assert_allclose(replacement.reference_set, expected)
    np.testing.assert_array_equal(reference.reference_set, [[0.0, 1.0], [1.0, 0.0]])


def test_requires_an_explicit_indicator(newton_args: dict) -> None:
    with pytest.raises(TypeError, match="indicator"):
        MMDN(**newton_args)


def test_does_not_accept_a_reference_set(newton_args: dict) -> None:
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    indicator = MMD(2, 2, reference)
    with pytest.raises(TypeError, match="unexpected keyword argument 'ref'"):
        MMDN(**newton_args, indicator=indicator, ref=ReferenceSet(reference.copy()))


def test_newton_evaluates_then_requests_shifts_without_owning_a_reference_set(newton_args: dict) -> None:
    class QuadraticIndicator:
        n_var = n_obj = 2

        def __init__(self):
            self.events = []
            self.shift_requests = []
            self.ref = SimpleNamespace(reference_set=np.array([[2.0, 2.0]]))

        def compute(self, Y):
            self.events.append("compute")
            return float(np.sum(Y**2))

        def shift_reference_set(self, indices):
            self.events.append("shift")
            self.shift_requests.append(indices.copy())

        def compute_derivatives(self, X, Y=None, compute_hessian=True, jacobian=None):
            self.events.append("derivatives")
            return (2 * X, 2 * np.eye(X.size)) if compute_hessian else 2 * X

        def trial_evaluation(self):
            return nullcontext()

    indicator = QuadraticIndicator()
    optimizer = MMDN(**(newton_args | {"max_iters": 2}), indicator=indicator)
    X, _, _ = optimizer.run()

    np.testing.assert_allclose(X, 0, atol=1e-12)
    assert not hasattr(optimizer, "ref")
    assert indicator.events[:3] == ["compute", "shift", "derivatives"]
    assert len(indicator.shift_requests) == 1
    np.testing.assert_array_equal(indicator.shift_requests[0], np.arange(optimizer.N))


def test_bootstrap_replaces_the_indicators_reference(newton_args: dict, monkeypatch) -> None:
    import matplotlib as mpl

    # The bootstrap module configures plotting on import; keep that local to this test.
    with mpl.rc_context():
        from hvd import bootstrap

        coordinate = np.linspace(0.2, 0.8, 4)
        population = np.column_stack([coordinate, 1 - coordinate])
        reference = ReferenceSet(population.copy())
        indicator = MMDMatching(2, 2, reference)
        optimizer = MMDN(**(newton_args | {"X0": population, "max_iters": 2}), indicator=indicator)
        problem = SimpleNamespace(get_pareto_front=lambda: population)
        monkeypatch.setattr(bootstrap, "filter_outliers", lambda data: data)

        X, Y, replacement, _ = bootstrap.bootstrap_reference_set(
            optimizer, problem, interval=1, with_rsg=False, plot=False
        )

    assert indicator.ref is replacement
    assert indicator.ref is not reference
    assert not hasattr(optimizer, "ref")
    assert X.shape == Y.shape == population.shape
    assert np.all(np.isfinite(Y))


@pytest.mark.parametrize("indicator_type", [MMD, MMDMatching])
def test_newton_only_requests_shifts_initially_or_for_reached_stationary_points(
    newton_args: dict, indicator_type, monkeypatch
) -> None:
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    indicator = indicator_type(2, 2, ReferenceSet(reference.copy(), eta={0: -np.ones(2)}))
    optimizer = MMDN(**(newton_args | {"X0": reference}), indicator=indicator)
    requests = []
    shift_reference_set = indicator.shift_reference_set

    def record_shift(indices):
        requests.append(indices.copy())
        shift_reference_set(indices=indices)

    monkeypatch.setattr(indicator, "shift_reference_set", record_shift)
    indicator.compute(Y=optimizer.state.Y)
    optimizer._shift_reference_set()
    np.testing.assert_array_equal(requests[0], [0, 1, 2])

    population = indicator.ref.reference_set.copy()
    population[2] += 0.1  # This point is stationary but has not reached a target.
    optimizer.state.update(population)
    optimizer.iter_count = 1
    optimizer.step = np.zeros_like(population)
    optimizer.step[1] = 1.0  # This point has reached a target but is still moving.
    indicator.compute(Y=optimizer.state.Y)
    optimizer._shift_reference_set()
    np.testing.assert_array_equal(requests[1], [0])

    # Shift selection is Newton's responsibility; ReferenceSet controls the movement.
    shifted_reference = indicator.ref.reference_set.copy()

    optimizer.step[:] = 1.0
    indicator.compute(Y=optimizer.state.Y)
    optimizer._shift_reference_set()
    assert len(requests) == 2
    np.testing.assert_allclose(indicator.ref.reference_set, shifted_reference)


@pytest.mark.parametrize("boundary_constraints", [False, True])
@pytest.mark.parametrize("matching", [False, True])
def test_idtlz1_runs_with_optional_boundary_constraints(boundary_constraints: bool, matching: bool) -> None:
    problem = IDTLZ1(boundary_constraints=boundary_constraints)
    x0 = np.vstack([np.zeros(problem.n_var), np.full(problem.n_var, 0.5)])
    kernel = RationalQuadratic(theta=0.7, alpha=1.3)
    indicator_type = MMDMatching if matching else MMD
    indicator = indicator_type(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        ref=ReferenceSet(problem.get_pareto_front()),
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        kernel=kernel,
        **({"beta": 0.37} if matching else {}),
    )
    optimizer = MMDN(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        indicator=indicator,
        func=problem.objective,
        jac=problem.objective_jacobian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
        N=len(x0),
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=1,
        verbose=False,
        regularization=True,
        project_box_constraints=True,
    )

    X, Y, _ = optimizer.run()

    expected_constraint_count = 2 * problem.n_var if boundary_constraints else 0
    assert optimizer.n_ieq == expected_constraint_count
    assert optimizer.indicator is indicator
    assert not hasattr(optimizer, "ref")
    assert optimizer.indicator.kernel is kernel
    assert not hasattr(optimizer.indicator, "theta")
    if matching:
        assert optimizer.indicator.beta == pytest.approx(0.37)
        assert optimizer.indicator.re_match
    else:
        assert not hasattr(optimizer.indicator, "re_match")
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

    optimizer = MMDN.__new__(MMDN)
    optimizer.N = 2
    optimizer.state = TrialState()
    optimizer.indicator = MMDMatching(2, 2, np.array([[0.0, 1.0], [1.0, 0.0]]))
    optimizer.logger = logging.getLogger(__name__)
    tested = []

    def residual(state):
        tested.append(state.X.copy())
        first = 2.0 if state.X[0, 0] > 0.5 else 0.1
        return (np.array([[first], [2.0]]),)

    optimizer._compute_R = residual
    step_size = optimizer._backtracking_line_search_individual(np.ones((2, 1)), np.ones((2, 1)))

    assert step_size == pytest.approx([0.5, 0.0])
    assert len(tested) == 8  # 2 trials for point 0, 6 for point 1
    assert optimizer.indicator.re_match


@pytest.mark.parametrize("method", ["global", "individual"])
@pytest.mark.parametrize("re_match", [None, False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_line_search_freezes_matching_indicators_and_reenables_matching_on_exit(
    method: str, re_match: bool | None, fail: bool
) -> None:
    matching = re_match is not None
    indicator_type = MMDMatching if matching else MMD
    indicator = indicator_type(2, 2, np.array([[0.0, 1.0], [1.0, 0.0]]))
    optimizer = MMDN(
        n_var=2,
        n_obj=2,
        func=lambda x: x,
        jac=lambda x: np.broadcast_to(np.eye(2), (*x.shape[:-1], 2, 2)),
        indicator=indicator,
        xl=np.zeros(2),
        xu=np.ones(2),
        X0=np.full((2, 2), 0.5),
        verbose=False,
    )
    if matching:
        optimizer.indicator.re_match = re_match
    reference_switch = indicator.ref.re_match
    evaluated = []

    def residual(state):
        evaluated.append(state.X.copy())
        if matching:
            assert not optimizer.indicator.re_match
        else:
            assert not hasattr(optimizer.indicator, "re_match")
        if fail:
            raise ValueError("trial evaluation failed")
        return (np.zeros_like(state.X),)

    optimizer._compute_R = residual
    line_search = getattr(optimizer, f"_backtracking_line_search_{method}")
    if fail:
        with pytest.raises(ValueError, match="trial evaluation failed"):
            line_search(np.full((2, 2), 0.1), np.ones((2, 2)))
    else:
        np.testing.assert_allclose(line_search(np.full((2, 2), 0.1), np.ones((2, 2))), 1.0)

    assert evaluated
    if matching:
        assert optimizer.indicator.re_match
    else:
        assert indicator.ref.re_match is reference_switch
        assert not hasattr(optimizer.indicator, "re_match")
