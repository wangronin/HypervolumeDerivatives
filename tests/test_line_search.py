import numpy as np
import pytest

from hvd.line_search import residual_armijo_line_search


def test_residual_armijo_accepts_full_newton_step():
    residual = np.array([2.0, -1.0])
    evaluated = []

    def evaluate(alpha):
        evaluated.append(alpha)
        return (1 - alpha) * residual, {"alpha": alpha}

    result = residual_armijo_line_search(residual, evaluate)

    assert result.converged
    assert result.step_size == 1.0
    assert result.evaluations == 1
    assert np.allclose(result.residual, 0)
    assert result.payload == {"alpha": 1.0}
    assert evaluated == [1.0]


def test_residual_armijo_backtracks_to_an_evaluated_step():
    residual = np.array([1.0])
    evaluated = []

    def evaluate(alpha):
        evaluated.append(alpha)
        return np.array([3 * alpha - 1])

    result = residual_armijo_line_search(residual, evaluate)

    assert result.converged
    assert result.step_size == 0.5
    assert result.step_size in evaluated
    assert evaluated == [1.0, 0.5]


def test_failed_search_returns_best_evaluated_improvement():
    residual = np.array([1.0])
    evaluated = []

    def evaluate(alpha):
        evaluated.append(alpha)
        return np.array([1 - 1e-6 * alpha])

    result = residual_armijo_line_search(
        residual, evaluate, c1=1e-2, max_evaluations=3
    )

    assert not result.converged
    assert result.step_size in evaluated
    assert result.step_size == max(evaluated)
    assert result.merit < 0.5


def test_failed_search_rejects_direction_without_improvement():
    residual = np.array([1.0])
    result = residual_armijo_line_search(
        residual, lambda alpha: np.array([1 + alpha]), max_evaluations=3
    )

    assert not result.converged
    assert result.step_size == 0
    assert result.evaluations == 3
    assert np.array_equal(result.residual, residual)


def test_residual_armijo_respects_maximum_step():
    residual = np.array([1.0])
    evaluated = []
    result = residual_armijo_line_search(
        residual,
        lambda alpha: evaluated.append(alpha) or np.array([1 - alpha]),
        max_step=0.25,
    )

    assert result.converged
    assert result.step_size == 0.25
    assert evaluated == [0.25]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"c1": 0},
        {"contraction": 1},
        {"max_evaluations": 0},
        {"initial_step": -1},
    ],
)
def test_residual_armijo_validates_configuration(kwargs):
    with pytest.raises(ValueError):
        residual_armijo_line_search(np.ones(1), lambda alpha: np.zeros(1), **kwargs)
