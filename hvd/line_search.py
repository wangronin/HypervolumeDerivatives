from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np

Payload = TypeVar("Payload")


@dataclass(frozen=True)
class LineSearchResult(Generic[Payload]):
    """Result of a scalar line search along a prescribed direction."""

    step_size: float
    residual: np.ndarray
    merit: float
    evaluations: int
    converged: bool
    payload: Payload | None = None


def residual_armijo_line_search(
    residual: np.ndarray,
    evaluate: Callable[[float], np.ndarray | tuple[np.ndarray, Payload]],
    *,
    initial_step: float = 1.0,
    max_step: float = 1.0,
    c1: float = 1e-4,
    contraction: float = 0.5,
    max_evaluations: int = 10,
    minimum_step: float = 0.0,
    directional_derivative: float | None = None,
) -> LineSearchResult[Payload]:
    """Armijo search for the nonlinear-system merit ``0.5 * ||R||**2``.

    By default the direction is assumed to be a Newton direction, giving
    ``phi'(0) = -||R||**2``. ``evaluate(alpha)`` may additionally return an
    arbitrary payload, such as an already evaluated trial state.
    """
    if not 0 < c1 < 1:
        raise ValueError("`c1` must lie strictly between zero and one.")
    if not 0 < contraction < 1:
        raise ValueError("`contraction` must lie strictly between zero and one.")
    if max_evaluations < 1:
        raise ValueError("`max_evaluations` must be positive.")
    if initial_step < 0 or max_step < 0 or minimum_step < 0:
        raise ValueError("Step sizes must be non-negative.")

    residual = np.asarray(residual)
    residual_norm_squared = float(np.vdot(residual.ravel(), residual.ravel()).real)
    phi0 = 0.5 * residual_norm_squared
    derivative0 = -residual_norm_squared if directional_derivative is None else float(directional_derivative)
    if derivative0 >= 0 and phi0 > 0:
        raise ValueError("The search direction must be a descent direction for the residual merit.")

    if phi0 <= np.finfo(float).eps or initial_step == 0 or max_step == 0:
        return LineSearchResult(0.0, residual.copy(), phi0, 0, True)

    alpha = min(float(initial_step), float(max_step))
    best = LineSearchResult(0.0, residual.copy(), phi0, 0, False)
    for evaluation in range(1, max_evaluations + 1):
        evaluated = evaluate(alpha)
        trial_residual, payload = evaluated if isinstance(evaluated, tuple) else (evaluated, None)
        trial_residual = np.asarray(trial_residual)
        merit = 0.5 * float(np.vdot(trial_residual.ravel(), trial_residual.ravel()).real)
        candidate = LineSearchResult(alpha, trial_residual, merit, evaluation, False, payload)
        if np.isfinite(merit) and merit < best.merit:
            best = candidate
        if np.isfinite(merit) and merit <= phi0 + c1 * alpha * derivative0:
            return LineSearchResult(alpha, trial_residual, merit, evaluation, True, payload)
        next_alpha = alpha * contraction
        if next_alpha < minimum_step or next_alpha == alpha:
            break
        alpha = next_alpha

    return LineSearchResult(
        best.step_size,
        best.residual,
        best.merit,
        evaluation,
        False,
        best.payload,
    )
