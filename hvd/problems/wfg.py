"""JAX-native WFG multi-objective benchmark problems.

WFG contains floor, absolute-value, and piecewise transformations, so its
derivatives are meaningful piecewise and remain non-smooth at their kinks.
"""

import warnings
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .base import CMOP

# Efficient means nondominated over all feasible decisions (Huband et al.,
# 2006, Sec. III-A, doi:10.1109/TEVC.2005.861417). Set samples below are derived
# inverse representatives, not a published complete parameterization. WFG3's
# full 3D front follows Ishibuchi et al. (2016), Eq. (25),
# doi:10.1109/TEVC.2015.2505784.
# Limitations: WFG1's 0.02 power amplifies float64 roundoff; failed set/front
# roundtrips raise unless allow_approximate=True (warns). Its full Jacobian and
# Hessian are undefined at efficient decisions. Full WFG3 fronts/sets support
# only 2 or 3 objectives. Sets select one inverse branch per target point;
# custom bounds may exclude these representatives.


def _wfg_front_integer_at_least(value: int, minimum: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _wfg_cover_candidates(candidates: np.ndarray, n_points: int, forced: int) -> np.ndarray:
    """Cover objective-space candidates in raw Euclidean distance."""
    selected = list(range(forced))
    minimum_squared = np.full(len(candidates), np.inf)
    for index in selected:
        minimum_squared = np.minimum(
            minimum_squared, np.sum((candidates - candidates[index]) ** 2, axis=1)
        )
    minimum_squared[selected] = -1.0
    while len(selected) < n_points:
        index = int(np.argmax(minimum_squared))
        selected.append(index)
        minimum_squared = np.minimum(
            minimum_squared, np.sum((candidates - candidates[index]) ** 2, axis=1)
        )
        minimum_squared[selected] = -1.0
    return candidates[selected]


def _wfg2_front_intervals() -> tuple[tuple[float, float], ...]:
    """Closures of the strict-record intervals of u*cos(5*pi*u)**2.

    The first left endpoint is included. Subsequent left endpoints tie an
    earlier record peak and are dominated, so the sampler excludes them.
    """
    from scipy.optimize import brentq

    def record(u: float) -> float:
        return u * np.cos(5 * np.pi * u) ** 2

    def derivative(u: float) -> float:
        return np.cos(5 * np.pi * u) ** 2 - 5 * np.pi * u * np.sin(10 * np.pi * u)

    intervals = []
    previous_record = 0.0
    for index in range(6):
        center = index / 5
        right = 1.0 if index == 5 else brentq(derivative, center, center + 0.05, xtol=1e-15)
        left = 0.0 if index == 0 else brentq(
            lambda u: record(u) - previous_record, center - 0.1, center, xtol=1e-15
        )
        intervals.append((left, right))
        previous_record = record(right)
    return tuple(intervals)


def _wfg12_shape(positions: np.ndarray, number: int) -> np.ndarray:
    """Zero-distance convex/mixed (WFG1) or convex/disconnected (WFG2) shapes."""
    # Half-angle identities avoid cancellation close to the shape boundaries.
    convex = 2 * np.sin(np.pi * positions / 4) ** 2
    complement = 2 * np.sin(np.pi * (1 - positions) / 4) ** 2
    convex[positions == 1] = 1.0
    complement[positions == 0] = 1.0
    n_obj = positions.shape[1] + 1
    columns = [np.prod(convex, axis=1)]
    for objective in range(2, n_obj):
        index = n_obj - objective
        columns.append(np.prod(convex[:, :index], axis=1) * complement[:, index])

    u = positions[:, 0]
    if number == 1:
        remainder = 1 - u
        angle = 10 * np.pi * remainder
        last = remainder - np.sin(angle) / (10 * np.pi)
        small = angle < 0.1
        squared = angle[small] ** 2
        last[small] = remainder[small] * squared * (
            1 / 6 - squared / 120 + squared**2 / 5040 - squared**3 / 362880
        )
    else:
        last = 1 - u * np.cos(5 * np.pi * u) ** 2
    last[u == 0] = 1.0
    last[u == 1] = 0.0
    return np.column_stack((*columns, last))


def _wfg12_front(number: int, n_obj: int, n_points: int) -> np.ndarray:
    """Cover the analytic WFG1/2 front without evaluating efficient decisions."""
    scales = 2.0 * np.arange(1, n_obj + 1)
    axes = np.eye(n_obj) * scales
    if n_points == n_obj:
        return axes

    from scipy.stats import qmc

    dimension = n_obj - 1
    intervals = ((0.0, 1.0),) if number == 1 else _wfg2_front_intervals()
    per_branch = max(256, (32 * n_points + len(intervals) - 1) // len(intervals))
    exponent = (per_branch - 1).bit_length()
    sample = qmc.Sobol(dimension, scramble=True, seed=711).random_base2(exponent)
    face_count = min(len(sample), max(32, 2 * n_points // len(intervals)))
    # Representatives of the remaining convex shape's objective-axis limits.
    angular_corners = np.array([
        [1.0] * (dimension - 1 - index) + [0.0] * index
        for index in range(dimension)
    ]).reshape(dimension, dimension - 1)
    positions = []
    peaks = []
    for branch, (left, right) in enumerate(intervals):
        block = sample.copy()
        block[:, 0] = left + (right - left) * block[:, 0]
        positions.append(block)
        for coordinate in range(1, dimension):
            for side in (0.0, 1.0):
                face = block[:face_count].copy()
                face[:, coordinate] = side
                positions.append(face)
        # Sample both u boundaries, moving inside later open branch starts.
        lower = left if branch == 0 else left + (right - left) / (2**exponent)
        for endpoint in (lower, right):
            face = sample[:face_count].copy()
            face[:, 0] = endpoint
            positions.append(face)
            positions.append(np.column_stack((np.full(dimension, endpoint), angular_corners)))
        if number == 2:
            peak = np.full(dimension, 0.5)
            peak[0] = right
            peaks.append(peak)

    candidates = _wfg12_shape(np.vstack(positions), number) * scales
    forced = axes
    if number == 2 and n_points >= n_obj + len(peaks):
        # Keep every disconnected component, including the small first one.
        forced = np.vstack((axes, _wfg12_shape(np.array(peaks), number) * scales))
        _, unique = np.unique(forced, axis=0, return_index=True)
        forced = forced[np.sort(unique)]
    return _wfg_cover_candidates(np.vstack((forced, candidates)), n_points, len(forced))


def _wfg3_front(n_obj: int, n_points: int) -> np.ndarray:
    """Cover WFG3's full front, including its three-objective sheet.

    Ishibuchi, Masuda and Nojima (2016), doi:10.1109/TEVC.2015.2505784,
    equations (17) and (25). The distance-zero line alone is incomplete for
    three or more objectives. The full higher-dimensional front is unsupported.
    """
    if n_obj > 3:
        raise NotImplementedError(
            "WFG3's full Pareto front is implemented for 2 and 3 objectives only; "
            "a distance-zero line omits its nondegenerate part for n_obj >= 3."
        )
    if n_obj == 2:
        t = np.linspace(0.0, 1.0, n_points)
        return np.column_stack((2 * t, 4 * (1 - t)))

    from scipy.stats import qmc

    exponent = (max(1024, 32 * n_points) - 1).bit_length()
    u = qmc.Sobol(2, scramble=True, seed=711).random_base2(exponent)
    edge_exponent = (max(32, 2 * n_points) - 1).bit_length()
    t = np.linspace(0.0, 1.0, 2**edge_exponent + 1)
    line = np.column_stack((t, 2 * t, 6 * (1 - t)))

    # The sheet has a in (2/3, 1] and d in (0, 1]. Neither its a=2/3
    # boundary nor the d=0 duplicate line belongs among sheet candidates.
    a = np.r_[2 / 3 + u[:, 0] / 3, np.ones(len(t) - 1), 2 / 3 + t[1:] / 3]
    d = np.r_[u[:, 1], t[1:], np.ones(len(t) - 1)]
    sheet = np.column_stack((d + a * (1 + d), d + 2 * a * (1 - d), d + 6 * (1 - a)))
    extremes = np.array([[0.0, 0.0, 6.0], [1.0, 2.0, 0.0], [3.0, 1.0, 1.0]])
    return _wfg_cover_candidates(np.vstack((extremes, line, sheet)), n_points, forced=3)


def _wfg_ellipsoid_cover(n_obj: int, n_points: int) -> np.ndarray:
    scales = 2.0 * np.arange(1, n_obj + 1)
    axes = np.eye(n_obj)
    if n_points == n_obj:
        return axes * scales

    from scipy.special import ndtri
    from scipy.stats import qmc

    # Sobol coordinates transformed to absolute normal directions sample the
    # normalized sphere. Only the final covering selection uses raw scales.
    exponent = (max(1024, 32 * n_points) - 1).bit_length()
    u = qmc.Sobol(n_obj, scramble=True, seed=711).random_base2(exponent)
    interior = np.abs(ndtri(np.clip(u, 1e-15, 1 - 1e-15)))
    interior /= np.linalg.norm(interior, axis=1, keepdims=True)

    edge_exponent = (max(32, 2 * n_points) - 1).bit_length()
    angles = np.linspace(0.0, np.pi / 2, 2**edge_exponent + 1)[1:-1]
    parts = [axes]
    for first in range(n_obj):
        for second in range(first + 1, n_obj):
            arc = np.zeros((len(angles), n_obj))
            arc[:, first] = np.cos(angles)
            arc[:, second] = np.sin(angles)
            parts.append(arc)
    parts.append(interior)
    candidates = np.vstack(parts) * scales

    # Force every objective-axis endpoint, then greedily reduce uncovered raw
    # Euclidean distances. Candidate order resolves ties deterministically.
    return _wfg_cover_candidates(candidates, n_points, forced=n_obj)


def _analytic_wfg_front(number: int, n_obj: int = 3, N: int = 500) -> np.ndarray:
    """Return exactly N analytic target points for WFG1–9.

    No decision-space variables, problem objective evaluation or global RNG
    state enter the construction. WFG4–9 return identical fronts at equal M,N.
    """
    number = _wfg_front_integer_at_least(number, 1, "number")
    if number > 9:
        raise ValueError("number must identify WFG1–9.")
    n_obj = _wfg_front_integer_at_least(n_obj, 2, "n_obj")
    N = _wfg_front_integer_at_least(N, n_obj, "N")
    if number in (1, 2):
        return _wfg12_front(number, n_obj, N)
    if number == 3:
        return _wfg3_front(n_obj, N)
    return _wfg_ellipsoid_cover(n_obj, N)


def _convex_front_positions(front: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Invert the convex shape's nested homogeneous gauge."""
    normalized = front[:, :-1] / scales[:-1]
    gauges = [normalized[:, 0].copy()]
    for coordinate in range(1, normalized.shape[1]):
        value = normalized[:, coordinate]
        gauges.append(gauges[-1] + value + np.sqrt(2 * gauges[-1] * value))
    columns = []
    for index in range(len(gauges) - 1, -1, -1):
        if index == len(gauges) - 1:
            cosine_complement = gauges[index]
        else:
            cosine_complement = np.divide(
                gauges[index], gauges[index + 1],
                out=np.zeros_like(gauges[index]), where=gauges[index + 1] > 0,
            )
        columns.append(4 * np.arcsin(np.sqrt(np.clip(cosine_complement, 0, 1) / 2)) / np.pi)
    return np.clip(np.column_stack(columns), 0.0, 1.0)


def _efficient_decisions_1_to_3(problem, front: np.ndarray) -> np.ndarray:
    """Invert the WFG1–3 transformations, including WFG3's efficient sheet."""
    gap = problem.k // (problem.n_obj - 1)
    values = np.full((len(front), problem.n_var), 0.35)
    if problem.problem_number in (1, 2):
        positions = _convex_front_positions(front, np.asarray(problem.S))
        if problem.problem_number == 1:
            positions = positions**50
        values[:, :problem.k] = np.repeat(positions, gap, axis=1)
    elif problem.n_obj == 2:
        values[:, :problem.k] = front[:, 0, None] / 2
    else:
        distance = np.clip((6 * front[:, 0] + 3 * front[:, 1] + 2 * front[:, 2] - 12) / 11, 0, 1)
        position = np.clip(1 + (distance - front[:, 2]) / 6, 0, 1)
        values[:, :gap] = position[:, None]
        values[:, gap:problem.k] = 1.0
        # The linear shift maps this suffix to pairs (d,0), whose
        # nonseparable reduction is (d+0+2*abs(d-0))/3=d.
        values[:, problem.k::2] += 0.65 * distance[:, None]
    return values * problem.xu


def _correct_to_01(x: jnp.ndarray, epsilon: float = 1.0e-10) -> jnp.ndarray:
    x = jnp.where((x < 0) & (x >= -epsilon), 0.0, x)
    return jnp.where((x > 1) & (x <= 1 + epsilon), 1.0, x)


def _shift_linear(value: jnp.ndarray, shift: float = 0.35) -> jnp.ndarray:
    result = jnp.abs(value - shift) / jnp.abs(jnp.floor(shift - value) + shift)
    return _correct_to_01(result)


def _shift_deceptive(
    y: jnp.ndarray,
    A: float = 0.35,
    B: float = 0.005,
    C: float = 0.05,
) -> jnp.ndarray:
    tmp1 = jnp.floor(y - A + B) * (1 - C + (A - B) / B) / (A - B)
    tmp2 = jnp.floor(A + B - y) * (1 - C + (1 - A - B) / B) / (1 - A - B)
    return _correct_to_01(1 + (jnp.abs(y - A) - B) * (tmp1 + tmp2 + 1 / B))


def _shift_multi_modal(y: jnp.ndarray, A: float, B: float, C: float) -> jnp.ndarray:
    tmp1 = jnp.abs(y - C) / (2 * (jnp.floor(C - y) + C))
    tmp2 = (4 * A + 2) * jnp.pi * (0.5 - tmp1)
    return _correct_to_01((1 + jnp.cos(tmp2) + 4 * B * tmp1**2) / (B + 2))


def _bias_flat(y: jnp.ndarray, a: float, b: float, c: float) -> jnp.ndarray:
    result = (
        a
        + jnp.minimum(0, jnp.floor(y - b)) * (a * (b - y) / b)
        - jnp.minimum(0, jnp.floor(c - y)) * ((1 - a) * (y - c) / (1 - c))
    )
    return _correct_to_01(result)


def _bias_poly(y: jnp.ndarray, alpha: float) -> jnp.ndarray:
    return _correct_to_01(y**alpha)


def _param_dependent(
    y: jnp.ndarray,
    y_degree: jnp.ndarray,
    A: float = 0.98 / 49.98,
    B: float = 0.02,
    C: float = 50.0,
) -> jnp.ndarray:
    auxiliary = A - (1 - 2 * y_degree) * jnp.abs(jnp.floor(0.5 - y_degree) + A)
    return _correct_to_01(y ** (B + (C - B) * auxiliary))


def _param_deceptive(
    y: jnp.ndarray,
    A: float = 0.35,
    B: float = 0.001,
    C: float = 0.05,
) -> jnp.ndarray:
    return _shift_deceptive(y, A=A, B=B, C=C)


def _weighted_sum(y: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return _correct_to_01(jnp.dot(y, weights) / weights.sum())


def _uniform_sum(y: jnp.ndarray) -> jnp.ndarray:
    return _correct_to_01(y.mean(axis=1))


def _non_separable(y: jnp.ndarray, A: int) -> jnp.ndarray:
    n_rows, n_columns = y.shape
    value = jnp.ceil(A / 2)
    numerator = jnp.zeros(n_rows, dtype=y.dtype)
    for column in range(n_columns):
        numerator = numerator + y[:, column]
        for offset in range(A - 1):
            numerator = numerator + jnp.abs(
                y[:, column] - y[:, (column + offset + 1) % n_columns]
            )
    denominator = n_columns * value * (1 + 2 * A - 2 * value) / A
    return _correct_to_01(numerator / denominator)


def _concave(x: jnp.ndarray, m: int) -> jnp.ndarray:
    n_columns = x.shape[1]
    if m == 1:
        result = jnp.prod(jnp.sin(0.5 * x * jnp.pi), axis=1)
    elif m <= n_columns:
        result = jnp.prod(jnp.sin(0.5 * x[:, : n_columns - m + 1] * jnp.pi), axis=1)
        result = result * jnp.cos(0.5 * x[:, n_columns - m + 1] * jnp.pi)
    else:
        result = jnp.cos(0.5 * x[:, 0] * jnp.pi)
    return _correct_to_01(result)


def _convex(x: jnp.ndarray, m: int) -> jnp.ndarray:
    n_columns = x.shape[1]
    if m == 1:
        result = jnp.prod(1 - jnp.cos(0.5 * x * jnp.pi), axis=1)
    elif m <= n_columns:
        result = jnp.prod(1 - jnp.cos(0.5 * x[:, : n_columns - m + 1] * jnp.pi), axis=1)
        result = result * (1 - jnp.sin(0.5 * x[:, n_columns - m + 1] * jnp.pi))
    else:
        result = 1 - jnp.sin(0.5 * x[:, 0] * jnp.pi)
    return _correct_to_01(result)


def _linear(x: jnp.ndarray, m: int) -> jnp.ndarray:
    n_columns = x.shape[1]
    if m == 1:
        result = jnp.prod(x, axis=1)
    elif m <= n_columns:
        result = jnp.prod(x[:, : n_columns - m + 1], axis=1)
        result = result * (1 - x[:, n_columns - m + 1])
    else:
        result = 1 - x[:, 0]
    return _correct_to_01(result)


def _mixed(x: jnp.ndarray, A: float = 5.0, alpha: float = 1.0) -> jnp.ndarray:
    auxiliary = 2 * A * jnp.pi
    return _correct_to_01((1 - x - jnp.cos(auxiliary * x + 0.5 * jnp.pi) / auxiliary) ** alpha)


def _disconnected(
    x: jnp.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    A: float = 5.0,
) -> jnp.ndarray:
    return _correct_to_01(1 - x**alpha * jnp.cos(A * jnp.pi * x**beta) ** 2)


def _post(t: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
    columns = [
        jnp.maximum(t[:, -1], A[index]) * (t[:, index] - 0.5) + 0.5
        for index in range(t.shape[1] - 1)
    ]
    return jnp.column_stack([*columns, t[:, -1]])


def _calculate(x: jnp.ndarray, scales: jnp.ndarray, shapes: list[jnp.ndarray]) -> jnp.ndarray:
    return x[:, -1, None] + scales * jnp.column_stack(shapes)


def _wfg1_t1(x: jnp.ndarray, n_var: int, k: int) -> jnp.ndarray:
    return x.at[:, k:n_var].set(_shift_linear(x[:, k:n_var]))


def _wfg1_t2(x: jnp.ndarray, n_var: int, k: int) -> jnp.ndarray:
    return x.at[:, k:n_var].set(_bias_flat(x[:, k:n_var], 0.8, 0.75, 0.85))


def _wfg1_t4(x: jnp.ndarray, n_obj: int, n_var: int, k: int) -> jnp.ndarray:
    weights = jnp.arange(2, 2 * n_var + 1, 2, dtype=x.dtype)
    gap = k // (n_obj - 1)
    columns = [
        _weighted_sum(x[:, (index - 1) * gap : index * gap], weights[(index - 1) * gap : index * gap])
        for index in range(1, n_obj)
    ]
    columns.append(_weighted_sum(x[:, k:n_var], weights[k:n_var]))
    return jnp.column_stack(columns)


def _wfg2_t2(x: jnp.ndarray, n_var: int, k: int) -> jnp.ndarray:
    columns = [x[:, index] for index in range(k)]
    for index in range(k, k + (n_var - k) // 2):
        head = k + 2 * (index - k)
        columns.append(_non_separable(x[:, head : head + 2], 2))
    return jnp.column_stack(columns)


def _wfg2_t3(x: jnp.ndarray, n_obj: int, n_var: int, k: int) -> jnp.ndarray:
    end = k + (n_var - k) // 2
    gap = k // (n_obj - 1)
    columns = [
        _uniform_sum(x[:, (index - 1) * gap : index * gap])
        for index in range(1, n_obj)
    ]
    columns.append(_uniform_sum(x[:, k:end]))
    return jnp.column_stack(columns)


def _wfg4_t2(x: jnp.ndarray, n_obj: int, k: int) -> jnp.ndarray:
    gap = k // (n_obj - 1)
    columns = [
        _uniform_sum(x[:, (index - 1) * gap : index * gap])
        for index in range(1, n_obj)
    ]
    columns.append(_uniform_sum(x[:, k:]))
    return jnp.column_stack(columns)


def _wfg6_t2(x: jnp.ndarray, n_obj: int, n_var: int, k: int) -> jnp.ndarray:
    gap = k // (n_obj - 1)
    columns = [
        _non_separable(x[:, (index - 1) * gap : index * gap], gap)
        for index in range(1, n_obj)
    ]
    columns.append(_non_separable(x[:, k:], n_var - k))
    return jnp.column_stack(columns)


def _evaluate_wfg(
    number: int,
    x: jnp.ndarray,
    n_var: int,
    n_obj: int,
    k: int,
    xu: jnp.ndarray,
    scales: jnp.ndarray,
    A: jnp.ndarray,
) -> jnp.ndarray:
    y = x / xu

    if number == 1:
        y = _wfg1_t1(y, n_var, k)
        y = _wfg1_t2(y, n_var, k)
        y = _bias_poly(y, 0.02)
        y = _wfg1_t4(y, n_obj, n_var, k)
        y = _post(y, A)
        shapes = [_convex(y[:, :-1], m) for m in range(1, n_obj)]
        shapes.append(_mixed(y[:, 0]))
    elif number in (2, 3):
        y = _wfg1_t1(y, n_var, k)
        y = _wfg2_t2(y, n_var, k)
        y = _wfg2_t3(y, n_obj, n_var, k)
        y = _post(y, A)
        shape = _convex if number == 2 else _linear
        shapes = [shape(y[:, :-1], m) for m in range(1, n_obj + 1)]
        if number == 2:
            shapes[-1] = _disconnected(y[:, 0])
    elif number in (4, 5):
        y = _shift_multi_modal(y, 30.0, 10.0, 0.35) if number == 4 else _param_deceptive(y)
        y = _wfg4_t2(y, n_obj, k)
        y = _post(y, A)
        shapes = [_concave(y[:, :-1], m) for m in range(1, n_obj + 1)]
    elif number == 6:
        y = _wfg1_t1(y, n_var, k)
        y = _wfg6_t2(y, n_obj, n_var, k)
        y = _post(y, A)
        shapes = [_concave(y[:, :-1], m) for m in range(1, n_obj + 1)]
    elif number == 7:
        for index in range(k):
            y = y.at[:, index].set(_param_dependent(y[:, index], _uniform_sum(y[:, index + 1 :])))
        y = _wfg1_t1(y, n_var, k)
        y = _wfg4_t2(y, n_obj, k)
        y = _post(y, A)
        shapes = [_concave(y[:, :-1], m) for m in range(1, n_obj + 1)]
    elif number == 8:
        transformed = [
            _param_dependent(
                y[:, index],
                _uniform_sum(y[:, :index]),
                A=0.98 / 49.98,
                B=0.02,
                C=50.0,
            )
            for index in range(k, n_var)
        ]
        y = y.at[:, k:n_var].set(jnp.column_stack(transformed))
        y = _wfg1_t1(y, n_var, k)
        y = _wfg4_t2(y, n_obj, k)
        y = _post(y, A)
        shapes = [_concave(y[:, :-1], m) for m in range(1, n_obj + 1)]
    elif number == 9:
        transformed = [
            _param_dependent(y[:, index], _uniform_sum(y[:, index + 1 :]))
            for index in range(n_var - 1)
        ]
        y = y.at[:, : n_var - 1].set(jnp.column_stack(transformed))
        y = jnp.column_stack(
            [
                *[_shift_deceptive(y[:, index], 0.35, 0.001, 0.05) for index in range(k)],
                *[_shift_multi_modal(y[:, index], 30.0, 95.0, 0.35) for index in range(k, n_var)],
            ]
        )
        y = _wfg6_t2(y, n_obj, n_var, k)
        shapes = [_concave(y[:, :-1], m) for m in range(1, n_obj + 1)]
    else:
        raise ValueError(f"Unsupported WFG problem number: {number}.")

    return _calculate(y, scales, shapes)


class WFG(CMOP):
    """Base class for JAX-native WFG problems."""

    problem_number: ClassVar[int]

    def __init__(
        self,
        n_var: int = 24,
        n_obj: int = 3,
        xl: ArrayLike = 0.0,
        xu: ArrayLike | None = None,
        boundary_constraints: bool = False,
        k: int | None = None,
        l: int | None = None,
    ) -> None:
        k = (4 if n_obj == 2 else 2 * (n_obj - 1)) if k is None else k
        l = n_var - k if l is None else l
        self._validate_parameters(n_var=n_var, n_obj=n_obj, k=k, l=l)

        self.k: int = k
        self.l: int = l
        self.S: jnp.ndarray = jnp.arange(2, 2 * n_obj + 1, 2, dtype=float)
        self.A: jnp.ndarray = (
            jnp.r_[1.0, jnp.zeros(n_obj - 2)]
            if self.problem_number == 3
            else jnp.ones(n_obj - 1)
        )
        if xu is None:
            xu = 2 * np.arange(1, n_var + 1, dtype=float)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            boundary_constraints=boundary_constraints,
        )

    def _validate_parameters(self, n_var: int, n_obj: int, k: int, l: int) -> None:
        if n_obj < 2:
            raise ValueError("WFG problems require n_obj >= 2.")
        if k < 4:
            raise ValueError("WFG position parameter k must be at least 4.")
        if k % (n_obj - 1) != 0:
            raise ValueError("WFG position parameter k must be divisible by n_obj - 1.")
        if k + l != n_var or l < 1 or k + l < n_obj:
            raise ValueError("WFG parameters must satisfy k + l = n_var and k + l >= n_obj.")
        if self.problem_number in (2, 3) and l % 2 != 0:
            raise ValueError("WFG2 and WFG3 require an even distance parameter l.")

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        values = _evaluate_wfg(
            self.problem_number,
            jnp.atleast_2d(x),
            self.n_var,
            self.n_obj,
            self.k,
            jnp.asarray(self.xu),
            self.S,
            self.A,
        )
        return values[0]

    def get_pareto_set(self, N: int = 500, *, allow_approximate: bool = False) -> np.ndarray:
        """Return N decision representatives of the analytic target front.

        One inverse preimage is selected for each row of get_pareto_front(N),
        rather than enumerating every inverse branch of the infinite efficient
        set. Bounds and objective roundtrips are checked before returning.

        WFG1 generally cannot reproduce its exact efficient set in float64.
        A failed WFG1 roundtrip raises FloatingPointError by default; explicitly
        allowing approximation returns nominal decisions with a RuntimeWarning.
        This does not alter objective values or derivative conventions.
        """
        if not isinstance(allow_approximate, (bool, np.bool_)):
            raise ValueError("allow_approximate must be a boolean.")
        front = self.get_pareto_front(N)
        if self.problem_number <= 3:
            decisions = _efficient_decisions_1_to_3(self, front)
        else:
            from ._wfg_pareto import efficient_decisions_4_to_9

            decisions = efficient_decisions_4_to_9(self, front)
        if (
            not np.isfinite(decisions).all()
            or np.any(decisions < self.xl)
            or np.any(decisions > self.xu)
        ):
            raise ValueError("The efficient-decision construction is incompatible with these WFG bounds.")
        actual = self.objective(decisions)
        if not np.all(np.isfinite(actual)) or not np.allclose(actual, front, rtol=1e-9, atol=1e-9):
            error = float(np.max(np.abs(actual - front)))
            if self.problem_number != 1:
                raise FloatingPointError(
                    f"WFG{self.problem_number} inverse objective roundtrip failed: max error {error:.6g}."
                )
            message = (
                "WFG1's nominal efficient decisions fail the float64 objective roundtrip "
                f"(max error {error:.6g}). Decimal optimal distance coordinates and the "
                "0.02 power are numerically ill-conditioned; the analytic target remains "
                "the reference. allow_approximate=True explicitly returns nominal decisions "
                "without certifying numerical efficiency."
            )
            if not allow_approximate:
                raise FloatingPointError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return decisions

    def get_pareto_front(self, N: int = 500) -> np.ndarray:
        return _analytic_wfg_front(self.problem_number, self.n_obj, N)


class WFG1(WFG):
    problem_number = 1


class WFG2(WFG):
    problem_number = 2


class WFG3(WFG):
    problem_number = 3


class WFG4(WFG):
    problem_number = 4


class WFG5(WFG):
    problem_number = 5


class WFG6(WFG):
    problem_number = 6


class WFG7(WFG):
    problem_number = 7


class WFG8(WFG):
    problem_number = 8


class WFG9(WFG):
    problem_number = 9
