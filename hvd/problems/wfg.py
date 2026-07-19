"""JAX-native WFG multi-objective benchmark problems.

WFG contains floor, absolute-value, and piecewise transformations, so its
derivatives are meaningful piecewise and remain non-smooth at their kinks.
"""

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .base import CMOP


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

    def _optimal_positions(self, positions: np.ndarray) -> np.ndarray:
        suffix = np.full((len(positions), self.l), 0.35)
        return np.column_stack((positions, suffix)) * self.xu

    def get_pareto_set(self, N: int = 500) -> np.ndarray:
        extremes = np.array(
            [
                [(row >> column) & 1 for column in range(self.k)]
                for row in range(2**self.k)
            ],
            dtype=float,
        )
        n_interior = max(0, N - len(extremes))
        interior = np.random.random((n_interior, self.k))
        if self.problem_number == 1:
            interior = interior**50
        positions = np.row_stack((extremes, interior)) if n_interior else extremes
        return self._optimal_positions(positions)

    def get_pareto_front(self, N: int = 500) -> np.ndarray:
        return self.objective_batch(self.get_pareto_set(N))


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

    def _optimal_positions(self, positions: np.ndarray) -> np.ndarray:
        values = positions
        for _ in range(self.l):
            degree = values.mean(axis=1)
            tmp1 = np.abs(np.floor(0.5 - degree) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1 - 2 * degree) * tmp1)
            suffix = 0.35 ** (tmp2**-1)
            values = np.column_stack((values, suffix))
        return values * self.xu


class WFG9(WFG):
    problem_number = 9

    def _optimal_positions(self, positions: np.ndarray) -> np.ndarray:
        values = np.column_stack((positions, np.zeros((len(positions), self.l))))
        values[:, -1] = 0.35
        for index in range(self.n_var - 2, self.k - 1, -1):
            degree = values[:, index + 1 :].mean(axis=1)
            values[:, index] = 0.35 ** ((0.02 + 1.96 * degree) ** -1)
        return values * self.xu
