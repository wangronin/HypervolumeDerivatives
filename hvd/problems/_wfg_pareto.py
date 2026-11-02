"""Inverse WFG transformations used to construct efficient decision samples.

These inverses choose one feasible decision preimage for each supplied analytic
front point. They do not modify objective values or derivative conventions.
"""

import numpy as np


def _concave_positions(front: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Invert the nested positive-sphere shape using stable angular ratios."""
    normalized = np.asarray(front, dtype=float) / scales
    n_obj = normalized.shape[1]
    columns = []
    for coordinate in range(n_obj - 1):
        last = n_obj - coordinate - 1
        columns.append(
            2 * np.arctan2(np.linalg.norm(normalized[:, :last], axis=1), normalized[:, last]) / np.pi
        )
    return np.column_stack(columns)


def _parameter_exponent(degree: np.ndarray) -> np.ndarray:
    """Exponent in WFG's original b_param transformation."""
    a = 0.98 / 49.98
    auxiliary = a - (1 - 2 * degree) * np.abs(np.floor(0.5 - degree) + a)
    return 0.02 + 49.98 * auxiliary


def _inverse_multimodal(values: np.ndarray) -> np.ndarray:
    """Choose a preimage of s_multi(y,30,10,.35) on [.35,1].

    The bracket is continuous and spans values zero through one. It need not
    be monotone: a bracketed scalar solve may choose any of its valid roots.
    """
    from scipy.optimize import brentq

    unique, inverse = np.unique(values, return_inverse=True)
    roots = np.empty(len(unique))

    def shift(y: float) -> float:
        offset = abs(y - 0.35) / (2 * (np.floor(0.35 - y) + 0.35))
        angle = 122 * np.pi * (0.5 - offset)
        return (1 + np.cos(angle) + 40 * offset**2) / 12

    for index, value in enumerate(unique):
        if value <= 0:
            roots[index] = 0.35
        elif value >= 1:
            roots[index] = 1.0
        else:
            roots[index] = brentq(
                lambda y: shift(y) - value,
                0.35,
                1.0,
                xtol=np.nextafter(0.0, 1.0),
                rtol=4 * np.finfo(float).eps,
                maxiter=200,
            )
    return roots[inverse].reshape(values.shape)


def _nonseparable_groups(positions: np.ndarray, group_size: int) -> np.ndarray:
    """Use homogeneity and a balanced binary vector with r_nonsep=1."""
    pattern = np.r_[np.ones((group_size + 1) // 2), np.zeros(group_size // 2)]
    return np.concatenate([positions[:, index, None] * pattern for index in range(positions.shape[1])], axis=1)


def efficient_decisions_4_to_9(problem, front: np.ndarray) -> np.ndarray:
    """Return one efficient decision preimage per WFG4–9 front point.

    Inverse operations are performed in normalized decision coordinates; the
    problem's upper bounds provide the final raw-coordinate scaling. Standard
    WFG bounds are assumed. Custom lower bounds can exclude this construction
    and are rejected rather than silently returning infeasible decisions.
    """
    number = problem.problem_number
    if number not in range(4, 10):
        raise ValueError("This inverse supports WFG4–9 only.")
    positions = _concave_positions(front, np.asarray(problem.S))
    positions = np.clip(positions, 0.0, 1.0)
    gap = problem.k // (problem.n_obj - 1)
    values = np.full((len(front), problem.n_var), 0.35)

    if number == 4:
        values[:, :problem.k] = np.repeat(_inverse_multimodal(positions), gap, axis=1)
    elif number == 5:
        # The narrow interval [.35,.351] maps linearly from zero to one.
        values[:, :problem.k] = np.repeat(0.35 + 0.001 * positions, gap, axis=1)
    elif number == 6:
        values[:, :problem.k] = _nonseparable_groups(positions, gap)
    elif number == 7:
        desired = np.repeat(positions, gap, axis=1)
        for index in range(problem.k - 1, -1, -1):
            degree = values[:, index + 1:].mean(axis=1)
            values[:, index] = desired[:, index] ** (1 / _parameter_exponent(degree))
    elif number == 8:
        values[:, :problem.k] = np.repeat(positions, gap, axis=1)
        for index in range(problem.k, problem.n_var):
            degree = values[:, :index].mean(axis=1)
            values[:, index] = 0.35 ** (1 / _parameter_exponent(degree))
    else:
        desired = np.full_like(values, 0.35)
        desired[:, :problem.k] = 0.35 + 0.001 * _nonseparable_groups(positions, gap)
        for index in range(problem.n_var - 2, -1, -1):
            degree = values[:, index + 1:].mean(axis=1)
            values[:, index] = desired[:, index] ** (1 / _parameter_exponent(degree))

    raw = values * problem.xu
    if not np.isfinite(raw).all() or np.any(raw < problem.xl) or np.any(raw > problem.xu):
        raise ValueError("The efficient-decision construction is incompatible with these WFG bounds.")
    return raw
