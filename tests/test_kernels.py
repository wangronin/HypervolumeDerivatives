import pickle
from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

from hvd.mmd import MMD
from hvd.mmd.kernels import RBF, CallableKernel, Laplace, RationalQuadratic


@pytest.mark.parametrize("kernel", [RBF(0.7), RationalQuadratic(0.7, 1.3), Laplace(0.7)])
def test_kernel_parameters_survive_serialization_and_cannot_change_after_jit(kernel) -> None:
    x, y = jnp.array([0.3, -0.5]), jnp.array([0.1, 0.4])
    compiled = jit(kernel)
    expected = compiled(x, y)

    derivatives = {
        name: getattr(kernel, name)(x, y)
        for name in ("gradient", "hessian", "mixed_hessian")
    }
    diagonal_derivatives = {
        name: getattr(kernel, name)(x)
        for name in ("diagonal", "diagonal_gradient", "diagonal_hessian")
    }
    restored = pickle.loads(pickle.dumps(kernel))
    np.testing.assert_allclose(jit(restored)(x, y), expected)
    for name, value in derivatives.items():
        np.testing.assert_allclose(getattr(restored, name)(x, y), value)
    for name, value in diagonal_derivatives.items():
        np.testing.assert_allclose(getattr(restored, name)(x), value)
    with pytest.raises(FrozenInstanceError):
        kernel.theta = 2.0
    np.testing.assert_allclose(compiled(x, y), expected)


def test_kernel_derivatives_are_cached_and_diagonal_derivatives_include_both_arguments() -> None:
    # A nonstationary kernel makes k(x, x) vary, exposing missing self-pair derivatives.
    kernel = CallableKernel(lambda x, y: (1 + jnp.dot(x, y)) ** 2)
    x, y = jnp.array([0.3, -0.5]), jnp.array([0.1, 0.4])

    gradient = kernel.gradient
    hessian = kernel.hessian
    assert kernel.gradient is gradient
    assert kernel.hessian is hessian
    np.testing.assert_allclose(gradient(x, y), 2 * (1 + x @ y) * y)
    np.testing.assert_allclose(hessian(x, y), 2 * np.outer(y, y))
    np.testing.assert_allclose(kernel.mixed_hessian(x, y), 2 * (np.outer(y, x) + (1 + x @ y) * np.eye(2)))
    np.testing.assert_allclose(kernel.diagonal_gradient(x), 4 * (1 + x @ x) * x)
    np.testing.assert_allclose(kernel.diagonal_hessian(x), 8 * np.outer(x, x) + 4 * (1 + x @ x) * np.eye(2))


def test_pairwise_evaluators_preserve_population_and_derivative_axes() -> None:
    kernel = RBF(theta=0.7)
    X = np.array([[0.3, -0.5], [0.8, 0.4], [-0.2, 0.1]])
    Y = np.array([[0.1, 0.4], [-0.4, 0.6]])
    difference = X[:, None, :] - Y[None, :, :]
    values = np.exp(-kernel.theta * np.sum(difference**2, axis=-1))
    gradient = -2 * kernel.theta * values[..., None] * difference
    outer = difference[..., :, None] * difference[..., None, :]
    hessian = values[..., None, None] * (
        4 * kernel.theta**2 * outer - 2 * kernel.theta * np.eye(2)
    )

    np.testing.assert_allclose(kernel.pairwise(X, Y), values)
    np.testing.assert_allclose(kernel.pairwise_gradient(X, Y), gradient)
    np.testing.assert_allclose(kernel.pairwise_hessian(X, Y), hessian)
    np.testing.assert_allclose(kernel.pairwise_mixed_hessian(X, Y), -hessian)
    # Aligned-pair derivatives must agree with the selected all-pairs entries.
    indices = np.arange(len(Y))
    np.testing.assert_allclose(kernel.matched(X[:2], Y), values[indices, indices])
    np.testing.assert_allclose(kernel.matched_gradient(X[:2], Y), gradient[indices, indices])
    np.testing.assert_allclose(kernel.matched_hessian(X[:2], Y), hessian[indices, indices])


def test_matched_and_diagonal_batches_have_distinct_pairing_and_derivatives() -> None:
    kernel = CallableKernel(lambda x, y: (1 + jnp.dot(x, y)) ** 2)
    X = np.array([[0.3, -0.5], [0.8, 0.4], [-0.2, 0.1]])
    Y = X[::-1].copy()  # Preserve the supplied order, even though X matches itself exactly.

    matched = 1 + np.sum(X * Y, axis=1)
    diagonal = 1 + np.sum(X**2, axis=1)
    np.testing.assert_allclose(kernel.matched(X, Y), matched**2)
    np.testing.assert_allclose(kernel.matched_gradient(X, Y), 2 * matched[:, None] * Y)
    np.testing.assert_allclose(kernel.matched_hessian(X, Y), 2 * np.einsum("ni,nj->nij", Y, Y))
    np.testing.assert_allclose(kernel.diagonal_batch(X), diagonal**2)
    np.testing.assert_allclose(kernel.diagonal_gradient_batch(X), 4 * diagonal[:, None] * X)
    np.testing.assert_allclose(
        kernel.diagonal_hessian_batch(X),
        8 * np.einsum("ni,nj->nij", X, X) + 4 * diagonal[:, None, None] * np.eye(2),
    )


def test_population_evaluators_are_cached_and_survive_serialization() -> None:
    kernel = RationalQuadratic(theta=0.7, alpha=1.3)
    X = np.array([[0.3, -0.5], [0.8, 0.4]])
    Y = np.array([[0.1, 0.4], [-0.4, 0.6]])
    pairwise = kernel.pairwise
    assert kernel.pairwise is pairwise
    values = {
        name: getattr(kernel, name)(X, Y)
        for name in (
            "pairwise", "pairwise_gradient", "pairwise_hessian", "pairwise_mixed_hessian",
            "matched", "matched_gradient", "matched_hessian",
        )
    }
    diagonal_values = {
        name: getattr(kernel, name)(X)
        for name in ("diagonal_batch", "diagonal_gradient_batch", "diagonal_hessian_batch")
    }

    restored = pickle.loads(pickle.dumps(kernel))
    for name, value in values.items():
        np.testing.assert_allclose(getattr(restored, name)(X, Y), value)
    for name, value in diagonal_values.items():
        np.testing.assert_allclose(getattr(restored, name)(X), value)


def test_mmd_accepts_a_two_argument_kernel_without_hyperparameters() -> None:
    def kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, y)

    points = np.array([[0.1, 0.3], [0.4, 0.5]])
    reference = np.array([[0.2, 0.7], [0.8, 0.1]])
    indicator = MMD(2, 2, reference, kernel=kernel)
    result = indicator.compute_hessian(points)

    difference = points.mean(axis=0) - reference.mean(axis=0)
    assert indicator.compute(Y=points) == pytest.approx(difference @ difference)
    np.testing.assert_allclose(result["MMDdY"], np.tile(difference, (2, 1)))
    np.testing.assert_allclose(result["MMDdY2"], np.kron(np.ones((2, 2)), np.eye(2) / 2))
