from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacfwd, jacrev, vmap

from hvd.mmd import MMD as LegacyMMD
from hvd.mmd import MMDMatching as LegacyMMDMatching
from hvd.mmd import laplace, linear, rational_quadratic, rbf
from hvd.mmd_vectorized import MMD, MMDMatching


KERNELS = [
    pytest.param(rbf, 0.7, id="rbf"),
    pytest.param(rational_quadratic, 0.7, id="rational_quadratic"),
    pytest.param(laplace, 0.7, id="laplace"),
    pytest.param(linear, 0.7, id="linear"),
]

X = np.array(
    [
        [-0.8, 0.2, 0.7],
        [0.1, -0.5, 0.9],
        [0.6, 0.3, -0.4],
    ]
)
REFERENCE_SET = np.array(
    [
        [-0.4, 0.8],
        [0.2, -0.7],
        [0.9, 0.3],
        [-0.6, -0.2],
        [0.5, 0.6],
    ]
)

# JAX uses float32 in this project. Vectorized reductions and the legacy Python
# loops sum terms in a different order, so agreement at a few ulps is expected.
RTOL = 2e-6
ATOL = 3e-7


def objective(x):
    x = jnp.asarray(x)
    return jnp.array([jnp.sum((x - 0.5) ** 2), jnp.sum((x + 0.25) ** 2)])


def objective_jacobian(x):
    x = np.asarray(x)
    return np.array([2 * (x - 0.5), 2 * (x + 0.25)])


def objective_hessian(x):
    return np.array([2 * np.eye(len(x)), 2 * np.eye(len(x))])


def pairwise(kernel, A, B):
    return vmap(vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))(A, B)


def flatten_hessian(hessian):
    n_points, n_var = hessian.shape[:2]
    return np.asarray(hessian).reshape(n_points * n_var, n_points * n_var)


@pytest.mark.parametrize("kernel,theta", KERNELS)
def test_vectorized_mmd_value_gradient_and_hessian_against_ad(kernel, theta):
    indicator = MMD(
        n_var=X.shape[1],
        n_obj=2,
        ref=REFERENCE_SET.copy(),
        func=objective,
        jac=objective_jacobian,
        hessian=objective_hessian,
        kernel=kernel,
        theta=theta,
    )
    result = indicator.compute_hessian(X)
    kernel = partial(kernel, theta=theta)
    reference_set = jnp.asarray(REFERENCE_SET)

    def indicator_value(points):
        images = vmap(objective)(points)
        return (
            pairwise(kernel, reference_set, reference_set).mean()
            + pairwise(kernel, images, images).mean()
            - 2 * pairwise(kernel, images, reference_set).mean()
        )

    expected_value = indicator_value(jnp.asarray(X))
    expected_gradient = jacrev(indicator_value)(jnp.asarray(X))
    expected_hessian = flatten_hessian(jacfwd(jacrev(indicator_value))(jnp.asarray(X)))

    assert np.isclose(indicator.compute(X=X), expected_value)
    assert np.allclose(result["MMDdX"], expected_gradient, rtol=RTOL, atol=ATOL)
    assert np.allclose(result["MMDdX2"], expected_hessian, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kernel,theta", KERNELS)
def test_vectorized_matching_value_gradient_and_hessian_against_ad(kernel, theta):
    beta = 0.37
    indicator = MMDMatching(
        n_var=X.shape[1],
        n_obj=2,
        ref=REFERENCE_SET.copy(),
        func=objective,
        jac=objective_jacobian,
        hessian=objective_hessian,
        kernel=kernel,
        theta=theta,
        beta=beta,
    )
    result = indicator.compute_hessian(X)
    matched_reference_set = jnp.asarray(indicator.ref.reference_set.copy())
    kernel = partial(kernel, theta=theta)

    def indicator_value(points):
        images = vmap(objective)(points)
        squared_rkhs_distance = vmap(
            lambda y, r: kernel(y, y) + kernel(r, r) - 2 * kernel(y, r)
        )(images, matched_reference_set)
        return beta * pairwise(kernel, images, images).mean() + squared_rkhs_distance.mean()

    expected_value = indicator_value(jnp.asarray(X))
    expected_gradient = jacrev(indicator_value)(jnp.asarray(X))
    expected_hessian = flatten_hessian(jacfwd(jacrev(indicator_value))(jnp.asarray(X)))

    assert np.isclose(indicator.compute(X=X), expected_value)
    assert np.allclose(result["MMDdX"], expected_gradient, rtol=RTOL, atol=ATOL)
    assert np.allclose(result["MMDdX2"], expected_hessian, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "kernel,theta",
    [
        pytest.param(rbf, 0.7, id="rbf"),
        pytest.param(rational_quadratic, 0.7, id="rational_quadratic"),
        pytest.param(linear, 0.7, id="linear"),
    ],
)
def test_vectorized_mmd_matches_legacy_for_smooth_kernels(kernel, theta):
    kwargs = dict(
        n_var=X.shape[1],
        n_obj=2,
        ref=REFERENCE_SET.copy(),
        func=objective,
        jac=objective_jacobian,
        hessian=objective_hessian,
        kernel=kernel,
        theta=theta,
    )
    legacy = LegacyMMD(**kwargs)
    vectorized = MMD(**kwargs)
    legacy_result = legacy.compute_hessian(X)
    vectorized_result = vectorized.compute_hessian(X)

    assert np.isclose(vectorized.compute(X=X), legacy.compute(X=X))
    assert np.allclose(vectorized_result["MMDdX"], legacy_result["MMDdX"], rtol=RTOL, atol=ATOL)
    assert np.allclose(vectorized_result["MMDdX2"], legacy_result["MMDdX2"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "kernel,theta",
    [
        pytest.param(rbf, 0.7, id="rbf"),
        pytest.param(rational_quadratic, 0.7, id="rational_quadratic"),
    ],
)
def test_vectorized_matching_matches_legacy_for_smooth_unit_diagonal_kernels(kernel, theta):
    kwargs = dict(
        n_var=X.shape[1],
        n_obj=2,
        ref=REFERENCE_SET.copy(),
        func=objective,
        jac=objective_jacobian,
        hessian=objective_hessian,
        kernel=kernel,
        theta=theta,
        beta=0.37,
    )
    legacy = LegacyMMDMatching(**kwargs)
    vectorized = MMDMatching(**kwargs)
    legacy_result = legacy.compute_hessian(X)
    vectorized_result = vectorized.compute_hessian(X)

    assert np.isclose(vectorized.compute(X=X), legacy.compute(X=X))
    assert np.allclose(vectorized_result["MMDdX"], legacy_result["MMDdX"], rtol=RTOL, atol=ATOL)
    assert np.allclose(vectorized_result["MMDdX2"], legacy_result["MMDdX2"], rtol=RTOL, atol=ATOL)


def test_linear_matching_uses_the_general_squared_rkhs_distance():
    theta = 0.7
    indicator = MMDMatching(
        n_var=X.shape[1],
        n_obj=2,
        ref=REFERENCE_SET.copy(),
        func=objective,
        jac=objective_jacobian,
        hessian=objective_hessian,
        kernel=linear,
        theta=theta,
    )
    Y = np.asarray(vmap(objective)(jnp.asarray(X)))
    indicator.ref.match(Y)
    matched = indicator.ref.reference_set
    expected_distances = theta * np.sum((Y - matched) ** 2, axis=1)
    computed_distances = np.array(
        [indicator.k(y, y) + indicator.k(r, r) - 2 * indicator.k(y, r) for y, r in zip(Y, matched)]
    )

    assert np.allclose(computed_distances, expected_distances)
