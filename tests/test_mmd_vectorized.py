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


def indicator_kwargs(kernel, theta):
    return dict(
        n_var=X.shape[1],
        n_obj=2,
        ref=REFERENCE_SET.copy(),
        func=objective,
        jac=objective_jacobian,
        hessian=objective_hessian,
        kernel=kernel,
        theta=theta,
    )


class TestMMD:
    @pytest.mark.parametrize("kernel,theta", KERNELS)
    def test_value_gradient_and_hessian_against_ad(self, kernel, theta):
        indicator = MMD(**indicator_kwargs(kernel, theta))
        result = indicator.compute_hessian(X)
        parameterized_kernel = partial(kernel, theta=theta)
        reference_set = jnp.asarray(REFERENCE_SET)

        def indicator_value(points):
            images = vmap(objective)(points)
            return (
                pairwise(parameterized_kernel, reference_set, reference_set).mean()
                + pairwise(parameterized_kernel, images, images).mean()
                - 2 * pairwise(parameterized_kernel, images, reference_set).mean()
            )

        expected_value = indicator_value(jnp.asarray(X))
        expected_gradient = jacrev(indicator_value)(jnp.asarray(X))
        expected_hessian = flatten_hessian(jacfwd(jacrev(indicator_value))(jnp.asarray(X)))

        assert np.isclose(indicator.compute(X=X), expected_value)
        assert np.allclose(result["MMDdX"], expected_gradient, rtol=RTOL, atol=ATOL)
        assert np.allclose(result["MMDdX2"], expected_hessian, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("kernel,theta", KERNELS)
    def test_legacy_behavior(self, kernel, theta):
        kwargs = indicator_kwargs(kernel, theta)
        legacy = LegacyMMD(**kwargs)
        vectorized = MMD(**kwargs)
        legacy_result = legacy.compute_hessian(X)
        vectorized_result = vectorized.compute_hessian(X)

        assert np.isclose(vectorized.compute(X=X), legacy.compute(X=X))
        if kernel is not laplace:
            assert np.allclose(
                vectorized_result["MMDdX"], legacy_result["MMDdX"], rtol=RTOL, atol=ATOL
            )
            assert np.allclose(
                vectorized_result["MMDdX2"], legacy_result["MMDdX2"], rtol=RTOL, atol=ATOL
            )


class TestMMDMatching:
    @pytest.mark.parametrize("kernel,theta", KERNELS)
    def test_value_gradient_and_hessian_against_ad(self, kernel, theta):
        beta = 0.37
        indicator = MMDMatching(**indicator_kwargs(kernel, theta), beta=beta)
        result = indicator.compute_hessian(X)
        matched_reference_set = jnp.asarray(indicator.ref.reference_set.copy())
        parameterized_kernel = partial(kernel, theta=theta)

        def indicator_value(points):
            images = vmap(objective)(points)
            squared_rkhs_distance = vmap(
                lambda y, r: parameterized_kernel(y, y)
                + parameterized_kernel(r, r)
                - 2 * parameterized_kernel(y, r)
            )(images, matched_reference_set)
            return (
                beta * pairwise(parameterized_kernel, images, images).mean()
                + squared_rkhs_distance.mean()
            )

        expected_value = indicator_value(jnp.asarray(X))
        expected_gradient = jacrev(indicator_value)(jnp.asarray(X))
        expected_hessian = flatten_hessian(jacfwd(jacrev(indicator_value))(jnp.asarray(X)))

        assert np.isclose(indicator.compute(X=X), expected_value)
        assert np.allclose(result["MMDdX"], expected_gradient, rtol=RTOL, atol=ATOL)
        assert np.allclose(result["MMDdX2"], expected_hessian, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("kernel,theta", KERNELS)
    def test_legacy_behavior(self, kernel, theta):
        kwargs = indicator_kwargs(kernel, theta) | {"beta": 0.37}
        legacy = LegacyMMDMatching(**kwargs)
        vectorized = MMDMatching(**kwargs)
        legacy_result = legacy.compute_hessian(X)
        vectorized_result = vectorized.compute_hessian(X)

        value_matches = kernel is not linear
        derivatives_match = kernel in (rbf, rational_quadratic)
        assert np.isclose(vectorized.compute(X=X), legacy.compute(X=X)) == value_matches
        assert (
            np.allclose(vectorized_result["MMDdX"], legacy_result["MMDdX"], rtol=RTOL, atol=ATOL)
            == derivatives_match
        )
        assert (
            np.allclose(
                vectorized_result["MMDdX2"], legacy_result["MMDdX2"], rtol=RTOL, atol=ATOL
            )
            == derivatives_match
        )
