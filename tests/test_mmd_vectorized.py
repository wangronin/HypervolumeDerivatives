from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacfwd, jacrev, vmap

from hvd.mmd import MMD, MMDMatching
from hvd.mmd.kernels import RBF, CallableKernel, Laplace, RationalQuadratic
from hvd.mmd.legacy import MMD as LegacyMMD
from hvd.mmd.legacy import MMDMatching as LegacyMMDMatching
from hvd.mmd.legacy import laplace, rational_quadratic, rbf
from hvd.reference_set import ReferenceSet


def polynomial(x, y, theta=1.0):
    return (1 + theta * jnp.dot(x, y)) ** 2


KERNELS = [
    pytest.param(rbf, 0.7, id="rbf"),
    pytest.param(rational_quadratic, 0.7, id="rational_quadratic"),
    pytest.param(laplace, 0.7, id="laplace"),
    pytest.param(polynomial, 0.7, id="custom_polynomial"),
]
KERNEL_CLASSES = {rbf: RBF, rational_quadratic: RationalQuadratic, laplace: Laplace}

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

# Vectorized reductions and the legacy Python loops sum terms in a different
# order, so allow small backend-dependent rounding differences.
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
        kernel=(
            KERNEL_CLASSES[kernel](theta=theta)
            if kernel in KERNEL_CLASSES else CallableKernel(partial(kernel, theta=theta))
        ),
    )


@pytest.mark.parametrize("indicator_type", [MMD, MMDMatching])
def test_evaluation_does_not_shift_reference_points(indicator_type):
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    indicator = indicator_type(2, 2, ReferenceSet(reference.copy(), eta={0: -np.ones(2)}))
    indicator.compute(Y=reference)
    indicator.compute_hessian(X=reference)
    np.testing.assert_array_equal(indicator.ref.reference_set, reference)
    assert not indicator.history_reference_set

    indicator.shift_reference_set()
    expected = reference - 0.04
    np.testing.assert_allclose(indicator.ref.reference_set, expected)
    assert set(indicator.history_reference_set) == {0, 1}
    for k, history in indicator.history_reference_set.items():
        np.testing.assert_allclose(history, expected[[k]])

    # Evaluating the model and its derivatives must not trigger another shift.
    indicator.compute(Y=reference)
    indicator.compute_hessian(X=reference)
    np.testing.assert_allclose(indicator.ref.reference_set, expected)
    assert all(len(history) == 1 for history in indicator.history_reference_set.values())


class TestMMD:
    def test_has_no_matching_switch(self):
        indicator = MMD(**indicator_kwargs(rbf, 0.7))
        assert not hasattr(indicator, "re_match")

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

    @pytest.mark.parametrize("kernel,theta", KERNELS[:-1])
    def test_legacy_behavior(self, kernel, theta):
        kwargs = indicator_kwargs(kernel, theta)
        legacy = LegacyMMD(**(kwargs | {"kernel": kernel, "theta": theta}))
        vectorized = MMD(**kwargs)
        legacy_result = legacy.compute_hessian(X)
        vectorized_result = vectorized.compute_hessian(X)

        assert np.isclose(vectorized.compute(X=X), legacy.compute(X=X))
        if kernel is not laplace:
            assert np.allclose(vectorized_result["MMDdX"], legacy_result["MMDdX"], rtol=RTOL, atol=ATOL)
            assert np.allclose(vectorized_result["MMDdX2"], legacy_result["MMDdX2"], rtol=RTOL, atol=ATOL)


class TestMMDMatching:
    def test_only_shifts_requested_matched_targets(self):
        reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        indicator = MMDMatching(2, 2, ReferenceSet(reference.copy(), eta={0: -np.ones(2)}))
        indicator.compute(Y=reference)
        indicator.shift_reference_set()
        indicator.shift_reference_set(indices=np.array([0]))

        expected = reference - 0.04
        expected[0] -= 0.04
        np.testing.assert_allclose(indicator.ref.reference_set, expected)
        assert [len(history) for history in indicator.history_reference_set.values()] == [2, 1, 1]
        np.testing.assert_allclose(indicator.history_reference_set[0][0], reference[0] - 0.04)

        indicator.shift_reference_set(indices=np.array([], dtype=int))
        np.testing.assert_allclose(indicator.ref.reference_set, expected)
        assert [len(history) for history in indicator.history_reference_set.values()] == [2, 1, 1]

    def test_trial_evaluation_reenables_matching_after_an_error(self):
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])
        indicator = MMDMatching(2, 2, reference.copy())
        indicator.compute(Y=reference)

        with pytest.raises(ValueError, match="trial failed"):
            with indicator.trial_evaluation():
                assert not indicator.re_match
                indicator.compute_gradient(X=reference[::-1])
                np.testing.assert_array_equal(indicator.ref.reference_set, reference)
                raise ValueError("trial failed")

        assert indicator.re_match
        indicator.compute(Y=reference[::-1])
        np.testing.assert_array_equal(indicator.ref.reference_set, reference[::-1])

    def test_matching_can_be_frozen_and_reenabled(self):
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])
        indicator = MMDMatching(2, 2, reference.copy())
        assert indicator.re_match
        indicator.compute(Y=reference)

        indicator.re_match = False
        assert not indicator.ref.re_match
        indicator.compute(Y=reference[::-1])
        np.testing.assert_array_equal(indicator.ref.reference_set, reference)

        indicator.re_match = True
        assert indicator.ref.re_match
        indicator.compute(Y=reference[::-1])
        np.testing.assert_array_equal(indicator.ref.reference_set, reference[::-1])

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
            return beta * pairwise(parameterized_kernel, images, images).mean() + squared_rkhs_distance.mean()

        expected_value = indicator_value(jnp.asarray(X))
        expected_gradient = jacrev(indicator_value)(jnp.asarray(X))
        expected_hessian = flatten_hessian(jacfwd(jacrev(indicator_value))(jnp.asarray(X)))

        assert np.isclose(indicator.compute(X=X), expected_value)
        assert np.allclose(result["MMDdX"], expected_gradient, rtol=RTOL, atol=ATOL)
        assert np.allclose(result["MMDdX2"], expected_hessian, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("kernel,theta", KERNELS[:-1])
    def test_legacy_behavior(self, kernel, theta):
        kwargs = indicator_kwargs(kernel, theta) | {"beta": 0.37}
        legacy = LegacyMMDMatching(**(kwargs | {"kernel": kernel, "theta": theta}))
        vectorized = MMDMatching(**kwargs)
        legacy_result = legacy.compute_hessian(X)
        vectorized_result = vectorized.compute_hessian(X)

        derivatives_match = kernel in (rbf, rational_quadratic)
        assert np.isclose(vectorized.compute(X=X), legacy.compute(X=X))
        assert (
            np.allclose(vectorized_result["MMDdX"], legacy_result["MMDdX"], rtol=RTOL, atol=ATOL)
            == derivatives_match
        )
        assert (
            np.allclose(vectorized_result["MMDdX2"], legacy_result["MMDdX2"], rtol=RTOL, atol=ATOL)
            == derivatives_match
        )
