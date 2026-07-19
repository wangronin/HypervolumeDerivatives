import sys

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit

sys.path.insert(0, "./")

from hvd.mmd import MMD, MMDMatching

np.random.seed(42)


def MOP1(x):
    x = jnp.array(x)
    return jnp.array([jnp.sum((x - 1) ** 2), jnp.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    dim = len(x)
    return np.array([2 * np.eye(dim), 2 * np.eye(dim)])


@jit
def kernel(x: np.ndarray, y: np.ndarray, theta: float = 1.0) -> float:
    return jnp.exp(-theta * jnp.sum((x - y) ** 2))


# @jit
# def kernel(x: np.ndarray, y: np.ndarray, theta: float = 1.0, alpha: float = 1.0) -> float:
#     return (jnp.sum((x - y) ** 2) * theta / (2 * alpha) + 1) ** (-alpha)


def distance(reference_set):
    def __mmd__(Y):
        XX = jnp.array([kernel(r1, r2) for r1 in reference_set for r2 in reference_set])
        YY = jnp.array([kernel(y1, y2) for y1 in Y for y2 in Y])
        XY = jnp.array([kernel(r, y) for r in reference_set for y in Y])
        return jnp.mean(XX) + jnp.mean(YY) - 2 * jnp.mean(XY)

    return __mmd__


def distance2(func, reference_set):
    def __mmd__(X):
        Y = jnp.array([func(x) for x in X])
        XX = jnp.array([kernel(r1, r2) for r1 in reference_set for r2 in reference_set])
        YY = jnp.array([kernel(y1, y2) for y1 in Y for y2 in Y])
        XY = jnp.array([kernel(r, y) for r in reference_set for y in Y])
        return jnp.mean(XX) + jnp.mean(YY) - 2 * jnp.mean(XY)

    return __mmd__


N = 3
dim = 5
X = np.random.randn(N, dim)
Y = np.array([MOP1(_) for _ in X])
p = np.linspace(0, 1, 50)
ref_X = np.c_[p, 1 - p]
ref = np.array([MOP1(_) for _ in ref_X])


def test_2D_objective_space_against_ad():
    mmd = MMD(n_var=2, n_obj=2, ref=ref, func=lambda x: x, jac=lambda _: np.diag([1, 1]))
    value = mmd.compute(X=Y)
    grad = mmd.compute_gradient(X=Y)["MMDdX"]
    H = mmd.compute_hessian(X=Y)["MMDdX2"]
    f = distance(ref)
    jac = jacrev(f)
    hess = jacfwd(jac)
    grad_ad = jac(Y)
    H_ = hess(Y)
    H_ad = np.zeros((N * 2, N * 2))
    for i in range(N):
        for j in range(N):
            H_ad[slice(i * 2, (i + 1) * 2), slice(j * 2, (j + 1) * 2)] = H_[i, :, j, :]

    assert np.isclose(value, float(f(Y)))
    assert np.all(np.isclose(grad, grad_ad))
    assert np.all(np.isclose(H, H_ad))


def test_2D_decision_space_against_ad():
    mmd = MMD(n_var=dim, n_obj=2, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian)
    value = mmd.compute(Y=Y)
    grad = mmd.compute_gradient(X=X)["MMDdX"]
    H = mmd.compute_hessian(X=X)["MMDdX2"]
    f = distance2(MOP1, ref)
    jac = jacrev(f)
    hess = jacfwd(jac)
    grad_ad = jac(X)
    H_ = hess(X)
    H_ad = np.zeros((N * dim, N * dim))
    for i in range(N):
        for j in range(N):
            H_ad[slice(i * dim, (i + 1) * dim), slice(j * dim, (j + 1) * dim)] = H_[i, :, j, :]

    assert np.isclose(value, float(f(X)))
    assert np.all(np.isclose(grad, grad_ad))
    assert np.all(np.isclose(H, H_ad))


def test_gradient_does_not_evaluate_objective_hessian():
    def unexpected_hessian(_):
        raise AssertionError("gradient-only evaluation must not call the objective Hessian")

    mmd = MMD(
        n_var=2,
        n_obj=2,
        ref=ref,
        func=lambda x: x,
        jac=lambda _: np.eye(2),
        hessian=unexpected_hessian,
    )
    gradient = mmd.compute_gradient(X=Y)["MMDdX"]
    assert gradient.shape == Y.shape


def test_supplied_jacobian_is_used_for_hessian_chain_rule():
    zero_jacobian = np.zeros((N, 2, dim))
    mmd = MMD(n_var=dim, n_obj=2, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian)
    gradient, _ = mmd.compute_derivatives(X, compute_hessian=True, jacobian=zero_jacobian)
    assert np.allclose(gradient, 0)


def test_matching_hessian_against_ad_with_fixed_matching():
    beta = 0.37
    mmd = MMDMatching(
        n_var=2,
        n_obj=2,
        ref=ref,
        func=lambda x: x,
        jac=lambda _: np.eye(2),
        beta=beta,
    )
    result = mmd.compute_hessian(X=Y)
    matched_reference = jnp.asarray(mmd.ref.reference_set.copy())

    def fixed_matching_indicator(points):
        yy = jnp.array([kernel(y1, y2) for y1 in points for y2 in points])
        matched = jnp.array([kernel(y, r) for y, r in zip(points, matched_reference)])
        return beta * jnp.mean(yy) + jnp.mean(2 - 2 * matched)

    hessian = jacfwd(jacrev(fixed_matching_indicator))(Y)
    expected = hessian.reshape(N * 2, N * 2)

    assert np.allclose(result["MMDdY2"], expected)
