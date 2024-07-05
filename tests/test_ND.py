from typing import Tuple

import numpy as np
import pytest

from hvd import HypervolumeDerivatives

np.set_printoptions(edgeitems=30, linewidth=100000)

np.random.seed(42)


def MOP1(n_objective: int) -> Tuple[callable, callable, callable]:
    dim = n_objective
    C = np.eye(n_objective)

    def func(x):
        x = np.array(x)
        return np.sum((x - C) ** 2, axis=1)

    def jac(x):
        x = np.array(x)
        return 2 * (x - C)

    def hessian(x):
        x = np.array(x)
        return np.array([2 * np.eye(dim)] * n_objective)

    return func, jac, hessian


@pytest.mark.parametrize("n_objective", [4, 5, 6])
def test_4D(n_objective):
    dim = n_objective
    N = 10
    C = np.eye(n_objective)
    func, jac, hessian = MOP1(n_objective)

    for _ in range(3):
        w = np.random.rand(N, n_objective) - 0.2
        w /= np.sum(w, axis=1).reshape(-1, 1)
        X = w @ C
        Y = np.array([func(x) for x in X])
        ref = Y.max(axis=0) * 1.2

        hvh = HypervolumeDerivatives(
            n_var=dim,
            n_obj=n_objective,
            ref=ref,
            func=func,
            jac=jac,
            hessian=hessian,
        )
        out = hvh._compute_hessian(X)
        AD = hvh.compute_automatic_differentiation(X)

        assert np.all(np.isclose(AD["HVdY"], out["HVdY"], atol=1e-5, rtol=1e-8))
        assert np.all(np.isclose(AD["HVdX"], out["HVdX"], atol=1e-5, rtol=1e-8))
        assert np.all(np.isclose(AD["HVdY2"], out["HVdY2"], atol=1e-5, rtol=1e-8))
        assert np.all(np.isclose(AD["HVdX2"], out["HVdX2"], atol=1e-5, rtol=1e-8))
