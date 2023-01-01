import numpy as np

from hvd import HypervolumeDerivatives

np.set_printoptions(edgeitems=30, linewidth=100000)

np.random.seed(42)

dim = 4
c1 = np.array([1, 0, 0, 0])
c2 = np.array([0, 1, 0, 0])
c3 = np.array([0, 0, 1, 0])
c4 = np.array([0, 0, 0, 1])


def MOP1(x):
    x = np.array(x)
    return np.array(
        [
            np.sum((x - c1) ** 2),
            np.sum((x - c2) ** 2),
            np.sum((x - c3) ** 2),
            np.sum((x - c4) ** 2),
        ]
    )


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - c1), 2 * (x - c2), 2 * (x - c3), 2 * (x - c4)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(dim)] * 4)


N = 10
for _ in range(10):
    w = np.random.rand(N, 4) - 0.2
    w /= np.sum(w, axis=1).reshape(-1, 1)
    X = w @ np.vstack([c1, c2, c3, c4])
    Y = np.array([MOP1(x) for x in X])
    ref = Y.max(axis=0) * 1.2

    hvh = HypervolumeDerivatives(
        n_decision_var=dim,
        n_objective=4,
        ref=ref,
        func=MOP1,
        jac=MOP1_Jacobian,
        hessian=MOP1_Hessian,
    )
    out = hvh.compute(X)
    AD = hvh.compute_automatic_differentiation(X)

    assert np.all(np.isclose(AD["HVdY"], out["HVdY"], atol=1e-5, rtol=1e-8))
    assert np.all(np.isclose(AD["HVdX"], out["HVdX"], atol=1e-5, rtol=1e-8))
    assert np.all(np.isclose(AD["HVdY2"], out["HVdY2"], atol=1e-5, rtol=1e-8))
    assert np.all(np.isclose(AD["HVdX2"], out["HVdX2"], atol=1e-5, rtol=1e-8))
