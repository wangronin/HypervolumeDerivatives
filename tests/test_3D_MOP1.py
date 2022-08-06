import numpy as np
from hvd import HypervolumeDerivatives

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000)

dim = 3


def MOP1(x):
    x = np.array(x)
    return np.array(
        [
            np.sum((x - np.array([1, 1, 0])) ** 2),
            np.sum((x - np.array([1, -1, 0])) ** 2),
            np.sum((x - np.array([-1, 0, 0])) ** 2),
        ]
    )


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array(
        [2 * (x - np.array([1, 1, 0])), 2 * (x - np.array([1, -1, 0])), 2 * (x - np.array([-1, 0, 0]))]
    )


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(dim), 2 * np.eye(dim), 2 * np.eye(dim)])


X = np.random.rand(20, dim)
Y = np.array([MOP1(x) for x in X])
ref = np.array([20, 20, 20])

hvh = HypervolumeDerivatives(dim_d=dim, dim_m=3, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian)
out = hvh.compute(X)

HVdY_FD = hvh.compute_HVdY_FD(Y)
HVdY2_FD = hvh.compute_HVdY2_FD(Y)
HVdX_FD = hvh.compute_HVdX_FD(X)
HVdX2_FD = hvh.compute_HVdX2_FD(X)

assert np.all(np.isclose(HVdY_FD.ravel(), out["HVdY"], atol=1e-3, rtol=1e-3))
assert np.all(np.isclose(HVdX_FD.ravel(), out["HVdX"], atol=1e-2, rtol=1e-2))
assert np.all(np.isclose(HVdY2_FD, out["HVdY2"], atol=1e-3, rtol=1e-3))
assert np.all(np.isclose(np.linalg.eigh(out["HVdX2"])[0], np.linalg.eigh(HVdX2_FD)[0], atol=1e-2, rtol=1e-2))
