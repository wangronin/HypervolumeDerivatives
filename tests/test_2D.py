import sys

import numpy as np

sys.path.insert(0, "./")
from hvd import HypervolumeDerivatives

np.random.seed(42)


def test_2D_case1():
    ref = np.array([11, 10])
    hvh = HypervolumeDerivatives(2, 2, ref, minimization=True)
    out = hvh.compute_hessian(X=np.array([(10, 1), (9.5, 3), (8, 6.5), (4, 8), (1, 9)]))
    assert np.all(out["HVdY"] == np.array([-2, -1, -3.5, -0.5, -1.5, -1.5, -1, -4, -1, -3]))
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, -1, 0, 0],
                [0, 0, -1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
                [0, 0, 0, 0, -1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, -1, 0, 1, 0],
            ]
        )
    )


def MOP1(x):
    x = np.array(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(2), 2 * np.eye(2)])


def test_2D_case2():
    ref = np.array([20, 20])
    hvh = HypervolumeDerivatives(
        dim_d=2, dim_m=2, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian, minimization=True
    )
    for _ in range(10):
        X = np.random.rand(3, 2)
        out = hvh.compute_hessian(X)
        HVdX_FD = hvh.compute_HVdX_FD(X).ravel()
        # HVdX2_FD = hvh.compute_HVdX2_FD(X)
        assert np.all(np.isclose(out["HVdX"], HVdX_FD, atol=1e-2))
        # if not np.all(np.isclose(out["HVdX2"], HVdX2_FD, atol=1e-2)):
