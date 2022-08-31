import autograd.numpy as np
import numpy
from autograd import hessian, jacobian
from hvd import HypervolumeDerivatives
from hvd.hypervolume import hypervolume
from hvd.utils import get_non_dominated

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000)

c1 = np.array([1.5, 0, np.sqrt(3) / 3])
c2 = np.array([1.5, 0.5, -1 * np.sqrt(3) / 6])
c3 = np.array([1.5, -0.5, -1 * np.sqrt(3) / 6])
ref = np.array([24, 24, 24])


def MOP1(x):
    x = np.array(x)
    return np.array(
        [
            np.sum((x - c1) ** 2),
            np.sum((x - c2) ** 2),
            np.sum((x - c3) ** 2),
        ]
    )


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array(
        [
            2 * (x - c1),
            2 * (x - c2),
            2 * (x - c3),
        ]
    )


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(3), 2 * np.eye(3), 2 * np.eye(3)])


def _hypervolumeY(y):
    return hypervolume(y.reshape(-1, 3), ref)


def _hypervolumeX(x):
    return hypervolume(np.array([MOP1(x_) for x_ in x.reshape(-1, 3)]), ref)


HV_Jac = jacobian(_hypervolumeY)
HV_Hessian = hessian(_hypervolumeY)
HVX_Jac = jacobian(_hypervolumeX)
HVX_Hessian = hessian(_hypervolumeX)

hvh = HypervolumeDerivatives(dim_d=3, dim_m=3, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian)


def test_against_autograd():
    for _ in range(10):
        # w = np.random.rand(5, 3)
        # w /= np.sum(w, axis=1).reshape(-1, 1)
        # X = w @ np.vstack([c1, c2, c3])
        # X = np.random.rand(1000, 3)
        # a = numpy.mgrid[-0.5:0.5:7j, -0.5:0.5:7j]

        # X = numpy.c_[numpy.tile(1.5, (49, 1)), numpy.array(list(zip(a[0].ravel(), a[1].ravel())))]
        X = numpy.c_[numpy.tile(1.5, (50, 1)), numpy.random.rand(50, 2) - 0.5]
        Y = numpy.array([MOP1(x) for x in X])
        idx = get_non_dominated(Y, return_index=True, weakly_dominated=False)
        X, Y = X[idx], Y[idx]
        out = hvh.compute_hessian(X)

        assert np.all(np.isclose(HV_Jac(Y.ravel()), out["HVdY"]))
        assert np.all(np.isclose(HV_Hessian(Y.ravel()), out["HVdY2"]))
        assert np.all(np.isclose(HVX_Jac(X.ravel()), out["HVdX"]))
        assert np.all(np.isclose(HVX_Hessian(X.ravel()), out["HVdX2"]))


# HVdY_FD = hvh.compute_HVdY_FD(Y)
# HVdY2_FD = hvh.compute_HVdY2_FD(Y)
# HVdX_FD = hvh.compute_HVdX_FD(X)
# HVdX2_FD = hvh.compute_HVdX2_FD(X)
# assert np.all(np.isclose(HVdY_FD.ravel(), out["HVdY"], atol=1e-3, rtol=1e-3))
# assert np.all(np.isclose(HVdX_FD.ravel(), out["HVdX"], atol=1e-2, rtol=1e-2))
# assert np.all(np.isclose(HVdY2_FD, out["HVdY2"], atol=1e-3, rtol=1e-3))
# assert np.all(np.isclose(np.linalg.eigh(out["HVdX2"])[0], np.linalg.eigh(HVdX2_FD)[0], atol=1e-2, rtol=1e-2))
