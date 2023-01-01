import autograd.numpy as np
import numpy

from hvd import HypervolumeDerivatives
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


hvh = HypervolumeDerivatives(
    n_decision_var=3, n_objective=3, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian
)


def test_against_autograd():
    for _ in range(5):
        w = np.random.rand(20, 3)
        w /= np.sum(w, axis=1).reshape(-1, 1)
        X = w @ np.vstack([c1, c2, c3])
        out = hvh.compute(X)
        AD = hvh.compute_automatic_differentiation(X)

        assert np.all(np.isclose(AD["HVdY"], out["HVdY"]))
        assert np.all(np.isclose(AD["HVdX"], out["HVdX"]))
        assert np.all(np.isclose(AD["HVdY2"], out["HVdY2"]))
        assert np.all(np.isclose(AD["HVdX2"], out["HVdX2"]))
