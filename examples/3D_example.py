import numpy as np
from hvd import HypervolumeDerivatives, get_non_dominated

np.random.seed(42)


def func(x):
    x = np.array(x)
    return np.array(
        [np.sum(x ** 2), np.sum((x - np.array([1] * 3)) ** 2), np.sum((x - np.array([3] * 3)) ** 2)]
    )


def jac(x):
    x = np.array(x)
    return np.array([2 * x, 2 * (x - np.array([1] * 3)), (x - np.array([3] * 3))])


def hessian(x):
    d = len(x)
    x = np.array(x)
    return np.array([2 * np.eye(d), 2 * np.eye(d), 2 * np.eye(d)])


ref = np.array([0, 0, 0])
hvh = HypervolumeDerivatives(dim_d=3, dim_m=3, ref=ref, func=func, jac=jac, hessian=hessian)
out = hvh.compute(X=np.random.rand(2, 3))
HdY2, HdX2 = out["HVdY2"], out["HVdX2"]
print(HdY2)
print(HdX2)
assert np.allclose(HdY2, HdY2.T)
assert np.allclose(HdX2, HdX2.T)
