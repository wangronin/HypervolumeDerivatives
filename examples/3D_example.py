import numpy as np
from hvd import HypervolumeDerivatives
from hvd.problems import Eq1DTLZ1

np.random.seed(42)

f = Eq1DTLZ1()
dim = 7
X = np.random.rand(2, dim)
Y = np.array([f.objective(x) for x in X])
ref = np.array([300, 300, 300])

hvh = HypervolumeDerivatives(
    dim_d=dim, dim_m=3, ref=ref, func=f.objective, jac=f.objective_jacobian, hessian=f.objective_hessian
)
out = hvh.compute(X)

HVdY_FD = hvh.compute_HVdY_FD(Y)
HVdY2_FD = hvh.compute_HVdY2_FD(Y)
HVdX_FD = hvh.compute_HVdX_FD(X)
HVdX2_FD = hvh.compute_HVdX2_FD(X)

assert np.all(np.isclose(HVdY_FD.ravel(), out["HVdY"]))
assert np.all(np.isclose(HVdY2_FD, out["HVdY2"], atol=1e-3))
assert np.all(np.isclose(HVdX2_FD.ravel(), out["HVdX2"]))
