import numpy as np
from hvd import HypervolumeDerivatives
from hvd.problems import Eq1DTLZ1

np.set_printoptions(edgeitems=30, linewidth=100000)

np.random.seed(42)

f = Eq1DTLZ1()
dim = 7

for _ in range(10):
    # X = np.random.rand(2, dim)
    X = np.array([np.r_[np.random.rand(2), [0.5] * 5], np.r_[np.random.rand(2), [0.5] * 5]])
    Y = np.array([f.objective(x) for x in X])
    ref = np.array([2, 2, 2])

    hvh = HypervolumeDerivatives(
        dim_d=dim, dim_m=3, ref=ref, func=f.objective, jac=f.objective_jacobian, hessian=f.objective_hessian
    )
    out = hvh.compute_hessian(X)

    HVdY_FD = hvh.compute_HVdY_FD(Y)
    HVdY2_FD = hvh.compute_HVdY2_FD(Y)
    HVdX_FD = hvh.compute_HVdX_FD(X)
    HVdX2_FD = hvh.compute_HVdX2_FD(X)

    if not np.all(np.isclose(HVdY_FD.ravel(), out["HVdY"], atol=1e-3, rtol=1e-3)):
        breakpoint()
    if not np.all(np.isclose(HVdX_FD.ravel(), out["HVdX"], atol=1e-2, rtol=1e-2)):
        breakpoint()
    if not np.all(np.isclose(HVdY2_FD, out["HVdY2"], atol=1e-3, rtol=1e-3)):
        breakpoint()
    if not np.all(
        np.isclose(np.linalg.eigh(out["HVdX2"])[0], np.linalg.eigh(HVdX2_FD)[0], atol=1e-2, rtol=1e-2)
    ):
        breakpoint()
    breakpoint()
    # if not np.all(np.isclose(HVdX2_FD, out["HVdX2"], atol=1e-3, rtol=1e-3)):
    # breakpoint()
