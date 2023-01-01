import numpy as np

from hvd import HypervolumeDerivatives
from hvd.problems import Eq1DTLZ1

np.set_printoptions(edgeitems=30, linewidth=100000)

np.random.seed(42)

f = Eq1DTLZ1()
dim = 7

for _ in range(10):
    X = np.array([np.r_[np.random.rand(2), [0.5] * 5], np.r_[np.random.rand(2), [0.5] * 5]])
    Y = np.array([f.objective(x) for x in X])
    ref = np.array([2, 2, 2])

    hvh = HypervolumeDerivatives(
        n_decision_var=dim,
        n_objective=3,
        ref=ref,
        func=f.objective,
        jac=f.objective_jacobian,
        hessian=f.objective_hessian,
    )
    out = hvh.compute(X)
    AD = hvh.compute_automatic_differentiation(X)

    assert np.all(np.isclose(AD["HVdY"], out["HVdY"]))
    assert np.all(np.isclose(AD["HVdX"], out["HVdX"]))
    assert np.all(np.isclose(AD["HVdY2"], out["HVdY2"]))
    assert np.all(np.isclose(AD["HVdX2"], out["HVdX2"]))
