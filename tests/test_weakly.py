import sys

import numpy as np

sys.path.insert(0, "./")
from hvd import HypervolumeDerivatives

np.random.seed(42)


def test_weakly_dominated_points():
    ref = np.array([11, 10])
    hvh = HypervolumeDerivatives(2, 2, ref, minimization=True)
    X = np.array([[6.5, 2.5], [5.125, 3.125], [3.125, 3.125], [3.125, 5.125], [2.5, 6.5]])
    out = hvh.compute_gradient(X=X)


test_weakly_dominated_points()
