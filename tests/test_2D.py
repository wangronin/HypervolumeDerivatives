import sys

import numpy as np

sys.path.insert(0, "./")
from hvd import HypervolumeDerivatives


def test_2D_case1():
    ref = np.array([11, 10])
    hvh = HypervolumeDerivatives(2, 2, ref, maximization=False)
    out = hvh.compute(X=np.array([(10, 1), (9.5, 3), (8, 6.5), (4, 8), (1, 9)]))
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
