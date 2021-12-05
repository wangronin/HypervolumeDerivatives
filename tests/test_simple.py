import numpy as np
from hvh import HypervolumeHessian


def test_3D_case1():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeHessian(3, 3, ref, maximization=False)
    out = hvh.compute(X=np.array([[5, 1, 7], [2, 3, 10]]))
    assert np.all(
        out["HdY2"]
        == np.array(
            [
                [0, 5, 9, 0, -2, -7],
                [5, 0, 4, 0, 0, -0],
                [9, 4, 0, 0, -0, 0],
                [0, 0, 0, 0, 2, 7],
                [-2, 0, 0, 2, 0, 3],
                [-7, 0, 0, 7, 3, 0],
            ]
        )
    )


def test_3D_case2():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeHessian(3, 3, ref, maximization=False)
    out = hvh.compute(X=np.array([[-1, 5, 7], [2, 3, 10]]))
    assert np.all(
        out["HdY2"]
        == np.array(
            [
                [0, 5, 5, 0, 0, -0],
                [5, 0, 10, -2, 0, -7],
                [5, 10, 0, -0, 0, 0],
                [0, -2, 0, 0, 2, 2],
                [0, 0, 0, 2, 0, 7],
                [0, -7, 0, 2, 7, 0],
            ]
        )
    )


def test_3D_case3():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeHessian(3, 3, ref, maximization=False)
    out = hvh.compute(X=np.array([[5, 3, 7], [2, 1, 10]]))
    assert np.all(
        out["HdY2"]
        == np.array(
            [
                [0, 3, 7, 0, 0, -7],
                [3, 0, 4, 0, 0, -4],
                [7, 4, 0, 0, 0, 0],
                [0, -0, 0, 0, 2, 9],
                [-0, 0, 0, 2, 0, 7],
                [-7, -4, 0, 9, 7, 0],
            ]
        )
    )


def test_with_dominated_points():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeHessian(3, 3, ref, maximization=False)
    out = hvh.compute(X=np.array([[-1, -2, 7], [2, 1, 10]]))
    assert np.all(
        out["HdY2"]
        == np.array(
            [
                [0, 5, 12, 0, 0, 0],
                [5, 0, 10, 0, 0, 0],
                [12, 10, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )
