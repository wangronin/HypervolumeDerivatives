import numpy as np
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3, Eq1DTLZ4


def test_Eq1DTLZ1():
    f = Eq1DTLZ1()
    x = np.array([0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649, 0.1576, 0.9706, 0.9572, 0.4854])
    assert np.all(
        np.isclose(f.objective(x), np.array([39.2583506542203, 363.3913996454754, 234.0513096302470]))
    )
    assert np.isclose(f.constraint(x), 0.01943601)


def test_Eq1DTLZ2():
    f = Eq1DTLZ2()
    x = np.array([0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649, 0.1576, 0.9706, 0.9572, 0.4854])
    assert np.all(
        np.isclose(f.objective(x), np.array([1.092253942326640, 0.168601869538367, 1.696393580953341]))
    )
    assert np.isclose(f.constraint(x), 0.01943601)


def test_Eq1DTLZ3():
    f = Eq1DTLZ3()
    x = np.array([0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649, 0.1576, 0.9706, 0.9572, 0.4854])
    assert np.all(
        np.isclose(f.objective(x), np.array([0.686973245563538, 0.106042165687341, 1.066946942373372]) * 1e3)
    )
    assert np.isclose(f.constraint(x), 0.01943601)


def test_Eq1DTLZ4():
    f = Eq1DTLZ4()
    x = np.array([0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649, 0.1576, 0.9706, 0.9572, 0.4854])
    assert np.all(np.isclose(f.objective(x), np.array([2.024647240000000, 0, 0])))
    assert np.isclose(f.constraint(x), 0.01943601)
