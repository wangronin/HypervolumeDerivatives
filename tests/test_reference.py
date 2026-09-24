import sys

sys.path.insert(0, "./")
import random

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from hvd.reference_set import ReferenceSet


def test_shift_selects_matched_medoids_without_moving_original_points():
    original = np.array([[0.0, 1.0], [0.3, 0.7], [0.7, 0.3], [1.0, 0.0]])
    reference = ReferenceSet(original.copy(), eta={0: -np.ones(2)})
    reference.match(original[[0, 3]])
    expected = reference.reference_set.copy()
    expected[1] -= 0.1

    reference.shift(0.1, np.array([1]))

    np.testing.assert_allclose(reference.reference_set, expected)
    np.testing.assert_array_equal(reference._ref[0], original)


@pytest.mark.parametrize("n_component", [1, 3, 5])
def test_initialization_reference(n_component: int):
    x = np.linspace(0, 1, 20)
    ref_ = {i: np.c_[x, 1 - x] + i for i in range(n_component)}
    ref = ReferenceSet(ref=ref_, eta=None, Y_idx=None)
    assert len(ref._ref) == n_component
    assert ref.n_obj == 2
    assert ref.n_components == n_component
    assert isinstance(ref.eta, dict)
    for i in range(n_component):
        assert np.all(np.isclose(ref.eta[i], np.array([-0.70710678, -0.70710678])))
        assert np.all(np.isclose(ref._ref[i], ref_[i]))
    assert not hasattr(ref, "_matched_medoids")


@pytest.mark.parametrize("n_component", [1, 3, 5])
def test_match(n_component: int):
    x = np.linspace(0, 1, 20)
    ref_ = {i: np.c_[x + 1.5 * i, 1 - x - 1.5 * i] + i for i in range(n_component)}
    Y = np.concatenate([np.c_[x + 1.5 * i + 0.2, 1.2 - x - 1.5 * i] + i for i in range(n_component)], axis=0)
    Y_idx = [np.arange(i * 20, (i + 1) * 20) for i in range(n_component)]

    if 11 < 2:  # for debugging
        import matplotlib.pyplot as plt

        for i in range(n_component):
            plt.plot(ref_[i][:, 0], ref_[i][:, 1], "k.")
            plt.plot(Y[i][:, 0], Y[i][:, 1], "r.")

        plt.show()

    ref = ReferenceSet(ref=ref_, eta=None, Y_idx=Y_idx)
    for i in range(n_component):
        assert np.all(np.isclose(ref.eta[i], np.array([-0.70710678, -0.70710678])))
        assert np.all(np.isclose(ref._ref[i], ref_[i]))

    ref.match(Y)
    # testing if the matching function yields the smallest total distance
    M = ref.reference_set
    dist_min = cdist(M, Y).sum()
    for i in range(100):
        random.shuffle(M)
        dist_rand = cdist(M, Y).sum()
        assert dist_min <= dist_rand


# test_match_and_shift()
# test_initialization_reference()
