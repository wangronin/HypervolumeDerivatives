import sys

sys.path.insert(0, "./")
import random

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from hvd.reference_set import ReferenceSet


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
    M = ref.medoids
    dist_min = cdist(M, Y).sum()
    for i in range(100):
        random.shuffle(M)
        dist_rand = cdist(M, Y).sum()
        assert dist_min <= dist_rand


# test_match_and_shift()
# test_initialization_reference()
