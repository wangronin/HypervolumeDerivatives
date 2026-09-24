import numpy as np
import pytest

from hvd.hypervolume import HV


def test_hypervolume_metric_interface() -> None:
    metric = HV([3.0, 3.0])
    # The last two points are dominated and outside the reference box.
    points = np.array([[1.0, 2.0], [2.0, 1.0], [2.5, 2.5], [4.0, 4.0]])

    assert metric.compute(Y=points) == pytest.approx(3.0)
    assert metric.compute(Y=np.empty((0, 2))) == 0.0
    assert metric.compute(Y=points) == pytest.approx(3.0)
