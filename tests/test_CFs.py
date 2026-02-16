import sys

sys.path.insert(0, "./")
import numpy as np
import pytest

from hvd.problems import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10


@pytest.mark.parametrize(
    "problem, expected, cstr",
    [
        (CF1, [0.52938713, 0.99615366], [0.33891173]),
        (CF2, [1.71077441, 1.54561383], [-3.18405368e-05]),
        (CF3, [13.53009123, 17.13769761], [-199.6512955]),
        (CF4, [7.92852758, 12.29133178], [0.02363948]),
        (CF5, [15.08562969, 17.41149153], [-0.14905891]),
        (CF6, [6.37884005, 8.20124485], [-0.93837952, -0.92107454]),
        (CF7, [10.85038778, 29.44993101], [1.64866296, 0.98684035]),
        (CF8, [15.2176, 16.7928, 13.7883], [7.6939]),
        (CF9, [4.22726212, 3.11873893, 3.56350472], [6.18845852]),
        (CF10, [6.73744149, 3.40704349, 8.33025044], [1.79565082]),
    ],
)
def test_CFs(problem, expected, cstr):
    if problem.__name__ == "CF1":
        np.random.seed(42)

    p = problem()
    interval = p.xu - p.xl
    x = np.random.rand(p.n_var) * interval + p.xl
    F = p.objective(x)
    C = p.ieq_constraint(x)
    p.objective_jacobian(x)
    p.ieq_constraint(x)
    assert np.all(np.isclose(F, expected))
    assert np.all(np.isclose(C, cstr))
