import sys

import numpy as np

sys.path.insert(0, "./")
from hvd.problems import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10


def test_CFs():
    np.random.seed(42)
    p = CF1()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [0.52938713, 0.99615366]))
    assert np.all(np.isclose(C, [0.33891173]))

    p = CF2()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [1.71077441, 1.54561383]))
    assert np.all(np.isclose(C, [-3.18405368e-05]))

    p = CF3()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [13.53009123, 17.13769761]))
    assert np.all(np.isclose(C, [-199.6512955]))

    p = CF4()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [7.92852758, 12.29133178]))
    assert np.all(np.isclose(C, [0.02363948]))

    p = CF5()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [15.08562969, 17.41149153]))
    assert np.all(np.isclose(C, [-0.14905891]))

    p = CF6()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [6.37884005, 8.20124485]))
    assert np.all(np.isclose(C, [-0.93837952, -0.92107454]))

    p = CF7()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [10.85038778, 29.44993101]))
    assert np.all(np.isclose(C, [1.64866296, 0.98684035]))

    p = CF8()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [15.21763177, 16.79283565, 13.7882774]))
    assert np.all(np.isclose(C, [7.69385584]))

    p = CF9()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [4.22726212, 3.11873893, 3.56350472]))
    assert np.all(np.isclose(C, [6.18845852]))

    p = CF10()
    interval = p.upper_bounds - p.lower_bounds
    x = np.random.rand(p.n_decision_vars) * interval + p.lower_bounds
    F = p.objective(x)
    C = p.constraint(x)
    p.objective_jacobian(x)
    p.constraint_jacobian(x)
    assert np.all(np.isclose(F, [6.73744149, 3.40704349, 8.33025044]))
    assert np.all(np.isclose(C, [1.79565082]))
