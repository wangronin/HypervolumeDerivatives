import numpy as np
from hvh import HypervolumeHessian, get_non_dominated

np.random.seed(42)

pareto_front = get_non_dominated(np.random.rand(5, 3))
ref = np.array([0, 0, 0])

hvh = HypervolumeHessian(pareto_front, ref)
H = hvh.compute()
print(H)
assert np.allclose(H, H.T)
