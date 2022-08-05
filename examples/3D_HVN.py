import matplotlib.pyplot as plt
import numpy as np
from hvd.algorithm import HVN
from hvd.problems import Eq1DTLZ1

np.random.seed(42)

f = Eq1DTLZ1()
dim = 7
ref = np.array([500, 500, 500])
max_iters = 100

opt = HVN(
    dim=dim,
    n_objective=3,
    ref=ref,
    func=f.objective,
    jac=f.objective_jacobian,
    hessian=f.objective_hessian,
    mu=10,
    lower_bounds=0,
    upper_bounds=1,
    minimization=True,
    x0=np.c_[np.random.rand(10, 2), np.tile(0.49, (10, 5))],
    # x0=np.array([np.r_[np.random.rand(2), [0.48] * 5], np.r_[np.random.rand(2), [0.48] * 5]]),
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

plt.semilogy(range(1, max_iters + 1), opt.hist_HV, "r-")
plt.show()

breakpoint()
