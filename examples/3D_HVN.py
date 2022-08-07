import matplotlib.pyplot as plt
import numpy as np
from hvd.algorithm import HVN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2

np.random.seed(666)

f = Eq1DTLZ2()
dim = 3
ref = np.array([50, 50, 50])
max_iters = 50
mu = 5

opt = HVN(
    dim=dim,
    n_objective=3,
    ref=ref,
    func=f.objective,
    jac=f.objective_jacobian,
    hessian=f.objective_hessian,
    # h=f.constraint,
    # h_jac=f.constraint_jacobian,
    # h_hessian=f.constraint_hessian,
    mu=mu,
    lower_bounds=0,
    upper_bounds=1,
    minimization=True,
    x0=np.c_[np.random.rand(mu, 2), np.tile(0.5, (mu, 1))],
    # x0=np.array([np.r_[[-0.5, 0.5], [0.48] * 1], np.r_[[2, -1.2], [0.48] * 1]]),
    # x0=np.array([np.r_[[-0.5, 0.5], [0.48] * 1], np.r_[[2, -1.2], [0.48] * 1]]),
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

plt.semilogy(range(1, max_iters + 1), opt.hist_HV, "r-")
plt.show()

breakpoint()
