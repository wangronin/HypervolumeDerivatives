import matplotlib.pyplot as plt
import numpy as np
from hvd.algorithm import HVN

np.random.seed(42)


dim = 3
ref = np.array([10, 10, 10])
max_iters = 15


def MOP1(x):
    x = np.array(x)
    return np.array(
        [
            np.sum((x - np.array([1, 1, 0])) ** 2),
            np.sum((x - np.array([1, -1, 0])) ** 2),
            np.sum((x - np.array([-1, 0, 0])) ** 2),
        ]
    )


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array(
        [2 * (x - np.array([1, 1, 0])), 2 * (x - np.array([1, -1, 0])), 2 * (x - np.array([-1, 0, 0]))]
    )


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(dim), 2 * np.eye(dim), 2 * np.eye(dim)])


opt = HVN(
    dim=dim,
    n_objective=3,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    mu=40,
    lower_bounds=-1,
    upper_bounds=1,
    minimization=True,
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "r-")
ax1.plot(opt.X[1:, 0], opt.X[1:, 1], "b.")
plt.show()

breakpoint()
