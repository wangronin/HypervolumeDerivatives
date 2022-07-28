import numpy as np
from hvd.algorithm import HVN

np.random.seed(55)

dim = 3
ref = np.array([30, 30, 30])


def MOP1(x):
    x = np.array(x)
    return np.array(
        [
            np.sum((x - np.array([1, 1, 0])) ** 2),
            np.sum((x - np.array([-1, -1, 0])) ** 2),
            np.sum((x - np.array([1, -1, 0])) ** 2),
        ]
    )


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array(
        [2 * (x - np.array([1, 1, 0])), 2 * np.array([-1, -1, 0]), 2 * (x - np.array([1, -1, 0]))]
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
    mu=2,
    lower_bounds=-1,
    upper_bounds=1,
    minimization=True,
    max_iters=30,
)
X, Y, stop = opt.run()
breakpoint()
