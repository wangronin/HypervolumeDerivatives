import numpy as np
from hvd.algorithm import HVN

np.random.seed(55)

dim = 2
ref = np.array([20, 20])


def MOP1(x):
    x = np.array(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(dim), 2 * np.eye(dim)])


def h(x):
    x = np.array(x)
    return np.sum(x**2) - 1


def h_Jacobian(x):
    x = np.array(x)
    return 2 * x


def h_Hessian(x):
    return 2 * np.eye(dim)


opt = HVN(
    dim=dim,
    n_objective=2,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    h=h,
    h_jac=h_Jacobian,
    h_hessian=h_Hessian,
    mu=5,
    # x0=np.array([[1.5, -0.5], [1.25, -0.75], [1, -1], [0.75, -1.25], [0.5, -1.5]]),
    lower_bounds=-2,
    upper_bounds=2,
    minimization=True,
    max_iters=50,
    verbose=True,
)
X, Y, stop = opt.run()
breakpoint()
