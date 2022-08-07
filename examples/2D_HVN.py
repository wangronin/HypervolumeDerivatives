import numpy as np
from hvd.algorithm import HVN

np.random.seed(42)

dim = 2
ref = np.array([20, 20])
mu = 10


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
    return np.abs(np.sum(x**2) - 1)


def h_Jacobian(x):
    x = np.array(x)
    sign = 1 if np.sum(x**2) - 1 >= 0 else -1
    return 2 * x * sign


def h_Hessian(x):
    x = np.array(x)
    sign = 1 if np.sum(x**2) - 1 >= 0 else -1
    return 2 * np.eye(dim) * sign


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
    mu=mu,
    x0=np.random.rand(mu, 2) * 2 - 1,
    lower_bounds=-2,
    upper_bounds=2,
    minimization=True,
    max_iters=25,
    verbose=True,
)
X, Y, stop = opt.run()
breakpoint()
