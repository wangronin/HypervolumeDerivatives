import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import hessian, jacobian
from hvd.algorithm import HVN

np.random.seed(42)

dim = 2
ref = np.array([300, 300, 1])
max_iters = 60
plt.ioff()


def MOP7(x):
    x = np.array(x)
    term = x[0] ** 2 + x[1] ** 2
    f1 = 0.5 * term + np.sin(term)

    term1 = ((3 * x[0] - 2 * x[1] + 4) ** 2) / 8
    term2 = ((x[0] - x[1] + 1) ** 2) / 27
    f2 = term1 + term2 + 15

    term = x[0] ** 2 + x[1] ** 2
    f3 = (1 / (term + 1)) - 1.1 * np.exp(-term)
    return np.array([f1, f2, f3])


opt = HVN(
    dim=dim,
    n_objective=3,
    ref=ref,
    func=MOP7,
    jac=jacobian(MOP7),
    hessian=hessian(MOP7),
    mu=15,
    lower_bounds=-0.5,
    upper_bounds=0,
    minimization=True,
    # x0=np.array([np.r_[np.random.rand(2), [0.45] * 5], np.r_[np.random.rand(2), [0.45] * 5]]),
    max_iters=max_iters,
    verbose=True,
)

X, Y, stop = opt.run()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.semilogy(range(1, len(opt.hist_HV) + 1), opt.hist_HV, "r-")
ax1.plot(opt.X[:, 0], opt.X[:, 1], "b.")
plt.show()


breakpoint()
