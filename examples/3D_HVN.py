import matplotlib.pyplot as plt
import numpy as np
from hvd.algorithm import HVN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3

np.random.seed(42)

f = Eq1DTLZ3()
dim = 3
ref = np.array([100, 100, 100])
max_iters = 30
mu = 20

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
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(X[:, 0], X[:, 1], X[:, 2], "k.")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "r.")

ax.set_xlabel("f1")
ax.set_ylabel("f2")
ax.set_zlabel("f3")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.show()
breakpoint()
