# import numpy as np

import autograd.numpy as np
from autograd import grad, hessian, jacobian


def _cumprod(x):
    # collect products
    cumprods = []
    for i in range(x.size):
        # get next number / column / row
        current_num = x[i]

        # deal with first case
        if i == 0:
            cumprods.append(current_num)
        else:
            # get previous number
            prev_num = cumprods[i - 1]

            # compute next number / column / row
            next_num = prev_num * current_num
            cumprods.append(next_num)
    return np.array(cumprods)


class MOOAnalytical:
    def __init__(self):
        self.objective_jacobian = jacobian(self.objective)
        self.objective_hessian = hessian(self.objective)
        self.constraint_jacobian = jacobian(self.constraint)
        self.constraint_hessian = hessian(self.constraint)


class Eq1DTLZ1(MOOAnalytical):
    def __init__(self):
        self.n_objectives = 3
        self.n_decision_vars = self.n_objectives + 4
        self.lower_bounds = np.zeros(self.n_decision_vars)
        self.upper_bounds = np.ones(self.n_decision_vars)
        super().__init__()

    def objective(self, x: np.ndarray) -> np.ndarray:
        D = len(x)
        M = self.n_objectives
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return 0.5 * (1 + g) * _cumprod(np.r_[1, x[0 : M - 1]])[::-1] * np.r_[1, 1 - x[0 : M - 1][::-1]]

    def constraint(self, x: np.ndarray) -> float:
        M = self.n_objectives
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        return np.abs(np.sum(xx**2) - r**2) - 1e-4

    # TODO: this is not needed for now
    # def sample_on_PF(self, N: int):
    #     P = UniformPoint(N,obj.Global.M)/2;


if __name__ == "__main__":
    f = Eq1DTLZ1()
    x = np.random.rand(20)
    print(x)
    print(f.objective(x))
    print(f.constraint(x))
    print(f.objective_jacobian(x))
    print(f.objective_hessian(x))
    print(f.constraint_jacobian(x).shape)
    print(f.constraint_hessian(x).shape)
