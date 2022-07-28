import numpy as np


class Eq1DTLZ1:
    def __init__(self):
        self.n_objectives = 3
        self.n_decision_vars = self.n_objectives + 4
        self.lower_bounds = np.zeros(self.n_decision_vars)
        self.upper_bounds = np.ones(self.n_decision_vars)

    def objective(self, x: np.ndarray) -> np.ndarray:
        D = len(x)
        M = self.n_objectives
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        return 0.5 * (1 + g) * np.cumprod(np.r_[1, x[0 : M - 1]])[::-1] * np.r_[1, 1 - x[0 : M - 1][::-1]]

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
