import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter


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


class CDTLZ1(ElementwiseProblem):
    def __init__(self, nobj=3, nvar=11):
        super().__init__(n_var=nvar, n_obj=nobj, n_eq_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        D = len(x)
        M = self.n_obj
        g = 100 * (D - M + 1 + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[M - 1 :] - 0.5))))
        out["F"] = 0.5 * (1 + g) * _cumprod(np.r_[1, x[0 : M - 1]])[::-1] * np.r_[1, 1 - x[0 : M - 1][::-1]]

        M = self.n_obj
        r = 0.4
        xx = x[0 : M - 1] - 0.5
        out["H"] = np.asarray(np.abs(np.sum(xx**2) - r**2) - 1e-4)


class Problem2(ElementwiseProblem):
    def __init__(self):
        nVar = 3
        xLowr = [-4.0] * nVar
        xUppr = [4.0] * nVar
        super().__init__(n_var=nVar, n_obj=3, n_eq_constr=1, xl=xLowr, xu=xUppr)

    def _evaluate(self, x, out, *args, **kwargs):
        c1 = [-1, -1, -1]
        c2 = [-1, 0, 0]
        c3 = [-2, -2, 4]
        out["F"] = [sum((x - c1) ** 2), sum((x - c2) ** 2), sum((x - c3) ** 2)]
        out["H"] = [min((x[0] - 1e-8), 0)]


for x in range(15):
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
    termination = get_termination("n_gen", 10000)

    # create the algorithm object
    algorithm = AdaptiveEpsilonConstraintHandling(NSGA3(pop_size=200, ref_dirs=ref_dirs), perc_eps_until=0.5)
    problem = CDTLZ1()
    # execute the optimization
    res = minimize(problem, algorithm, termination, seed=x, verbose=True)
    print(res.X.shape)

    Scatter().add(res.F).show()

    breakpoint()

    for i in range(len(res.X)):
        print(res.X[i])


# vars(res.pop[1])
# print(res.pop.shape)

# for i in range(len(res.pop)):
#     print(res.pop[i]._X)
#     print(res.pop[i]._F)
#     print(res.pop[i]._H)
#     list1 = np.concatenate((res.pop[i]._X, res.pop[i]._F))
#     list2 = np.concatenate((list1, res.pop[i]._H))
#     #file.write(list2)
#     list2 = np.array2string(list2)
#     file.write(list2+"\n")
# file.close
