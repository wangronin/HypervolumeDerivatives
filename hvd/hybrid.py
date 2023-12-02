import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from hvd.problems import CONV4

from .interpolate import ReferenceSetInterpolation
from .newton import DpN
from .problems.base import MOOAnalytical


class ProblemWrapper(ElementwiseProblem):
    """Wrapper for Eq-DTLZ problems"""

    def __init__(self, problem: MOOAnalytical):
        self._problem = problem
        super().__init__(
            n_var=problem.n_decision_vars, n_obj=problem.n_objectives, n_eq_constr=1, xl=0.0, xu=1.0
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        out["F"] = self._problem.objective(x)
        if hasattr(self._problem, "constraint"):
            out["H"] = self._problem.constraint(x)


pop_to_numpy = lambda pop: np.array([np.r_[ind.X, ind.F] for ind in pop])


def minimize(
    problem, algorithm, termination=None, copy_algorithm=True, copy_termination=True, run_id=None, **kwargs
):
    data = []
    columns = [f"x{i}" for i in range(1, problem.n_var + 1)] + [f"f{i}" for i in range(1, problem.n_obj + 1)]
    # create a copy of the algorithm object to ensure no side-effects
    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        if termination is not None:
            if copy_termination:
                termination = copy.deepcopy(termination)

            kwargs["termination"] = termination

        algorithm.setup(problem, **kwargs)

    # actually execute the algorithm
    k = 1
    while algorithm.has_next():
        algorithm.next()
        pop = copy.deepcopy(algorithm.pop)
        if algorithm.n_gen == k + 1:
            df = pd.DataFrame(pop_to_numpy(pop), columns=columns)
            df.insert(0, "iteration", k)
            data.append(df)
            k += 1
    res = algorithm.result()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm
    data = pd.concat(data, axis=0)
    if run_id is not None:
        data.insert(0, "run", run_id)
    return data


class NSGA_DpN:
    def __init__(
        self,
        problem: MOOAnalytical,
        n_iters_ea: int = 1000,
        n_iters_newton: int = 10,
        random_seed: int = 42,
    ):
        self.n_objectives = problem.n_objectives
        self.n_decision_vars = problem.n_decision_vars
        self.problem = problem
        self.random_seed = random_seed
        self.n_iters_ea = n_iters_ea
        self.n_iters_newton = n_iters_newton
        self._ref_gen = ReferenceSetInterpolation(n_objective=self.n_objectives)
        self._init_ea(self.n_iters_ea)

    def _init_ea(self, n_iters_ea: int):
        if self.n_objectives == 2:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        elif self.n_objectives == 3:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
        elif self.n_objectives == 4:
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=11)
        self._ea_termination = get_termination("n_gen", n_iters_ea)
        ea = NSGA3(pop_size=400, ref_dirs=ref_dirs)
        if hasattr(self.problem, "constraint"):
            self._ea = AdaptiveEpsilonConstraintHandling(ea, perc_eps_until=0.5)
        else:
            self._ea = ea

    def _init_newton(
        self, problem: MOOAnalytical, X0: np.ndarray, reference_set: np.ndarray, n_iters_newton: int
    ):
        self._newton = DpN(
            dim=self.n_decision_vars,
            n_obj=self.n_objectives,
            ref=reference_set,
            x0=X0,
            func=problem.objective,
            jac=problem.objective_jacobian,
            hessian=problem.objective_hessian,
            h=problem.constraint if hasattr(problem, "constraint") else None,
            h_jac=problem.constraint_jacobian if hasattr(problem, "constraint_jacobian") else None,
            xl=problem.lower_bounds,
            xu=problem.upper_bounds,
            max_iters=n_iters_newton,
            verbose=True,
        )

    def run(self) -> dict:
        data = minimize(
            ProblemWrapper(self.problem), self._ea, self._ea_termination, seed=self.random_seed, verbose=True
        )
        data_last_iter = data[data.iteration == self.n_iters_ea]
        X_ea = data_last_iter.loc[:, "x1":f"x{self.n_decision_vars}"].values
        Y_ea = data_last_iter.loc[:, "f1":"f3"].values
        CPU_time_ea = self.problem.CPU_time / 1e9

        pareto_front = self.problem.get_pareto_front(1000)
        fig = plt.figure(figsize=plt.figaspect(1 / 2))
        plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
        ax = fig.add_subplot(1, 2, 1, projection="3d")

        ax.set_box_aspect((1, 1, 1))
        ax.view_init(45, 45)
        ax.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "k.", alpha=0.4)
        ax.plot(Y_ea[:, 0], Y_ea[:, 1], Y_ea[:, 2], "g.", alpha=0.4)
        ax.set_title("NSGA-III")
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")
        ax.set_zlabel(r"$f_3$")

        # generation of reference set for DpN
        F = (
            # data[data.iteration.isin([self.n_iters_ea])]
            data[(data.iteration <= self.n_iters_ea) & (data.iteration >= self.n_iters_ea - 3)]
            .loc[:, "f1":"f3"]
            .values
        )
        reference_set0 = self._ref_gen.interpolate(F, N=400, return_clusters=False)
        delta = 1
        # shift the reference set
        reference_set = reference_set0 - delta

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(45, 45)
        ax.plot(reference_set[:, 0], reference_set[:, 1], reference_set[:, 2], "g.", alpha=0.4)
        ax.set_title("interpolation")
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")
        ax.set_zlabel(r"$f_3$")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        plt.show()
        breakpoint()

        # clear the CPU_time counter since we only need to measure the time taken by HVN
        self.problem.CPU_time = 0
        self._init_newton(self.problem, X_ea, reference_set, self.n_iters_newton)

        while not self._newton.terminate():
            self._newton.newton_iteration()
            self._newton.log()
            # exponential decay of the shift
            delta *= 0.5
            reference_set -= delta
            self._newton.reference_set = reference_set

        X = self._newton._get_primal_dual(self._newton.X)[0]
        Y = self._newton.Y
        # X, Y, _ = self._newton.run()
        CPU_time_newton = self.problem.CPU_time / 1e9
        out = dict(
            X_ea=X_ea,
            X=X,
            Y_ea=Y_ea,
            Y=Y,
            CPU_time_ea=CPU_time_ea,
            CPU_time_newton=CPU_time_newton,
            reference_set=reference_set,
            reference_set0=reference_set0,
        )
        return out
