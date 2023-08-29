import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from .interpolate import ReferenceSetInterpolation
from .newton import DpN
from .problems import MOOAnalytical


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
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=11)
        elif self.n_objectives == 4:
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=11)
        self._ea_termination = get_termination("n_gen", n_iters_ea)
        # ea = NSGA2(pop_size=200)
        ea = NSGA3(pop_size=200, ref_dirs=ref_dirs)
        if hasattr(self.problem, "constraint"):
            self._ea = AdaptiveEpsilonConstraintHandling(ea, perc_eps_until=0.5)
        else:
            self._ea = ea

    def _init_newton(
        self, problem: MOOAnalytical, X0: np.ndarray, reference_set: np.ndarray, n_iters_newton: int
    ):
        self._newton = DpN(
            dim=self.n_decision_vars,
            n_objective=self.n_objectives,
            ref=reference_set,
            x0=X0,
            func=problem.objective,
            jac=problem.objective_jacobian,
            hessian=problem.objective_hessian,
            h=problem.constraint if hasattr(problem, "constraint") else None,
            h_jac=problem.constraint_jacobian if hasattr(problem, "constraint_jacobian") else None,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
            max_iters=n_iters_newton,
            verbose=True,
        )

    def run(self) -> dict:
        res = minimize(
            ProblemWrapper(self.problem), self._ea, self._ea_termination, seed=self.random_seed, verbose=True
        )
        CPU_time_ea = self.problem.CPU_time / 1e9
        X_ea = np.array([p._X for p in res.pop])  # final approximation set of NSGA-II
        Y_ea = np.array([p._F for p in res.pop])  # final approximation set of NSGA-II
        # generation of reference set for DpN
        reference_set0 = self._ref_gen.interpolate(Y_ea, N=400, return_clusters=False)
        delta = 1
        # shift the reference set
        reference_set = reference_set0 - delta
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
