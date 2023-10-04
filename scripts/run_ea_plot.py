import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from hvd.problems import (
    CF9,
    Eq1DTLZ1,
    Eq1DTLZ2,
    Eq1DTLZ3,
    Eq1DTLZ4,
    Eq1IDTLZ1,
    Eq1IDTLZ2,
    Eq1IDTLZ3,
    Eq1IDTLZ4,
    MOOAnalytical,
)


class ProblemWrapper(ElementwiseProblem):
    """Wrapper for Eq-DTLZ problems"""

    def __init__(self, problem: MOOAnalytical):
        self._problem = problem
        super().__init__(
            n_var=problem.n_decision_vars, n_obj=problem.n_objectives, n_eq_constr=1, xl=0.0, xu=1.0
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        out["F"] = self._problem.objective(x)
        out["H"] = self._problem.constraint(x)


np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000)
plt.style.use("ggplot")


ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
termination = get_termination("n_gen", 500)
algorithm = AdaptiveEpsilonConstraintHandling(NSGA3(pop_size=400, ref_dirs=ref_dirs), perc_eps_until=0.8)

problem = Eq1DTLZ1()
# pareto_set = problem.get_pareto_set(500)
# pareto_front = problem.get_pareto_front(500)
res = minimize(ProblemWrapper(problem), algorithm, termination, seed=42, verbose=True)

X, Y = res.X, res.F

fig = plt.figure(figsize=plt.figaspect(1 / 2))
plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(50, -20)

# ax.plot(pareto_set[:, 0], pareto_set[:, 1], pareto_set[:, 2], "gray", alpha=0.4)

# plot the initial decision points
# ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], "g.", ms=8)
# plot the final decision points
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g*", ms=6)
ax.set_title("decision space")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_xlim([0, 1.1])
ax.set_ylim([0, 1.1])
ax.set_zlim([0.4, 0.6])

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(45, 45)

# ax.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "gray", alpha=0.4)
# plot the final Pareton approximation set
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "g*", ms=8)

ax.set_title("objective space")
ax.set_xlabel(r"$f_1$")
ax.set_ylabel(r"$f_2$")
ax.set_zlabel(r"$f_3$")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()
# plt.savefig(f"3D-example1-{mu}.pdf", dpi=100)
