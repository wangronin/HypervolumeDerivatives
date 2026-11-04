"""Run fresh MOEAs with a fixed-generation or per-run Newton-derived budget."""

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.problem import Problem as PymooProblem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd import problems
from hvd.moea import SMSEMOA, ConstraintAwareMOEAD
from scripts.utils import (
    PROBLEM_NAMES,
    compute_performance,
    get_pareto_front,
    get_reference_point,
    set_random_seed,
)
ALGORITHMS = ["NSGA-II", "NSGA-III", "MOEAD", "SMS-EMOA"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=PROBLEM_NAMES)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="NSGA-III")
    parser.add_argument("--budget", choices=["fixed", "newton"], default="newton")
    parser.add_argument(
        "--generation", type=int, default=600, help="Fixed generations, or source Newton CSV generation"
    )
    parser.add_argument("--newton", default="MMDNMatching", help="Method tag in the source performance CSV")
    parser.add_argument(
        "--jacobian-cost", type=float, default=3.215, help="Function evaluations per Jacobian call"
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-var", type=int)
    parser.add_argument("--n-obj", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--ref-point", nargs="+", type=float)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--save-populations", action="store_true")
    parser.add_argument("--last-generations", type=int, default=7)
    parser.add_argument("--population-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def get_algorithm(n_obj: int, name: str, pop_size: int, constrained: bool) -> Algorithm:
    """Use the same algorithm setup for both budgeting modes."""
    if name == "NSGA-II":
        algorithm = NSGA2(pop_size=pop_size)
    elif name == "SMS-EMOA":
        algorithm = SMSEMOA(pop_size=pop_size)
    else:
        if pop_size < n_obj:
            raise ValueError("Population size must cover at least one direction per objective")
        partitions = 1
        while math.comb(partitions + n_obj - 1, n_obj - 1) < pop_size:
            partitions += 1
        directions = get_reference_directions("das-dennis", n_obj, n_partitions=partitions)
        if len(directions) > pop_size:
            corners = np.flatnonzero((directions == 1).any(axis=1))
            interior = np.setdiff1d(np.arange(len(directions)), corners)
            chosen = np.random.choice(interior, pop_size - len(corners), replace=False)
            directions = directions[np.r_[corners, chosen]]
        if name == "NSGA-III":
            algorithm = NSGA3(pop_size=pop_size, ref_dirs=directions)
        elif name == "MOEAD":
            algorithm = ConstraintAwareMOEAD(
                directions, n_neighbors=min(15, pop_size), prob_neighbor_mating=0.7, perc_eps_until=0.8
            )
        else:
            raise ValueError(f"Unknown algorithm: {name}")
    if constrained and name != "MOEAD":
        algorithm = AdaptiveEpsilonConstraintHandling(algorithm, perc_eps_until=0.8)
    return algorithm


def get_budgets(args: argparse.Namespace, runs: list[int]) -> dict[int, int]:
    """Match each run ID to ceil(jacobian-cost * that run's Jac_calls)."""
    if args.budget == "fixed":
        return {run: args.generation for run in runs}
    source = args.results_dir / f"{args.problem}-{args.newton}-{args.algorithm}-{args.generation}.csv"
    calls = pd.read_csv(source).set_index("run", verify_integrity=True)["Jac_calls"]
    missing = set(runs) - set(calls.index)
    if missing:
        raise ValueError(f"Missing Jacobian counts for runs {sorted(missing)}: {source}")
    values = calls.loc[runs].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values != np.floor(values)):
        raise ValueError("Jac_calls must contain finite positive integer counts")
    return {run: math.ceil(args.jacobian_cost * calls.loc[run]) for run in runs}


def population_frame(
    population: Population, problem: PymooProblem, run: int, generation: int
) -> pd.DataFrame:
    """Keep decisions, objectives and constraints in separate named columns."""
    columns = (
        [f"x{i+1}" for i in range(problem.n_var)]
        + [f"f{i+1}" for i in range(problem.n_obj)]
        + [f"h{i+1}" for i in range(problem.n_eq_constr)]
        + [f"g{i+1}" for i in range(problem.n_ieq_constr)]
    )
    values = np.column_stack([population.get(key) for key in ("X", "F", "H", "G")])
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "iteration", generation)
    frame.insert(0, "run", run)
    return frame


def execute(
    run: int,
    args: argparse.Namespace,
    problem: problems.MOP,
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
    budget: int,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    set_random_seed(run)
    native = problem.as_pymoo_problem()
    algorithm = get_algorithm(problem.n_obj, args.algorithm, args.pop_size, native.has_constraints())
    termination = get_termination("n_gen" if args.budget == "fixed" else "n_eval", budget)
    algorithm.setup(native, termination=termination, seed=run, verbose=False)
    history = deque(maxlen=args.last_generations)

    while algorithm.has_next():
        algorithm.next()
        if args.save_populations:
            history.append(population_frame(algorithm.pop, native, run, algorithm.n_gen - 1))

    y = algorithm.result().F
    performance = compute_performance(y, pareto_front, ref_point)
    record = {"run": run, **performance}
    return record, pd.concat(history, ignore_index=True) if history else None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dimensions = {
        key: value
        for key, value in {"n_var": args.n_var, "n_obj": args.n_obj}.items()
        if value is not None
    }
    problem = getattr(problems, args.problem)(**dimensions, boundary_constraints=False)
    args.pop_size = args.pop_size or {2: 100, 3: 300, 4: 600}.get(problem.n_obj, 100 * problem.n_obj)
    runs = list(range(1, args.runs + 1))
    budgets = get_budgets(args, runs)
    pareto_front = get_pareto_front(problem)
    ref_point = get_reference_point(args, problem)

    params = (args, problem, pareto_front, ref_point)
    if args.n_jobs == 1:
        data = [execute(run, *params, budgets[run]) for run in runs]
    else:
        data = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run, *params, budgets[run]) for run in runs)
    # save the performance results
    output = args.output or args.results_dir / f"{args.problem}-{args.algorithm}-{args.generation}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record for record, _ in data], columns=["run", "HV", "IGD", "GD"]).to_csv(
        output, index=False
    )
    # save the population data
    if args.save_populations:
        populations = pd.concat([frame for _, frame in data], ignore_index=True)
        args.population_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.population_dir / f"{args.problem.upper()}_{args.algorithm}"
        populations.to_csv(prefix.with_suffix(".csv"), index=False, float_format="%.17g")
        savemat(
            prefix.with_suffix(".mat"),
            {"data": populations.to_numpy(), "columns": populations.columns.to_numpy()},
        )


if __name__ == "__main__":
    main()
