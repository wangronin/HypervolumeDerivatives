"""Benchmark multiobjective gradient descent (MGDA) on stored MOEA populations."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd import problems
from hvd.gradient_based import MGDA
from scripts.utils import (
    PROBLEM_NAMES,
    compute_performance,
    get_dimensions_from_data,
    get_pareto_front,
    get_reference_point,
    get_run_instances,
    read_reference_set_data,
    set_random_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=PROBLEM_NAMES)
    parser.add_argument(
        "--algorithm", choices=["NSGA-II", "NSGA-III", "MOEAD", "SMS-EMOA"], default="NSGA-III"
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--generation", type=int, default=600)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--ref-point", nargs="+", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def execute(
    run: int,
    args: argparse.Namespace,
    problem: problems.CMOP,
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
) -> dict[str, float | int]:
    set_random_seed(66 + run * 1009)
    _, x0, _, _, _ = read_reference_set_data(
        args.data_path, args.problem, args.algorithm, run, args.generation, matching=True
    )
    optimizer = MGDA(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        func=problem.objective,
        jac=problem.objective_jacobian,
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=args.max_iters,
        step_size=args.step_size,
        verbose=False,
    )
    y = optimizer.run()[1]
    performance = compute_performance(y, pareto_front, ref_point)
    return {"run": run, **performance, "Jac_calls": optimizer.state.n_jac_evals}


def main() -> None:
    args = build_parser().parse_args()
    instances = get_run_instances(args.problem, args.algorithm, args.generation, args.data_path)
    if not instances:
        raise FileNotFoundError("No stored runs found for this problem, algorithm and generation")
    n_var, n_obj = get_dimensions_from_data(args, instances[0])
    problem = getattr(problems, args.problem)(n_var=n_var, n_obj=n_obj, boundary_constraints=False)
    if problem.n_eq_constr or problem.n_ieq_constr:
        raise ValueError(
            "This gradient method supports box bounds, not intrinsic equality/inequality constraints"
        )
    pareto_front = get_pareto_front(problem)
    ref_point = get_reference_point(args, problem)
    params = (args, problem, pareto_front, ref_point)
    if args.n_jobs == 1:
        records = [execute(run, *params) for run in instances]
    else:
        records = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run, *params) for run in instances)
    output = args.output or args.results_dir / f"{args.problem}-MGDA-{args.algorithm}-{args.generation}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=["run", "HV", "IGD", "GD", "Jac_calls"]).to_csv(output, index=False)


if __name__ == "__main__":
    main()
