"""Benchmark MMD Newton with population-based reference bootstrapping; save one CSV."""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd import problems
from hvd.bootstrap import bootstrap_reference_set
from scripts.benchmark_MMD import build_parser as mmd_parser
from scripts.benchmark_MMD import get_config, get_optimizer
from scripts.utils import (
    MMDConfig,
    compute_performance,
    get_dimensions_from_data,
    get_pareto_front,
    get_reference_point,
    get_run_instances,
)


def build_parser() -> ArgumentParser:
    parser = mmd_parser()
    parser.description = __doc__
    parser.set_defaults(max_iters=9)
    parser.add_argument("--bootstrap-interval", type=int, default=3)
    parser.add_argument("--last-n", type=int, default=3, help="Recent populations used for final resampling")
    return parser


def execute(
    run: int,
    args: Namespace,
    configuration: MMDConfig,
    problem: problems.CMOP,
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
) -> dict[str, float | int]:
    optimizer = get_optimizer(run, args, configuration, problem)
    _, y, _, _ = bootstrap_reference_set(
        optimizer,
        problem,
        interval=args.bootstrap_interval,
        last_n=args.last_n,
        with_rsg=False,
        plot=False,
    )
    performance = compute_performance(y, pareto_front, ref_point)
    return {"run": run, **performance, "Jac_calls": optimizer.state.n_jac_evals}


def main() -> None:
    args = build_parser().parse_args()
    configuration, args = get_config(args)
    if min(args.max_iters, args.bootstrap_interval, args.last_n, args.n_jobs) < 1:
        raise ValueError("Iteration counts, bootstrap interval, last-n and n-jobs must be positive")
    instances = get_run_instances(args.problem, args.algorithm, args.generation, args.data_path)
    if not instances:
        raise FileNotFoundError("No stored runs found for this problem, algorithm and generation")
    n_var, n_obj = get_dimensions_from_data(args, instances[0])
    problem = getattr(problems, args.problem)(n_var=n_var, n_obj=n_obj, boundary_constraints=True)
    pareto_front = get_pareto_front(problem)
    ref_point = get_reference_point(args, problem)
    params = (args, configuration, problem, pareto_front, ref_point)
    if args.n_jobs == 1:
        records = [execute(run, *params) for run in instances]
    else:
        records = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run, *params) for run in instances)
    method = "MMDNMatchingBootstrap" if args.matching else "MMDNBootstrap"
    output = (
        args.output or args.results_dir / f"{args.problem}-{method}-{args.algorithm}-{args.generation}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=["run", "HV", "IGD", "GD", "Jac_calls"]).to_csv(output, index=False)


if __name__ == "__main__":
    main()
