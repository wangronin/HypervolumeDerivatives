"""Benchmark DpN on stored EMOA populations, including WFG2–9."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd import problems
from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import HV
from hvd.newton import DpN
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import (
    PROBLEM_NAMES,
    compute_performance,
    get_dimensions_from_data,
    get_pareto_front,
    get_reference_point,
    get_run_instances,
    plot,
    read_reference_set_data,
    set_random_seed,
)

BOUNDARY_CONSTRAINTS = True
DEFAULT_NEWTON_MAX_ITERS = 5
DEFAULT_MOEA_GENERATION = 600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DpN on stored EMOA populations.")
    parser.add_argument("problem", choices=PROBLEM_NAMES)
    parser.add_argument(
        "--algorithm", default="NSGA-III", choices=["NSGA-II", "NSGA-III", "MOEAD", "SMS-EMOA"]
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--generation", type=int, default=DEFAULT_MOEA_GENERATION)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_NEWTON_MAX_ITERS)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ref-point", nargs="+", type=float)
    parser.add_argument("--type", choices=["gd", "igd", "deltap"], default="igd")
    parser.add_argument("--regularization", action="store_true")
    parser.add_argument("--project-box-constraints", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-populations", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "plots")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def get_optimizer(
    args: argparse.Namespace,
    problem: problems.CMOP,
    reference_set: ReferenceSet,
    x0: np.ndarray,
    metrics: dict,
) -> DpN:
    """Build DpN from the loaded population and reference set."""
    return DpN(
        dim=problem.n_var,
        n_obj=problem.n_obj,
        ref=reference_set,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        h=problem.eq_constraint,
        h_jac=problem.eq_jacobian,
        h_hessian=problem.eq_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
        N=len(x0),
        x0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=args.max_iters,
        type=args.type,
        verbose=args.verbose,
        metrics=metrics,
        regularization=args.regularization,
        project_box_constraints=args.project_box_constraints,
    )


def execute(
    run: int,
    args: argparse.Namespace,
    problem: problems.CMOP,
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
) -> dict[str, float | int]:
    set_random_seed(66 + run * 1009)
    ref, x0, y0, y_indices, eta = read_reference_set_data(
        args.data_path, args.problem, args.algorithm, run, args.generation, matching=True
    )
    reference_set = ReferenceSet(ref=ref, eta=eta, Y_idx=y_indices)
    initial_reference = reference_set.reference_set.copy()
    metrics = dict(
        GD=GenerationalDistance(pareto_front),
        IGD=InvertedGenerationalDistance(pareto_front),
        HV=HV(ref_point),
    )
    print(f"run {run}: precomputed eta={eta is not None}")
    for key, val in metrics.items():
        print(f"initial {key}: {val.compute(Y=y0)}")

    optimizer = get_optimizer(args, problem, reference_set, x0, metrics)
    started = time.process_time_ns()
    result = optimizer.run()[1]
    cpu_time = (time.process_time_ns() - started) / 1e9
    Y = get_non_dominated(result)
    name = f"{args.problem}_DpN_{args.algorithm}_run{run}_{args.generation}"
    if not args.skip_plots:
        plot(y0, Y, initial_reference, pareto_front, args.plot_dir / f"{name}.pdf", optimizer)
    if not args.skip_populations:
        objective_columns = [f"f{i + 1}" for i in range(problem.n_obj)]
        pd.DataFrame(y0, columns=objective_columns).to_csv(args.results_dir / f"{name}_y0.csv", index=False)
        pd.DataFrame(Y, columns=objective_columns).to_csv(args.results_dir / f"{name}_y.csv", index=False)
    out = compute_performance(Y, pareto_front, ref_point)
    out["run"] = run
    out["Jac_calls"] = optimizer.state.n_jac_evals
    out["CPU_time"] = cpu_time
    return out


def main() -> None:
    args = build_parser().parse_args()
    instances = get_run_instances(args.problem, args.algorithm, args.generation, args.data_path)
    n_var, n_obj = get_dimensions_from_data(args, instances[0])
    problem = getattr(problems, args.problem)(
        n_var=n_var, n_obj=n_obj, boundary_constraints=BOUNDARY_CONSTRAINTS
    )
    pareto_front = get_pareto_front(problem)
    ref_point = get_reference_point(args, problem)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"optimize {args.problem}")
    print(f"JAX 64-bit enabled: {jax_config.jax_enable_x64}")
    print(f"parallel benchmark workers: {args.n_jobs}")
    params = (args, problem, pareto_front, ref_point)
    if args.n_jobs == 1:
        data = [execute(run, *params) for run in instances]
    else:
        data = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run, *params) for run in instances)

    columns = ["run", "HV", "IGD", "GD", "Jac_calls", "CPU_time"]
    output = args.results_dir / f"{args.problem}-DpN-{args.algorithm}-{args.generation}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=columns).to_csv(output, index=False)


if __name__ == "__main__":
    np.random.seed(66)
    main()
