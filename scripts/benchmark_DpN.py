"""Benchmark DpN on the CF, ZDT, DTLZ, IDTLZ, and CONV4_2F problems."""

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

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import HV
from hvd.newton import DpN
from hvd.problems import (
    CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10,
    CMOP, CONV4_2F,
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7,
    IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4,
    ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
)
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import get_pareto_front, get_run_instances, plot, read_reference_set_data

PROBLEMS = {
    problem.__name__: problem
    for problem in (
        CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10,
        ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
        DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7,
        IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4,
        CONV4_2F,
    )
}
BOUNDARY_CONSTRAINTS = True
DEFAULT_MAX_ITERS = 5
DEFAULT_GENERATION = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DpN on stored EMOA populations.")
    parser.add_argument("problem", choices=PROBLEMS)
    parser.add_argument("--algorithm", default="NSGA-III", choices=["NSGA-II", "NSGA-III", "MOEAD", "SMS-EMOA"])
    parser.add_argument("--data-path", type=Path, default=ROOT / "mmd_data")
    parser.add_argument("--generation", type=int, default=DEFAULT_GENERATION)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "plots")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def execute(
    run: int,
    args: argparse.Namespace,
    problem: CMOP,
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
) -> dict[str, float | int]:
    ref, x0, y0, y_indices, eta = read_reference_set_data(
        args.data_path, args.problem, args.algorithm, run, args.generation, matching=True
    )
    reference = ReferenceSet(ref=ref, eta=eta, Y_idx=y_indices)
    initial_reference = reference.reference_set.copy()
    metrics = dict(
        GD=GenerationalDistance(pareto_front),
        IGD=InvertedGenerationalDistance(pareto_front),
        HV=HV(ref_point),
    )
    print(f"run {run}: precomputed eta={eta is not None}")
    for key, val in metrics.items():
        print(f"initial {key}: {val.compute(Y=y0)}")

    started = time.perf_counter_ns()
    optimizer = DpN(
        dim=problem.n_var,
        n_obj=problem.n_obj,
        ref=reference,
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
        type="igd",
        verbose=True,
        metrics=metrics,
        regularization=True,
    )
    Y = get_non_dominated(optimizer.run()[1])
    elapsed_microseconds = (time.perf_counter_ns() - started) / 1000.0
    name = f"{args.problem}_DpN_{args.algorithm}_run{run}_{args.generation}"
    plot(y0, Y, initial_reference, pareto_front, args.plot_dir / f"{name}.pdf", optimizer)
    objective_columns = [f"f{i + 1}" for i in range(problem.n_obj)]
    pd.DataFrame(y0, columns=objective_columns).to_csv(args.results_dir / f"{name}_y0.csv", index=False)
    pd.DataFrame(Y, columns=objective_columns).to_csv(args.results_dir / f"{name}_y.csv", index=False)
    out = {key: val.compute(Y=Y) for key, val in metrics.items()}
    out["run"] = run
    out["Jac_calls"] = optimizer.state.n_jac_evals
    out["wall_clock_time"] = elapsed_microseconds
    return out


def main() -> None:
    args = build_parser().parse_args()
    instances = get_run_instances(args.problem, args.algorithm, args.generation, args.data_path)
    if not instances:
        raise FileNotFoundError(
            f"No {args.problem} / {args.algorithm} runs at generation {args.generation} in {args.data_path}."
        )
    # Historical datasets use different decision dimensions, especially for DTLZ.
    population_file = args.data_path / (
        f"{args.problem}_{args.algorithm}_run_{instances[0]}_lastpopu_x_gen{args.generation}.csv"
    )
    n_var = pd.read_csv(population_file, header=None, nrows=1).shape[1]
    dimensions = {} if args.problem == "CONV4_2F" else {"n_var": n_var}
    problem = PROBLEMS[args.problem](boundary_constraints=BOUNDARY_CONSTRAINTS, **dimensions)
    pareto_front = get_pareto_front(problem)
    ref_point = (
        pd.read_csv(ROOT / "scripts" / "ref_point.csv", index_col="problem")
        .loc[args.problem].dropna().to_numpy(dtype=float)
    )

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"optimize {args.problem}")
    print(f"JAX 64-bit enabled: {jax_config.jax_enable_x64}")
    print(f"parallel benchmark workers: {args.n_jobs}")
    params = (args, problem, pareto_front, ref_point)
    if args.n_jobs == 1:
        data = [execute(run, *params) for run in instances]
    else:
        data = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run, *params) for run in instances)
    columns = ["run", "HV", "IGD", "GD", "Jac_calls", "wall_clock_time"]
    output = args.results_dir / f"{args.problem}-DpN-{args.algorithm}-{args.generation}.csv"
    pd.DataFrame(data, columns=columns).to_csv(output, index=False)


if __name__ == "__main__":
    np.random.seed(66)
    main()
