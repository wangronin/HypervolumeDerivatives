from __future__ import annotations

import argparse
import json
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
from hvd.mmd import MMD, MMDMatching
from hvd.mmd.kernels import RBF, RationalQuadratic
from hvd.mmd_newton import MMDN
from hvd.problems import CMOP, IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import MMDConfig, get_pareto_front, kernel_theta, plot, read_reference_set_data

PROBLEMS = {
    "IDTLZ1": IDTLZ1,
    "IDTLZ2": IDTLZ2,
    "IDTLZ3": IDTLZ3,
    "IDTLZ4": IDTLZ4,
}
KERNELS = {
    "rbf": RBF,
    "rational_quadratic": RationalQuadratic,
}
BOUNDARY_CONSTRAINTS = True
DEFAULT_CONFIG: MMDConfig = {"kernel": "rbf", "theta_multiplier": 1.0, "regularization": False}
DEFAULT_MAX_ITERS = 5
DEFAULT_GENERATION = 300


def get_run_instances(problem: str, algorithm: str, generation: int, data_path: Path) -> list[int]:
    file_name = f"{problem}_{algorithm}_run_*_lastpopu_x_gen{generation}.csv"
    return sorted(int(path.name.split("_run_")[1].split("_")[0]) for path in data_path.glob(file_name))


def get_config(args: argparse.Namespace) -> tuple[MMDConfig, argparse.Namespace]:
    indicator = "MMDMatching" if args.matching else "MMD"
    path = args.tuning_dir / f"MMDNewton-{indicator}-{args.problem}-{args.algorithm}-bounds-best.json"
    result = {}
    if path.is_file():
        with path.open() as stream:
            result = json.load(stream)
    args.generation = (
        args.generation if args.generation is not None else result.get("generation", DEFAULT_GENERATION)
    )
    args.max_iters = (
        args.max_iters if args.max_iters is not None else result.get("max_iters", DEFAULT_MAX_ITERS)
    )
    return result.get("best_config", DEFAULT_CONFIG), args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MMD Newton on IDTLZ.")
    parser.add_argument("problem", choices=PROBLEMS)
    parser.add_argument("--algorithm", default="NSGA-III", choices=["NSGA-II", "NSGA-III", "MOEAD"])
    parser.add_argument("--data-path", type=Path, default=ROOT / "MMD_data")
    parser.add_argument("--tuning-dir", type=Path, default=Path.home() / "mmd-tuning")
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--matching", action="store_true", dest="matching")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "plots")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def execute(
    run: int,
    args: argparse.Namespace,
    config: MMDConfig,
    problem: CMOP,
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
) -> dict[str, float | int]:
    ref, x0, y0, y_indices, eta = read_reference_set_data(
        args.data_path, args.problem, args.algorithm, run, args.generation, args.matching
    )
    reference = ReferenceSet(ref=ref, eta=eta, Y_idx=y_indices)
    theta = kernel_theta(config["theta_multiplier"], y0, np.vstack(list(ref.values())))
    kernel = KERNELS[config["kernel"]](theta=theta)
    metrics = dict(
        GD=GenerationalDistance(pareto_front),
        IGD=InvertedGenerationalDistance(pareto_front),
        MMD=MMD(problem.n_obj, problem.n_obj, pareto_front, kernel=kernel),
        HV=HV(ref_point),
    )
    print(f"run {run}: theta={theta}, precomputed eta={eta is not None}")
    for key, val in metrics.items():
        print(f"initial {key}: {val.compute(Y=y0)}")

    started = time.perf_counter_ns()
    indicator_type = MMDMatching if args.matching else MMD
    indicator = indicator_type(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        ref=reference,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        kernel=kernel,
        **({"beta": config.get("beta", 0.25)} if args.matching else {}),
    )
    optimizer = MMDN(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        indicator=indicator,
        func=problem.objective,
        jac=problem.objective_jacobian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
        N=len(x0),
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=args.max_iters,
        verbose=True,
        metrics=metrics,
        regularization=config["regularization"],
    )
    Y = get_non_dominated(optimizer.run()[1])
    elapsed_microseconds = (time.perf_counter_ns() - started) / 1000.0
    figure_name = args.plot_dir / f"{args.problem}_MMD_{args.algorithm}_run{run}_{args.generation}.pdf"
    plot(y0, Y, reference.reference_set, pareto_front, figure_name, optimizer)
    out = {key: val.compute(Y=Y) for key, val in metrics.items()}
    out["Jac_calls"] = optimizer.state.n_jac_evals
    out["wall_clock_time"] = elapsed_microseconds
    return out


def main() -> None:
    args = build_parser().parse_args()
    config, args = get_config(args)
    problem = PROBLEMS[args.problem](boundary_constraints=BOUNDARY_CONSTRAINTS)
    pareto_front = get_pareto_front(problem)
    instances = get_run_instances(args.problem, args.algorithm, args.generation, args.data_path)
    ref_point = pd.read_csv(ROOT / "scripts" / "ref_point.csv", index_col="problem").loc[args.problem].values

    # make the plot and result folder
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"optimize {args.problem}")
    print(f"configuration: {json.dumps(config, sort_keys=True)}")
    print(f"JAX 64-bit enabled: {jax_config.jax_enable_x64}")
    print(f"parallel benchmark workers: {args.n_jobs}")
    # execute the algorithm
    params = (args, config, problem, pareto_front, ref_point)
    if args.n_jobs == 1:  # sequential
        data = [execute(run, *params) for run in instances]
    else:  # multi-process execution
        data = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run, *params) for run in instances)
    # save the benchmarking data
    columns = ["HV", "IGD", "GD", "MMD", "Jac_calls", "wall_clock_time"]
    output = args.results_dir / f"{args.problem}-MMD-{args.algorithm}-{args.generation}.csv"
    pd.DataFrame(data, columns=columns).to_csv(output, index=False)


if __name__ == "__main__":
    np.random.seed(66)
    main()
