from __future__ import annotations

import argparse
import json
import re
import sys
import time
from glob import glob
from pathlib import Path

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import hypervolume
from hvd.mmd_newton import MMDNewton
from hvd.mmd_vectorized import MMD, linear, rational_quadratic, rbf
from hvd.problems import IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import plot, read_reference_set_data

PROBLEMS = {
    "IDTLZ1": IDTLZ1,
    "IDTLZ2": IDTLZ2,
    "IDTLZ3": IDTLZ3,
    "IDTLZ4": IDTLZ4,
}
KERNELS = {
    "rbf": rbf,
    "rational_quadratic": rational_quadratic,
    "linear": linear,
}
BOUNDARY_CONSTRAINTS = True


def kernel_theta(
    kernel_name: str,
    multiplier: float,
    approximation: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Convert a scale-free multiplier to the kernel's inverse length scale."""
    if kernel_name == "linear":
        return 1.0
    distances = cdist(approximation, reference, metric="sqeuclidean").ravel()
    distances = distances[np.isfinite(distances) & (distances > np.finfo(float).eps)]
    characteristic_distance = np.median(distances) if len(distances) else 1.0
    return float(multiplier / characteristic_distance)


def find_best_result(args) -> tuple[Path, dict]:
    if args.best_config is not None:
        candidates = [args.best_config.expanduser().resolve()]
    else:
        name = f"MMDNewton-MMD-{args.problem}-{args.algorithm}-bounds-best.json"
        candidates = sorted(args.tuning_dir.expanduser().glob(name))

    if not candidates:
        raise FileNotFoundError(
            "No matching tuning result was found. Pass --best-config explicitly or check --tuning-dir."
        )
    if len(candidates) > 1:
        paths = "\n".join(f"  {path}" for path in candidates)
        raise ValueError(f"Multiple tuning results match; select one with --best-config:\n{paths}")

    path = candidates[0]
    with path.open() as stream:
        result = json.load(stream)
    if result.get("problem") != args.problem:
        raise ValueError(f"{path} contains results for {result.get('problem')}, not {args.problem}")
    if result.get("algorithm") != args.algorithm:
        raise ValueError(f"{path} contains results for {result.get('algorithm')}, not {args.algorithm}")
    if result.get("indicator") != "MMD":
        raise ValueError(f"{path} is not an MMD tuning result")
    if result.get("boundary_constraints") != BOUNDARY_CONSTRAINTS:
        raise ValueError(f"{path} was not tuned with boundary constraints")
    required = {"kernel", "regularization", "theta_multiplier"}
    missing = required - result.get("best_config", {}).keys()
    if missing:
        raise ValueError(f"{path} is missing tuned parameters: {sorted(missing)}")
    if result["best_config"]["kernel"] not in KERNELS:
        raise ValueError(f"{path} contains an unsupported kernel: {result['best_config']['kernel']}")
    return path, result


def get_pareto_front(problem) -> np.ndarray:
    pareto_front = np.asarray(problem.get_pareto_front())
    if len(pareto_front) > 1000:
        model = KMedoids(
            n_clusters=1000,
            method="alternate",
            random_state=0,
            init="k-medoids++",
        ).fit(pareto_front)
        pareto_front = pareto_front[model.medoid_indices_]
    return pareto_front


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark tuned MMD Newton on IDTLZ.")
    parser.add_argument("problem", choices=PROBLEMS)
    parser.add_argument("--algorithm", default="NSGA-III", choices=["NSGA-II", "NSGA-III", "MOEAD"])
    parser.add_argument("--data-path", type=Path, default=ROOT / "MMD_data")
    parser.add_argument("--tuning-dir", type=Path, default=Path.home() / "data" / "mmd-tuning")
    parser.add_argument("--best-config", type=Path, default=None)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=30,
        help="number of benchmark runs executed concurrently; use 1 for sequential execution",
    )
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "plots")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_jobs == 0:
        raise ValueError("n-jobs must not be zero")
    best_path, tuning_result = find_best_result(args)
    config = tuning_result["best_config"]

    generation = args.generation or int(tuning_result.get("generation", 300))
    max_iters = args.max_iters or int(tuning_result.get("max_iters", 5))
    kernel_name = config["kernel"]
    kernel = KERNELS[kernel_name]
    theta_multiplier = float(config.get("theta_multiplier", 1.0))
    regularization = bool(config["regularization"])

    problem = PROBLEMS[args.problem](boundary_constraints=BOUNDARY_CONSTRAINTS)
    pareto_front = get_pareto_front(problem)
    ref_point = pd.read_csv(ROOT / "scripts" / "ref_point.csv", index_col="problem").loc[
        args.problem
    ].values
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"optimize {args.problem}")
    print(f"tuning result: {best_path}")
    print(f"configuration: {json.dumps(config, sort_keys=True)}")
    print(f"JAX 64-bit enabled: {jax_config.jax_enable_x64}")
    print(f"parallel benchmark workers: {args.n_jobs}")

    def execute(run: int) -> np.ndarray:
        ref, x0, y0, y_indices, eta = read_reference_set_data(
            args.data_path,
            args.problem,
            args.algorithm,
            run,
            generation,
        )
        reference = np.vstack(list(ref.values()))
        theta = kernel_theta(kernel_name, theta_multiplier, y0, reference)
        metric = MMD(
            n_var=problem.n_obj,
            n_obj=problem.n_obj,
            ref=pareto_front.copy(),
            kernel=kernel,
            theta=theta,
        )
        metrics = {
            "GD": GenerationalDistance(pareto_front),
            "IGD": InvertedGenerationalDistance(pareto_front),
        }
        print(f"run {run}: theta={theta}, precomputed eta={eta is not None}")
        print(f"initial HV: {hypervolume(y0, ref=ref_point)}")
        print(f"initial GD: {metrics['GD'].compute(Y=y0)}")
        print(f"initial IGD: {metrics['IGD'].compute(Y=y0)}")
        print(f"initial MMD: {metric.compute(Y=y0)}")

        started = time.perf_counter_ns()
        optimizer = MMDNewton(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            ref=ReferenceSet(ref=ref, eta=eta, Y_idx=y_indices),
            func=problem.objective,
            jac=problem.objective_jacobian,
            hessian=problem.objective_hessian,
            g=problem.ieq_constraint,
            g_jac=problem.ieq_jacobian,
            g_hessian=problem.ieq_hessian,
            N=len(x0),
            X0=x0,
            xl=problem.xl,
            xu=problem.xu,
            max_iters=max_iters,
            verbose=True,
            metrics=metrics,
            matching=False,
            regularization=regularization,
            theta=theta,
            kernel=kernel,
        )
        Y = get_non_dominated(optimizer.run()[1])
        elapsed_microseconds = (time.perf_counter_ns() - started) / 1000.0
        print(optimizer.history_R_norm)

        figure = args.plot_dir / f"{args.problem}_MMD_{args.algorithm}_run{run}_{generation}.pdf"
        plot(y0, Y, reference, pareto_front, figure, optimizer)
        hv_value = hypervolume(Y, ref=ref_point)
        igd_value = InvertedGenerationalDistance(pareto_front).compute(Y=Y)
        gd_value = GenerationalDistance(pareto_front).compute(Y=Y)
        mmd_value = metric.compute(Y=Y)
        return np.array(
            [hv_value, igd_value, gd_value, mmd_value, optimizer.state.n_jac_evals, elapsed_microseconds]
        )

    run_ids = sorted(
        int(re.findall(r"run_(\d+)_", path)[0])
        for path in glob(
            str(
                args.data_path
                / f"{args.problem}_{args.algorithm}_run_*_lastpopu_x_gen{generation}.csv"
            )
        )
    )
    if args.n_jobs == 1:
        data = [execute(run) for run in run_ids]
    else:
        data = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run=run) for run in run_ids)
    columns = ["HV", "IGD", "GD", "MMD", "Jac_calls", "wall_clock_time"]
    output = args.results_dir / f"{args.problem}-MMD-{args.algorithm}-{generation}.csv"
    pd.DataFrame(np.asarray(data), columns=columns).to_csv(output, index=False)


if __name__ == "__main__":
    np.random.seed(66)
    main()
