"""Tune MMD Newton on the IDTLZ benchmark problems.

The tuner uses the vectorized original MMD indicator, without point matching,
and the per-run reference shift directions stored in the MMD data. IDTLZ box
bounds are always included as inequality constraints. Install Optuna to run a
study::

    python -m pip install optuna
    python scripts/tune_MMD.py IDTLZ1 --workers 15

On Slurm, launch one problem per job from the cluster-specific submission
script and pass ``--workers 15`` when 15 CPUs are allocated to that job.
``--trials`` is the total number of trials, not the number assigned to every
worker.

Each finished trial is appended to ``<study>-progress.log`` in the output
directory. The default ``<study>.journal`` file is Optuna's durable study
storage and allows interrupted studies to resume.

The objective is the mean averaged Hausdorff distance over the tuning runs,
where averaged Hausdorff distance is ``max(GD, IGD)``.  Validation runs are
never shown to the tuner.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd_newton import MMDNewton
from hvd.mmd_vectorized import linear, rational_quadratic, rbf
from hvd.problems import IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import read_reference_set_data

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


@dataclass(frozen=True)
class RunData:
    ref: dict[int, np.ndarray]
    x0: np.ndarray
    y0: np.ndarray
    y_indices: list[np.ndarray] | None
    eta: dict[int, np.ndarray] | None


def parse_runs(specification: str) -> list[int]:
    """Parse comma-separated run numbers and inclusive ranges."""
    runs: list[int] = []
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(value) for value in token.split("-", maxsplit=1))
            if stop < start:
                raise ValueError(f"invalid run range: {token}")
            runs.extend(range(start, stop + 1))
        else:
            runs.append(int(token))
    return list(dict.fromkeys(runs))


def representative_indices(points: np.ndarray, size: int | None) -> np.ndarray:
    """Deterministic farthest-point subset in normalized objective space."""
    points = np.asarray(points)
    if size is None or size <= 0 or size >= len(points):
        return np.arange(len(points))

    scale = np.ptp(points, axis=0)
    scale[scale == 0] = 1
    normalized = (points - points.min(axis=0)) / scale
    center = normalized.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(normalized - center, axis=1)))]
    min_distance = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    min_distance[selected[0]] = -np.inf
    while len(selected) < size:
        index = int(np.argmax(min_distance))
        selected.append(index)
        distance = np.linalg.norm(normalized - normalized[index], axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    return np.sort(np.asarray(selected))


def subset_partitions(y_indices: list[np.ndarray] | None, selected: np.ndarray) -> list[np.ndarray] | None:
    if y_indices is None or len(y_indices) <= 1:
        return None
    labels = np.full(max(max(indices) for indices in y_indices) + 1, -1, dtype=int)
    for label, indices in enumerate(y_indices):
        labels[indices] = label
    selected_labels = labels[selected]
    return [np.flatnonzero(selected_labels == label) for label in np.unique(selected_labels)]


def load_run(
    data_path: Path,
    problem_name: str,
    algorithm: str,
    run: int,
    generation: int,
    population_size: int | None,
) -> RunData:
    ref, x0, y0, y_indices, eta = read_reference_set_data(data_path, problem_name, algorithm, run, generation)

    # ``read_reference_set_data`` has a legacy corner case: when a merged
    # reference contains fewer points than the population, it truncates to the
    # number of components rather than to the number of reference points.  Do
    # the intended truncation locally so the existing helper remains untouched.
    raw_x = pd.read_csv(
        data_path / f"{problem_name}_{algorithm}_run_{run}_lastpopu_x_gen{generation}.csv",
        header=None,
    ).values
    raw_y = pd.read_csv(
        data_path / f"{problem_name}_{algorithm}_run_{run}_lastpopu_y_gen{generation}.csv",
        header=None,
    ).values
    raw_labels = (
        pd.read_csv(
            data_path / f"{problem_name}_{algorithm}_run_{run}_lastpopu_labels_gen{generation}.csv",
            header=None,
        ).values.ravel()
        - 1
    )
    keep = (raw_labels != -2) & (raw_labels != -1)
    raw_x, raw_y, raw_labels = raw_x[keep], raw_y[keep], raw_labels[keep]
    intended_size = min(len(raw_x), sum(len(component) for component in ref.values()))
    if len(x0) != intended_size:
        x0, y0, raw_labels = raw_x[:intended_size], raw_y[:intended_size], raw_labels[:intended_size]
        y_indices = (
            None
            if len(ref) == 1
            else [np.flatnonzero(raw_labels == label) for label in np.unique(raw_labels)]
        )

    selected = representative_indices(y0, population_size)
    return RunData(
        ref=ref,
        x0=x0[selected],
        y0=y0[selected],
        y_indices=subset_partitions(y_indices, selected),
        eta=eta,
    )


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


def averaged_hausdorff(points: np.ndarray, pareto_front: np.ndarray) -> tuple[float, float, float]:
    points = get_non_dominated(np.asarray(points))
    gd = GenerationalDistance(pareto_front).compute(Y=points)
    igd = InvertedGenerationalDistance(pareto_front).compute(Y=points)
    return float(max(gd, igd)), float(gd), float(igd)


def evaluate_configuration(
    *,
    problem,
    problem_name: str,
    pareto_front: np.ndarray,
    config: dict,
    run_ids: Iterable[int],
    data_path: Path,
    algorithm: str,
    generation: int,
    max_iters: int,
    population_size: int | None,
    report: Callable[[float, int], None] | None = None,
) -> tuple[float, list[dict]]:
    records: list[dict] = []
    completed_scores: list[float] = []
    run_ids = list(run_ids)
    for run_index, run in enumerate(run_ids):
        data = load_run(data_path, problem_name, algorithm, run, generation, population_size)
        reference = np.vstack(list(data.ref.values()))
        theta = kernel_theta(config["kernel"], config.get("theta_multiplier", 1.0), data.y0, reference)
        optimizer = MMDNewton(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            ref=ReferenceSet(ref=data.ref, eta=data.eta, Y_idx=data.y_indices),
            func=problem.objective,
            jac=problem.objective_jacobian,
            hessian=problem.objective_hessian,
            g=problem.ieq_constraint,
            g_jac=problem.ieq_jacobian,
            g_hessian=problem.ieq_hessian,
            N=len(data.x0),
            X0=data.x0,
            xl=problem.xl,
            xu=problem.xu,
            max_iters=max_iters,
            verbose=False,
            matching=False,
            regularization=config["regularization"],
            theta=theta,
            kernel=KERNELS[config["kernel"]],
        )

        started = time.perf_counter()
        score, gd, igd = averaged_hausdorff(data.y0, pareto_front)
        initial_score = score
        for iteration in range(max_iters):
            optimizer.newton_iteration()
            score, gd, igd = averaged_hausdorff(optimizer.state.Y, pareto_front)
            if not np.isfinite(score):
                score = float("inf")
            if report is not None:
                aggregate = float(np.mean(completed_scores + [score]))
                step = run_index * max_iters + iteration + 1
                report(aggregate, step)

        completed_scores.append(score)
        records.append(
            {
                "run": run,
                "theta": theta,
                "initial_ahd": initial_score,
                "ahd": score,
                "gd": gd,
                "igd": igd,
                "seconds": time.perf_counter() - started,
                "n_points": len(data.x0),
                "precomputed_shift_direction": data.eta is not None,
            }
        )
    return float(np.mean(completed_scores)), records


def choice(trial, name: str, values: list):
    return values[0] if len(values) == 1 else trial.suggest_categorical(name, values)


def sample_configuration(trial, args) -> dict:
    kernel_name = choice(trial, "kernel", args.kernels)
    config = {
        "kernel": kernel_name,
        "regularization": choice(trial, "regularization", args.regularization_choices),
    }
    if kernel_name != "linear":
        config["theta_multiplier"] = trial.suggest_float(
            "theta_multiplier", args.theta_min, args.theta_max, log=True
        )
    return config


def config_from_params(params: dict, args) -> dict:
    return {
        "kernel": params.get("kernel", args.kernels[0]),
        "regularization": params.get("regularization", args.regularization_choices[0]),
        "theta_multiplier": params.get("theta_multiplier", 1.0),
    }


def boolean_choices(value: str) -> list[bool]:
    return {"both": [False, True], "false": [False], "true": [True]}[value]


def get_pareto_front(problem, max_points: int = 1000) -> np.ndarray:
    pareto_front = np.asarray(problem.get_pareto_front())
    return pareto_front[representative_indices(pareto_front, max_points)]


def make_pruner(optuna, args, max_resource: int):
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=max(3, args.startup_trials), n_warmup_steps=args.max_iters
        )
    return optuna.pruners.HyperbandPruner(
        min_resource=max(1, args.max_iters),
        max_resource=max_resource,
        reduction_factor=args.reduction_factor,
    )


def study_name(problem_name: str, args) -> str:
    return f"MMDNewton-MMD-{problem_name}-{args.algorithm}-bounds"


def storage_specification(problem_name: str, args) -> str:
    if args.storage:
        return args.storage if "://" in args.storage else str(Path(args.storage).resolve())
    return str((args.output_dir.resolve() / f"{study_name(problem_name, args)}.journal"))


def progress_log_path(problem_name: str, args) -> Path:
    return args.output_dir.resolve() / f"{study_name(problem_name, args)}-progress.log"


def make_progress_callback(problem_name: str, args):
    """Append one process-safe JSON record after every finished trial."""
    path = progress_log_path(problem_name, args)

    def log_progress(study, trial) -> None:
        trials = study.get_trials(deepcopy=False)
        state_counts: dict[str, int] = {}
        for completed_trial in trials:
            state = completed_trial.state.name
            state_counts[state] = state_counts.get(state, 0) + 1
        try:
            best_value = study.best_value
        except ValueError:
            best_value = None
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "problem": problem_name,
            "pid": os.getpid(),
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "best_value": best_value,
            "state_counts": state_counts,
            "params": trial.params,
        }
        with path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return log_progress


def make_storage(optuna, specification: str):
    if "://" in specification:
        return specification
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(file_path=specification))


def create_study(problem_name: str, args, optuna, worker_id: int = 0):
    train_runs = parse_runs(args.train_runs)
    sampler = optuna.samplers.TPESampler(
        seed=args.seed + worker_id,
        n_startup_trials=args.startup_trials,
        multivariate=True,
        group=True,
        constant_liar=args.workers > 1,
    )
    max_resource = len(train_runs) * args.max_iters
    return optuna.create_study(
        study_name=study_name(problem_name, args),
        storage=make_storage(optuna, storage_specification(problem_name, args)),
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=make_pruner(optuna, args, max_resource),
    )


def make_objective(problem_name: str, args, optuna):
    problem = PROBLEMS[problem_name](boundary_constraints=BOUNDARY_CONSTRAINTS)
    pareto_front = get_pareto_front(problem)
    train_runs = parse_runs(args.train_runs)

    def objective(trial):
        config = sample_configuration(trial, args)
        trial.set_user_attr("worker_pid", os.getpid())

        def report(value: float, step: int) -> None:
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            score, records = evaluate_configuration(
                problem=problem,
                problem_name=problem_name,
                pareto_front=pareto_front,
                config=config,
                run_ids=train_runs,
                data_path=args.data_path,
                algorithm=args.algorithm,
                generation=args.generation,
                max_iters=args.max_iters,
                population_size=args.tune_points,
                report=report,
            )
        except optuna.TrialPruned:
            raise
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            trial.set_user_attr("failure", f"{type(error).__name__}: {error}")
            return args.failure_value
        trial.set_user_attr("runs", records)
        return score

    return objective


def run_study_worker(payload) -> None:
    problem_name, args, trial_count, worker_id = payload
    if trial_count == 0:
        return
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)
    study = create_study(problem_name, args, optuna, worker_id)
    study.optimize(
        make_objective(problem_name, args, optuna),
        n_trials=trial_count,
        timeout=args.timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
        callbacks=[make_progress_callback(problem_name, args)],
    )


def distribute_trials(n_trials: int, n_workers: int) -> list[int]:
    """Split an exact total number of trials as evenly as possible."""
    n_workers = min(n_trials, n_workers)
    quotient, remainder = divmod(n_trials, n_workers)
    return [quotient + (worker_id < remainder) for worker_id in range(n_workers)]


def tune_problem(problem_name: str, args, optuna) -> dict:
    train_runs = parse_runs(args.train_runs)
    validation_runs = parse_runs(args.validation_runs)
    if not train_runs:
        raise ValueError("at least one tuning run is required")
    overlap = set(train_runs) & set(validation_runs)
    if overlap:
        raise ValueError(f"tuning and validation runs overlap: {sorted(overlap)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = storage_specification(problem_name, args)
    progress_log = progress_log_path(problem_name, args)
    print(f"Optuna storage: {storage}", flush=True)
    print(f"Trial progress: {progress_log}", flush=True)
    if args.workers > 1 and storage.startswith("sqlite"):
        raise ValueError(
            "SQLite is not safe for parallel Optuna workers; use the default journal or an RDB server"
        )

    # Create the study before workers start so initialization cannot race.
    create_study(problem_name, args, optuna)
    trial_counts = distribute_trials(args.trials, args.workers)
    payloads = [
        (problem_name, args, trial_count, worker_id) for worker_id, trial_count in enumerate(trial_counts)
    ]
    if len(payloads) == 1:
        run_study_worker(payloads[0])
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(payloads)) as pool:
            pool.map(run_study_worker, payloads)

    study = create_study(problem_name, args, optuna)
    name = study_name(problem_name, args)
    study.trials_dataframe().to_csv(output_dir / f"{name}-trials.csv", index=False)

    problem = PROBLEMS[problem_name](boundary_constraints=BOUNDARY_CONSTRAINTS)
    pareto_front = get_pareto_front(problem)
    best_config = config_from_params(study.best_trial.params, args)
    validation_score, validation_records = evaluate_configuration(
        problem=problem,
        problem_name=problem_name,
        pareto_front=pareto_front,
        config=best_config,
        run_ids=validation_runs or train_runs,
        data_path=args.data_path,
        algorithm=args.algorithm,
        generation=args.generation,
        max_iters=args.max_iters,
        population_size=args.validation_points,
    )
    result = {
        "problem": problem_name,
        "algorithm": args.algorithm,
        "generation": args.generation,
        "max_iters": args.max_iters,
        "indicator": "MMD",
        "boundary_constraints": BOUNDARY_CONSTRAINTS,
        "best_tuning_ahd": study.best_value,
        "validation_ahd": validation_score,
        "best_config": best_config,
        "best_trial": study.best_trial.number,
        "completed_trials": sum(trial.state.name == "COMPLETE" for trial in study.trials),
        "workers": len(payloads),
        "storage": storage,
        "progress_log": str(progress_log),
        "validation_runs": validation_records,
    }
    with (output_dir / f"{name}-best.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    pd.DataFrame(
        [
            {
                "problem": result["problem"],
                "boundary_constraints": result["boundary_constraints"],
                "best_tuning_ahd": result["best_tuning_ahd"],
                "validation_ahd": result["validation_ahd"],
                **result["best_config"],
            }
        ]
    ).to_csv(output_dir / f"{name}-summary.csv", index=False)
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="+", choices=PROBLEMS)
    parser.add_argument("--data-path", type=Path, default="/home/wangh5/data/MMD_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts" / "tuning_results")
    parser.add_argument("--algorithm", default="NSGA-III", choices=["NSGA-II", "NSGA-III", "MOEAD"])
    parser.add_argument("--generation", type=int, default=300)
    parser.add_argument("--train-runs", default="1-20")
    parser.add_argument("--validation-runs", default="21-30")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--tune-points",
        type=int,
        default=0,
        help="representative population size during tuning; 0 keeps every point",
    )
    parser.add_argument(
        "--validation-points",
        type=int,
        default=0,
        help="representative validation population size; 0 keeps every point",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=150,
        help="total trials added by this invocation, divided across all workers",
    )
    parser.add_argument("--timeout", type=float, default=None, help="study timeout in seconds")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes for each problem; normally match this to allocated CPUs",
    )
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna journal path or RDB URL; defaults to one journal per problem",
    )
    parser.add_argument("--pruner", choices=["hyperband", "median", "none"], default="hyperband")
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=list(KERNELS))
    parser.add_argument("--theta-min", type=float, default=1e-3)
    parser.add_argument("--theta-max", type=float, default=1e4)
    parser.add_argument("--regularization", choices=["both", "true", "false"], default="both")
    parser.add_argument("--failure-value", type=float, default=1e12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_iters < 1 or args.trials < 1 or args.workers < 1:
        raise ValueError("max-iters, trials, and workers must all be positive")
    if not 0 < args.theta_min < args.theta_max:
        raise ValueError("theta bounds must satisfy 0 < theta-min < theta-max")
    args.regularization_choices = boolean_choices(args.regularization)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Each worker owns one CPU core. Prevent BLAS from starting another thread
    # pool inside every process and oversubscribing the Slurm allocation.
    if args.workers > 1:
        thread_variables = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        for variable in thread_variables:
            os.environ.setdefault(variable, "1")

    try:
        import optuna
    except ImportError as error:
        raise SystemExit("Optuna is required: install it with `python -m pip install optuna`.") from error

    optuna.logging.set_verbosity(optuna.logging.INFO)
    for problem_name in args.problems:
        tune_problem(problem_name, args, optuna)


if __name__ == "__main__":
    main()
