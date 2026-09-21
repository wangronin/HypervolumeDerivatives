"""Tune MMD Newton on the IDTLZ benchmark problems.

The tuner deliberately lives outside the optimizer implementation.  It uses
the vectorized indicators, while keeping ``hvd.mmd`` and ``hvd.mmd_newton``
unchanged.  Install Optuna to run a study::

    python -m pip install optuna
    python scripts/tune_MMD.py IDTLZ1 IDTLZ2 IDTLZ3 IDTLZ4

The objective is the mean averaged Hausdorff distance over the tuning runs,
where averaged Hausdorff distance is ``max(GD, IGD)``.  Validation runs are
never shown to the tuner.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd import laplace, linear, rational_quadratic, rbf
from hvd.mmd_newton import MMDNewton
from hvd.mmd_vectorized import MMD, MMDMatching
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
    "laplace": laplace,
    "linear": linear,
}


@dataclass(frozen=True)
class RunData:
    ref: dict[int, np.ndarray]
    x0: np.ndarray
    y0: np.ndarray
    y_indices: list[np.ndarray] | None
    eta: dict[int, np.ndarray] | None


class TunableMMDNewton(MMDNewton):
    """Use vectorized indicators and select a line-search policy externally."""

    def __init__(self, *args, line_search: str = "global", beta: float = 0.25, **kwargs):
        if line_search not in {"global", "individual"}:
            raise ValueError("line_search must be 'global' or 'individual'")

        kernel = kwargs.get("kernel", rbf)
        theta = kwargs.get("theta", 1.0)
        matching = kwargs.get("matching", True)
        super().__init__(*args, **kwargs)
        self.line_search = line_search

        indicator_type = MMDMatching if matching else MMD
        indicator_kwargs = dict(
            n_var=self.dim_p,
            n_obj=self.n_obj,
            ref=self.ref,
            func=self.state.func,
            jac=self.state.jac,
            hessian=kwargs["hessian"],
            kernel=kernel,
            theta=theta,
        )
        if matching:
            indicator_kwargs["beta"] = beta
        self.indicator = indicator_type(**indicator_kwargs)

    def _backtracking_line_search_global(self, step, residual, max_step_size=None):
        if self.line_search == "individual":
            return super()._backtracking_line_search_individual(step, residual, max_step_size)
        return super()._backtracking_line_search_global(step, residual, max_step_size)


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


def subset_partitions(
    y_indices: list[np.ndarray] | None, selected: np.ndarray
) -> list[np.ndarray] | None:
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
    ref, x0, y0, y_indices, eta = read_reference_set_data(
        data_path, problem_name, algorithm, run, generation
    )

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
    metric = "cityblock" if kernel_name == "laplace" else "sqeuclidean"
    distances = cdist(approximation, reference, metric=metric).ravel()
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
        data = load_run(
            data_path, problem_name, algorithm, run, generation, population_size
        )
        reference = np.vstack(list(data.ref.values()))
        theta = kernel_theta(
            config["kernel"], config.get("theta_multiplier", 1.0), data.y0, reference
        )
        optimizer = TunableMMDNewton(
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
            matching=config["matching"],
            regularization=config["regularization"],
            theta=theta,
            kernel=KERNELS[config["kernel"]],
            beta=config.get("beta", 0.25),
            line_search=config["line_search"],
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
            }
        )
    return float(np.mean(completed_scores)), records


def choice(trial, name: str, values: list):
    return values[0] if len(values) == 1 else trial.suggest_categorical(name, values)


def sample_configuration(trial, args) -> dict:
    kernel_name = choice(trial, "kernel", args.kernels)
    config = {
        "kernel": kernel_name,
        "matching": choice(trial, "matching", args.matching_choices),
        "regularization": choice(trial, "regularization", args.regularization_choices),
        "line_search": choice(trial, "line_search", args.line_searches),
    }
    if kernel_name != "linear":
        config["theta_multiplier"] = trial.suggest_float(
            "theta_multiplier", args.theta_min, args.theta_max, log=True
        )
    if config["matching"]:
        config["beta"] = trial.suggest_float("beta", args.beta_min, args.beta_max, log=True)
    return config


def config_from_params(params: dict, args) -> dict:
    return {
        "kernel": params.get("kernel", args.kernels[0]),
        "matching": params.get("matching", args.matching_choices[0]),
        "regularization": params.get("regularization", args.regularization_choices[0]),
        "line_search": params.get("line_search", args.line_searches[0]),
        "theta_multiplier": params.get("theta_multiplier", 1.0),
        "beta": params.get("beta", 0.25),
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


def tune_problem(problem_name: str, args, optuna) -> dict:
    problem = PROBLEMS[problem_name](boundary_constraints=args.boundary_constraints)
    pareto_front = get_pareto_front(problem)
    train_runs = parse_runs(args.train_runs)
    validation_runs = parse_runs(args.validation_runs)
    if not train_runs:
        raise ValueError("at least one tuning run is required")
    overlap = set(train_runs) & set(validation_runs)
    if overlap:
        raise ValueError(f"tuning and validation runs overlap: {sorted(overlap)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output_dir / 'mmd_tuning.db'}"
    suffix = "bounds" if args.boundary_constraints else "no_bounds"
    study_name = f"MMDNewton-{problem_name}-{args.algorithm}-{suffix}"
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.startup_trials,
        multivariate=True,
        group=True,
        constant_liar=args.jobs > 1,
    )
    max_resource = len(train_runs) * args.max_iters
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=make_pruner(optuna, args, max_resource),
    )

    def objective(trial):
        config = sample_configuration(trial, args)

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

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.jobs,
        gc_after_trial=True,
        show_progress_bar=args.jobs == 1,
    )
    study.trials_dataframe().to_csv(output_dir / f"{study_name}-trials.csv", index=False)

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
        "boundary_constraints": args.boundary_constraints,
        "best_tuning_ahd": study.best_value,
        "validation_ahd": validation_score,
        "best_config": best_config,
        "best_trial": study.best_trial.number,
        "completed_trials": sum(trial.state.name == "COMPLETE" for trial in study.trials),
        "validation_runs": validation_records,
    }
    with (output_dir / f"{study_name}-best.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="+", choices=PROBLEMS, help="IDTLZ problem(s) to tune")
    parser.add_argument("--data-path", type=Path, default=ROOT / "MMD_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts" / "tuning_results")
    parser.add_argument("--algorithm", default="NSGA-III", choices=["NSGA-II", "NSGA-III", "MOEAD"])
    parser.add_argument("--generation", type=int, default=300)
    parser.add_argument("--train-runs", default="1-10")
    parser.add_argument("--validation-runs", default="21-30")
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument(
        "--tune-points",
        type=int,
        default=60,
        help="representative population size during tuning; 0 keeps every point",
    )
    parser.add_argument(
        "--validation-points",
        type=int,
        default=0,
        help="representative validation population size; 0 keeps every point",
    )
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=None, help="study timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--storage", default=None, help="Optuna storage URL; defaults to local SQLite")
    parser.add_argument("--pruner", choices=["hyperband", "median", "none"], default="hyperband")
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=list(KERNELS))
    parser.add_argument("--theta-min", type=float, default=1e-2)
    parser.add_argument("--theta-max", type=float, default=1e2)
    parser.add_argument("--beta-min", type=float, default=5e-2)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--matching", choices=["both", "true", "false"], default="both")
    parser.add_argument("--regularization", choices=["both", "true", "false"], default="both")
    parser.add_argument(
        "--line-searches",
        nargs="+",
        choices=["global", "individual"],
        default=["global", "individual"],
    )
    parser.add_argument("--boundary-constraints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--failure-value", type=float, default=1e12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_iters < 1 or args.trials < 1 or args.jobs < 1:
        raise ValueError("max-iters, trials, and jobs must all be positive")
    if not 0 < args.theta_min < args.theta_max:
        raise ValueError("theta bounds must satisfy 0 < theta-min < theta-max")
    if not 0 < args.beta_min < args.beta_max:
        raise ValueError("beta bounds must satisfy 0 < beta-min < beta-max")
    args.matching_choices = boolean_choices(args.matching)
    args.regularization_choices = boolean_choices(args.regularization)
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import optuna
    except ImportError as error:
        raise SystemExit("Optuna is required: install it with `python -m pip install optuna`.") from error

    optuna.logging.set_verbosity(optuna.logging.INFO)
    results = [tune_problem(problem_name, args, optuna) for problem_name in args.problems]
    rows = [
        {
            "problem": result["problem"],
            "boundary_constraints": result["boundary_constraints"],
            "best_tuning_ahd": result["best_tuning_ahd"],
            "validation_ahd": result["validation_ahd"],
            **result["best_config"],
        }
        for result in results
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "MMDNewton-best-configurations.csv", index=False)


if __name__ == "__main__":
    main()
