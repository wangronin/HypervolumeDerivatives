"""Tune MMD Newton on training runs and export candidates for visual review.

Input: prepared populations/reference sets in data/. Output: an Optuna journal,
trial history, and the five best COMPLETE configurations. Held-out runs are
reserved for benchmarking after a candidate has been inspected and selected.
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tune_MMD.mmd_workflow import (
    ALGORITHMS,
    KERNELS,
    PROBLEM_NAMES,
    evaluate,
    exclusive_lock,
    load_problem,
    parse_runs,
    study_name,
    write_json,
)
from scripts.utils import get_pareto_front


class EvaluationError(RuntimeError):
    """A failed trial, distinct from pruning or a completed training score."""


def create_study(problem_name, args, worker=0):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    name = study_name(problem_name, args.algorithm, args.matching)
    if args.pruner == "none":
        pruner = optuna.pruners.NopPruner()
    elif args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(3, args.startup_trials), n_warmup_steps=args.max_iters
        )
    else:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=args.max_iters,
            max_resource=len(parse_runs(args.train_runs)) * args.max_iters,
            reduction_factor=3,
        )
    sampler = optuna.samplers.TPESampler(
        seed=args.seed + worker,
        n_startup_trials=args.startup_trials,
        multivariate=True,
        group=True,
        constant_liar=args.workers > 1,
    )
    return optuna.create_study(
        study_name=name,
        storage=JournalStorage(JournalFileBackend(str(args.output_dir / f"{name}.journal"))),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def sample_configuration(trial, args):
    def choice(name, values):
        return values[0] if len(values) == 1 else trial.suggest_categorical(name, values)

    kernel = choice("kernel", args.kernels)
    regularization = choice(
        "regularization", {"both": [False, True], "true": [True], "false": [False]}[args.regularization]
    )
    beta = trial.suggest_float("beta", args.beta_min, args.beta_max, log=True) if args.matching else None
    configuration = dict(
        kernel=kernel,
        regularization=regularization,
        theta_multiplier=trial.suggest_float("theta_multiplier", args.theta_min, args.theta_max, log=True),
    )
    if args.matching:
        configuration["beta"] = beta
    return configuration


def make_objective(problem_name, args):
    import optuna

    runs = parse_runs(args.train_runs)
    problem = load_problem(problem_name, args, runs[0])
    front = np.load(args.front_file)

    def objective(trial):
        configuration = sample_configuration(trial, args)
        trial.set_user_attr("config", configuration)
        records = []
        try:
            for run_index, run in enumerate(runs):

                def report(value, iteration):
                    if time.monotonic() >= args.stop_deadline:
                        raise EvaluationError("Allocation checkpoint deadline reached")
                    score = float(np.mean([row["ahd"] for row in records] + [value]))
                    trial.report(score, run_index * args.max_iters + iteration)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                records.append(evaluate(run, problem_name, args, configuration, problem, front, report))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, EvaluationError) as error:
            trial.set_user_attr("failure", f"{type(error).__name__}: {error}")
            raise EvaluationError(str(error)) from error
        finally:
            trial.set_user_attr("runs", records)
        return float(np.mean([row["ahd"] for row in records]))

    return objective


def log_progress(study, trial, path):
    """Keep completion order for later reporting; no live status exports."""
    record = dict(
        trial=trial.number,
        state=trial.state.name,
        value=trial.value,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record) + "\n")
        stream.flush()


def run_worker(payload):
    import optuna

    problem_name, args, count, worker = payload
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = create_study(problem_name, args, worker)
    timeout = args.start_deadline - time.monotonic()
    if timeout <= 0:
        return
    name = study_name(problem_name, args.algorithm, args.matching)
    study.optimize(
        make_objective(problem_name, args),
        n_trials=count,
        timeout=None if np.isinf(timeout) else timeout,
        n_jobs=1,
        gc_after_trial=True,
        callbacks=[lambda study, trial: log_progress(study, trial, args.output_dir / f"{name}-progress.log")],
        catch=(EvaluationError,),
    )


def export_results(problem_name, args, study):
    """Export full trial records and ranked training candidates, without validation."""
    name = study.study_name
    trials = study.get_trials(deepcopy=False)
    counts = {
        state: sum(trial.state.name == state for trial in trials)
        for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
    }
    history = [
        dict(
            number=trial.number,
            state=trial.state.name,
            value=trial.value,
            params=trial.params,
            intermediate_values=trial.intermediate_values,
            user_attrs=trial.user_attrs,
            datetime_start=trial.datetime_start.isoformat() if trial.datetime_start else None,
            datetime_complete=trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        )
        for trial in trials
    ]
    write_json(args.output_dir / f"{name}-trials.json", history)
    valid = sorted(
        (
            trial
            for trial in trials
            if trial.state.name == "COMPLETE"
            and trial.value is not None
            and np.isfinite(trial.value)
            and len(trial.user_attrs.get("runs", [])) == len(parse_runs(args.train_runs))
        ),
        key=lambda trial: (trial.value, trial.number),
    )
    metadata = dict(
        problem=problem_name,
        algorithm=args.algorithm,
        generation=args.generation,
        max_iters=args.max_iters,
        indicator="MMDMatching" if args.matching else "MMD",
        project_box_constraints=True,
        train_run_ids=parse_runs(args.train_runs),
        validation_run_ids=parse_runs(args.validation_runs),
        target_trials=args.trials,
        state_counts=counts,
        tuning_complete=len(trials) >= args.trials and not counts["RUNNING"] and not counts["WAITING"],
        search_space=dict(
            kernels=args.kernels,
            theta=[args.theta_min, args.theta_max],
            beta=[args.beta_min, args.beta_max] if args.matching else None,
            regularization=args.regularization,
        ),
        front_file=str(args.front_file.relative_to(args.output_dir)),
    )
    candidates = [
        {
            **metadata,
            "best_trial": trial.number,
            "best_tuning_ahd": trial.value,
            "best_config": trial.user_attrs["config"],
            "training_runs": trial.user_attrs["runs"],
        }
        for trial in valid[:5]
    ]
    write_json(args.output_dir / f"{name}-candidates.json", candidates)
    # The numeric best remains separate from the later frozen selection.
    write_json(args.output_dir / f"{name}-best.json", candidates[0] if candidates else metadata)
    return metadata


def tune_problem(problem_name, args):
    name = study_name(problem_name, args.algorithm, args.matching)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.output_dir / f"{name}.lock"):
        study = create_study(problem_name, args)
        if any(trial.state.name in {"RUNNING", "WAITING"} for trial in study.trials):
            raise RuntimeError("Study has unfinished trials. Resolve their state before resuming.")
        settings = dict(
            generation=args.generation,
            max_iters=args.max_iters,
            seed=args.seed,
            train_runs=args.train_runs,
            validation_runs=args.validation_runs,
            kernels=args.kernels,
            regularization=args.regularization,
            theta=[args.theta_min, args.theta_max],
            beta=[args.beta_min, args.beta_max],
        )
        previous = study.user_attrs.get("settings")
        if study.trials and previous != settings:
            raise ValueError("Use a new output directory for a different tuning setup or an older study.")
        study.set_user_attr("settings", settings)
        args.front_file = args.output_dir / "fronts" / f"{name}.npy"
        remaining = max(0, args.trials - len(study.trials))
        if remaining:
            if not args.front_file.exists():
                random.seed(args.seed)
                np.random.seed(args.seed)
                problem = load_problem(problem_name, args, parse_runs(args.train_runs)[0])
                args.front_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.front_file, get_pareto_front(problem))
            started = time.monotonic()
            args.start_deadline = started + args.timeout if args.timeout is not None else np.inf
            args.stop_deadline = started + args.stop_after if args.stop_after is not None else np.inf
            workers = min(args.workers, remaining)
            quotient, remainder = divmod(remaining, workers)
            payloads = [
                (problem_name, args, quotient + (worker < remainder), worker) for worker in range(workers)
            ]
            if workers == 1:
                run_worker(payloads[0])
            else:
                with mp.get_context("spawn").Pool(workers) as pool:
                    pool.map(run_worker, payloads)
        result = export_results(problem_name, args, create_study(problem_name, args))
        print(
            f"{problem_name} / {args.algorithm}: {sum(result['state_counts'].values())}/{args.trials} attempts; "
            f"{result['state_counts']}",
            flush=True,
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="+", choices=PROBLEM_NAMES)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="NSGA-III")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mmd_tuning_results" / "tuning")
    parser.add_argument("--generation", type=int, default=600)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--train-runs", default="1-20")
    parser.add_argument("--validation-runs", default="21-30")
    parser.add_argument(
        "--trials", type=int, default=100, help="Cumulative attempts, including pruned/failed trials"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, help="Stop starting trials after this many seconds")
    parser.add_argument("--stop-after", type=float, help="Stop an evaluation at its next Newton checkpoint")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--matching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pruner", choices=["hyperband", "median", "none"], default="hyperband")
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=list(KERNELS))
    parser.add_argument("--theta-min", type=float, default=1e-3)
    parser.add_argument("--theta-max", type=float, default=1e4)
    parser.add_argument("--beta-min", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--regularization", choices=["both", "true", "false"], default="both")
    return parser


def main():
    args = build_parser().parse_args()
    if min(args.trials, args.workers, args.max_iters) < 1:
        raise ValueError("trials, workers, and max-iters must be positive")
    if not 0 < args.theta_min < args.theta_max or not 0 < args.beta_min < args.beta_max:
        raise ValueError("theta and beta bounds must be positive and increasing")
    train, validation = parse_runs(args.train_runs), parse_runs(args.validation_runs)
    if not train or set(train) & set(validation):
        raise ValueError("Training runs must be nonempty and disjoint from held-out runs")
    for value in (args.timeout, args.stop_after):
        if value is not None and value <= 0:
            raise ValueError("Timeouts must be positive")
    args.output_dir = args.output_dir.resolve()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for problem_name in args.problems:
        tune_problem(problem_name, args)


if __name__ == "__main__":
    main()
