"""Shared data loading and Newton evaluation for tuning and candidate plots."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from jax import config as jax_config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
jax_config.update("jax_enable_x64", True)

from hvd import problems
from hvd.delta_p import averaged_hausdorff
from hvd.mmd import MMD, MMDMatching
from hvd.mmd.kernels import RBF, RationalQuadratic
from hvd.mmd_newton import MMDN
from hvd.reference_set import ReferenceSet
from scripts.utils import get_data_prefix, kernel_theta, read_reference_set_data

PROBLEM_NAMES = [
    name
    for name in problems.__all__
    if name.startswith(("IDTLZ", "WFG", "Eq1DTLZ", "Eq1IDTLZ")) and isinstance(getattr(problems, name), type)
]
ALGORITHMS = ["NSGA-II", "NSGA-III", "MOEAD"]
KERNELS = {"rbf": RBF, "rational_quadratic": RationalQuadratic}


def parse_runs(specification: str) -> list[int]:
    """Convert a run list such as '1-20,25' to distinct positive run IDs."""
    runs = []
    for token in specification.split(","):
        if not token.strip():
            continue
        bounds = [int(value) for value in token.split("-")]
        if len(bounds) not in (1, 2):
            raise ValueError(f"Invalid run range: {token}")
        start, stop = bounds if len(bounds) == 2 else (bounds[0], bounds[0])
        if start < 1 or stop < start:
            raise ValueError(f"Invalid run range: {token}")
        runs.extend(range(start, stop + 1))
    return list(dict.fromkeys(runs))


def study_name(problem: str, algorithm: str, matching: bool = True) -> str:
    indicator = "MMDMatching" if matching else "MMD"
    return f"MMDNewton-{indicator}-{problem}-{algorithm}-bounds"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Prevent two jobs from adding trials to the same study simultaneously."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_problem(problem_name: str, args, run: int):
    prefix = get_data_prefix(problem_name, args.algorithm, run, args.generation, args.data_path)
    n_var, n_obj = [
        pd.read_csv(
            args.data_path / f"{prefix}_lastpopu_{space}_gen{args.generation}.csv",
            header=None,
            nrows=1,
        ).shape[1]
        for space in ("x", "y")
    ]
    return getattr(problems, problem_name)(n_var=n_var, n_obj=n_obj, boundary_constraints=True)


def load_run(problem_name: str, args, run: int):
    random.seed(66 + run * 1009)
    np.random.seed(66 + run * 1009)
    return read_reference_set_data(
        args.data_path, problem_name, args.algorithm, run, args.generation, args.matching
    )


def evaluate(
    run: int,
    problem_name: str,
    args,
    configuration: dict,
    problem,
    front,
    report=None,
    directory: Path | None = None,
) -> dict:
    """Evaluate one configuration; optionally save the population for visual review.

    Input: one stored MOEA run and a fixed scoring front. Output: initial/final
    nondominated-set AHD (= max(GD, IGD)) and the final GD/IGD values.
    """
    ref, x0, y0, labels, eta = load_run(problem_name, args, run)
    initial_reference = np.vstack(list(ref.values())).copy()
    reference = ReferenceSet(ref=ref, eta=eta, Y_idx=labels)
    theta = kernel_theta(configuration["theta_multiplier"], y0, initial_reference)
    indicator = (MMDMatching if args.matching else MMD)(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        ref=reference,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        kernel=KERNELS[configuration["kernel"]](theta=theta),
        **({"beta": configuration.get("beta", 0.25)} if args.matching else {}),
    )
    optimizer = MMDN(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        indicator=indicator,
        func=problem.objective,
        jac=problem.objective_jacobian,
        h=problem.eq_constraint,
        h_jac=problem.eq_jacobian,
        h_hessian=problem.eq_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
        N=len(x0),
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=args.max_iters,
        regularization=configuration["regularization"],
        project_box_constraints=True,
        verbose=False,
    )
    initial_ahd, initial_gd, initial_igd = averaged_hausdorff(y0, front)
    history = {"AHD": [initial_ahd], "GD": [initial_gd], "IGD": [initial_igd]}
    for iteration in range(args.max_iters):
        optimizer.newton_iteration()
        if not all(
            np.isfinite(array).all()
            for array in (optimizer.state.Y, optimizer.state.primal, optimizer.step_size)
        ):
            raise FloatingPointError("Nonfinite Newton step or population")
        violation = max(
            np.maximum(problem.xl - optimizer.state.primal, 0).max(),
            np.maximum(optimizer.state.primal - problem.xu, 0).max(),
        )
        if violation > 1e-8:
            raise ValueError(f"Decision bounds violated by {violation:.6g}")
        ahd, gd, igd = averaged_hausdorff(optimizer.state.Y, front)
        if not np.isfinite(ahd):
            raise FloatingPointError("Nonfinite AHD")
        for key, value in zip(history, (ahd, gd, igd)):
            history[key].append(value)
        if report is not None:
            report(ahd, iteration + 1)
    record = dict(run=run, initial_ahd=initial_ahd, ahd=ahd, gd=gd, igd=igd)
    if directory is not None:
        from scripts.tune_MMD.plot_MMD import distribution_metrics, plot_front

        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / f"run{run}.npz",
            X0=x0,
            Y0=y0,
            X=optimizer.state.primal,
            Y=optimizer.state.Y,
            front=front,
            reference_initial=initial_reference,
            reference=reference.reference_set,
        )
        plot_front(
            y0,
            optimizer.state.Y,
            initial_reference,
            front,
            history,
            directory / f"run{run}.png",
            f"{problem_name} / {args.algorithm} / run {run}",
        )
        record.update(
            initial=distribution_metrics(y0, front), final=distribution_metrics(optimizer.state.Y, front)
        )
        write_json(directory / f"run{run}.json", record)
    return record
