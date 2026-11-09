"""Measure single-point function, Jacobian and Hessian CPU times on ALICE.

Warm up JAX before timing. Public problem methods return NumPy arrays, so
each call finishes before the timer stops. Save raw samples, CSVs, a LaTeX
table and basic machine information. Run with measure_CPU_time.sh.
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Set these before importing NumPy or JAX. Slurm also binds the process to one CPU.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import numpy as np
import pandas as pd

from hvd import problems

PROBLEM_NAMES = (
    [f"ZDT{i}" for i in range(1, 5)]
    + [f"DTLZ{i}" for i in range(1, 8)]
    + [f"IDTLZ{i}" for i in range(1, 5)]
    + [f"WFG{i}" for i in range(2, 10)]
)
CALLS = {"FE": "objective", "Jacobian": "objective_jacobian", "Hessian": "objective_hessian"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=PROBLEM_NAMES, default=PROBLEM_NAMES)
    parser.add_argument(
        "--samples", type=int, default=100_000, help="Measured calls per problem and call type"
    )
    parser.add_argument(
        "--warmup", type=int, default=100, help="Unmeasured calls per call type, including JIT"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cpu_time")
    return parser


def measure(function, points: np.ndarray, warmup: np.ndarray) -> np.ndarray:
    """Return CPU nanoseconds per call, excluding compilation and garbage collection."""
    for x in warmup:
        value = function(x)
    if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
        raise ValueError("Expected a finite NumPy result after warmup")

    times = np.empty(len(points), dtype=np.int64)
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i, x in enumerate(points):
            start = time.process_time_ns()
            value = function(x)
            times[i] = time.process_time_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    if not np.isfinite(value).all() or np.any(times <= 0):
        raise ValueError("Invalid measured result or CPU time")
    return times


def write_summary(directory: Path, records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(directory / "CPU_time.csv", index=False)
    by_problem = {}
    for record in records:
        by_problem.setdefault(record["problem"], {})[record["type"]] = record
    lines = [r"""% CPU times in microseconds; see metadata.json for CPU, software and method.
\begin{tabular}{lrrrrrrrrrrr}
\toprule
\multirow{2}{*}{Problem} & \multirow{2}{*}{$n$} & \multirow{2}{*}{$m$} & \multicolumn{3}{c}{Function evaluation ($\mu$s)} & \multicolumn{3}{c}{Jacobian call ($\mu$s)} & \multicolumn{3}{c}{Hessian call ($\mu$s)} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}
 & & & mean & median & 90\% & mean & median & 90\% & mean & median & 90\% \\
\midrule"""]
    ratios = []
    for problem, calls in by_problem.items():
        record = calls["FE"]
        values = [f"{calls[call][stat]:.3f}" for call in CALLS for stat in ("mean_us", "median_us", "p90_us")]
        lines.append(" & ".join([problem, str(record["n_var"]), str(record["n_obj"]), *values]) + r" \\")
        ratios.append(
            {
                "problem": problem,
                "Jacobian_over_FE": calls["Jacobian"]["mean_us"] / record["mean_us"],
                "Hessian_over_FE": calls["Hessian"]["mean_us"] / record["mean_us"],
            }
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (directory / "CPU_time.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(ratios).to_csv(directory / "CPU_time_ratios.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.samples < 1 or args.warmup < 1:
        raise ValueError("Sample and warmup counts must be positive")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metadata.json").exists():
        raise FileExistsError(f"Use a new output directory to preserve the existing run: {directory}")

    cpuinfo = Path("/proc/cpuinfo")
    cpu_models = [platform.processor()]
    if cpuinfo.is_file():
        cpu_models = sorted({
            line.split(":", 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith("model name")
        })
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cpu_models": cpu_models,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "clock": "time.process_time_ns",
        "samples_per_call": args.samples,
        "warmup_per_call": args.warmup,
        "seed": args.seed,
        "problems": args.problems,
        "method": "Single-threaded float64 calls at uniform random inputs; identical points for "
        "all three call types; JIT/warmup excluded; GC disabled while timing. "
        "Dimensions are recorded in CPU_time.csv; DTLZ uses constructor defaults (11, 3).",
        "status": "running",
    }
    metadata_file = directory / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Host: {metadata['hostname']}; CPU: {cpu_models}", flush=True)
    records = []
    try:
        for name in args.problems:
            kwargs = {"n_var": 10, "n_obj": 3} if name.startswith("WFG") else {}
            problem = getattr(problems, name)(**kwargs, boundary_constraints=False)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, PROBLEM_NAMES.index(name)]))
            points = problem.xl + (problem.xu - problem.xl) * rng.random((args.samples, problem.n_var))
            warmup = problem.xl + (problem.xu - problem.xl) * rng.random((args.warmup, problem.n_var))
            raw = {"points": points}
            for label, method in CALLS.items():
                times = measure(getattr(problem, method), points, warmup)
                raw[f"{label}_cpu_ns"] = times
                us = times / 1000.0
                record = dict(
                    type=label,
                    problem=name,
                    n_var=problem.n_var,
                    n_obj=problem.n_obj,
                    samples=args.samples,
                    mean_us=float(np.mean(us)),
                    median_us=float(np.median(us)),
                    p90_us=float(np.quantile(us, 0.9)),
                    min_us=float(np.min(us)),
                    max_us=float(np.max(us)),
                )
                records.append(record)
                print(
                    f"{name} n={problem.n_var} m={problem.n_obj} {label}: "
                    f"mean={record['mean_us']:.3f}, median={record['median_us']:.3f}, p90={record['p90_us']:.3f} us",
                    flush=True,
                )
            np.savez_compressed(directory / f"{name}_samples.npz", **raw)
            write_summary(directory, records)
            # Compiled executables are not needed by the following problem.
            jax.clear_caches()
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
