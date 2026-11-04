"""Inspect training candidates, freeze a choice, benchmark it, and write a report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hvd.delta_p import averaged_hausdorff
from scripts.tune_MMD.mmd_workflow import (
    ALGORITHMS,
    PROBLEM_NAMES,
    evaluate,
    load_problem,
    load_run,
    study_name,
    write_json,
)
from scripts.utils import get_run_instances


def read_json(path):
    return json.loads(path.read_text())


def apply_settings(args, candidate):
    args.generation = candidate["generation"]
    args.max_iters = candidate["max_iters"]
    args.matching = candidate["indicator"] == "MMDMatching"


def review(args):
    """Plot each candidate's best, upper-median and worst TRAINING run."""
    name = study_name(args.problem, args.algorithm, args.matching)
    candidates = read_json(args.directory / f"{name}-candidates.json")
    if not candidates or not candidates[0]["tuning_complete"]:
        raise ValueError("Finish tuning before generating candidate plots")
    apply_settings(args, candidates[0])
    problem = load_problem(args.problem, args, candidates[0]["train_run_ids"][0])
    front = np.load(args.directory / candidates[0]["front_file"])
    reviews = []
    for candidate in candidates:
        directory = args.directory / "review" / name / f"trial{candidate['best_trial']}"
        ranked = sorted(candidate["training_runs"], key=lambda row: (row["ahd"], row["run"]))
        runs = list(dict.fromkeys(row["run"] for row in (ranked[0], ranked[len(ranked) // 2], ranked[-1])))
        pending = [
            run
            for run in runs
            if not all((directory / f"run{run}.{suffix}").exists() for suffix in ("json", "npz", "png"))
        ]
        if pending:
            Parallel(n_jobs=args.workers)(
                delayed(evaluate)(
                    run, args.problem, args, candidate["best_config"], problem, front, directory=directory
                )
                for run in pending
            )
        reviews.append(
            dict(
                trial=candidate["best_trial"],
                training_ahd=candidate["best_tuning_ahd"],
                runs=runs,
                directory=str(directory.relative_to(args.directory)),
            )
        )
    write_json(args.directory / "review" / name / "review.json", reviews)
    print(f"Open the PNGs under {args.directory / 'review' / name}, then select a candidate.")


def select(args):
    """Freeze a manually inspected candidate, with notes about spread and stability."""
    name = study_name(args.problem, args.algorithm, args.matching)
    destination = args.directory / "selected" / f"{name}-best.json"
    if destination.exists():
        raise ValueError("A candidate is already frozen for this pair")
    reviews = read_json(args.directory / "review" / name / "review.json")
    reviewed = next((row for row in reviews if row["trial"] == args.trial), None)
    if reviewed is None:
        raise ValueError("Generate this candidate's training plots before selecting it")
    directory = args.directory / reviewed["directory"]
    if not all((directory / f"run{run}.png").is_file() for run in reviewed["runs"]):
        raise FileNotFoundError("Candidate training plots are missing")
    candidate = next(
        row
        for row in read_json(args.directory / f"{name}-candidates.json")
        if row["best_trial"] == args.trial
    )
    candidate.update(
        selection_frozen=True, visual_accepted=not args.visual_failed, training_visual_notes=args.notes
    )
    write_json(destination, candidate)
    print(destination)


def benchmark(args):
    """Run the standalone CSV benchmark using the frozen configuration."""
    name = study_name(args.problem, args.algorithm, args.matching)
    selected = args.directory / "selected" / f"{name}-best.json"
    candidate = read_json(selected)
    if not candidate.get("selection_frozen"):
        raise ValueError("Freeze a visually reviewed candidate before benchmarking")
    if candidate.get("benchmark_complete"):
        print("This frozen configuration has already been benchmarked.")
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark_MMD.py"),
        args.problem,
        "--algorithm",
        args.algorithm,
        "--config-file",
        str(selected),
        "--generation",
        str(candidate["generation"]),
        "--max-iters",
        str(candidate["max_iters"]),
        "--data-path",
        str(args.data_path),
        "--results-dir",
        str(args.results_dir),
        "--n-jobs",
        str(args.workers),
    ]
    if candidate["indicator"] == "MMDMatching":
        command.append("--matching")
    subprocess.run(command, check=True)
    candidate["benchmark_complete"] = True
    write_json(selected, candidate)


def report(args):
    """Read frozen choices and performance CSVs; report mean initial/final AHD.

    Initialization is scored from the same prepared populations. Final AHD is
    always max(GD, IGD) from the CSV. Training, held-out and all-run means are
    separate rows. This step performs no Newton iterations.
    """
    rows, selections = [], []
    for path in sorted((args.directory / "selected").glob("*-best.json")):
        candidate = read_json(path)
        if not candidate.get("selection_frozen"):
            raise ValueError(f"Selection is not frozen: {path}")
        if not candidate.get("benchmark_complete"):
            raise ValueError(f"Benchmark this frozen configuration before reporting: {path}")
        args.algorithm = candidate["algorithm"]
        apply_settings(args, candidate)
        problem_name = candidate["problem"]
        method = "MMDNMatching" if args.matching else "MMDN"
        performance = args.results_dir / f"{problem_name}-{method}-{args.algorithm}-{args.generation}.csv"
        final = pd.read_csv(performance).set_index("run", verify_integrity=True)
        runs = get_run_instances(problem_name, args.algorithm, args.generation, args.data_path)
        if set(final.index) != set(runs):
            raise ValueError(f"Benchmark must include every available run: {performance}")
        if not np.isfinite(final[["GD", "IGD"]].to_numpy()).all():
            raise ValueError(f"Nonfinite GD/IGD: {performance}")
        front = np.load(args.directory / candidate["front_file"])
        initial = {run: averaged_hausdorff(load_run(problem_name, args, run)[2], front)[0] for run in runs}
        final_ahd = final[["GD", "IGD"]].max(axis=1)
        for label, group in [
            ("Training", candidate["train_run_ids"]),
            ("Held-out", candidate["validation_run_ids"]),
            ("All", runs),
        ]:
            group = [run for run in group if run in initial]
            if not group:
                continue
            initial_mean = float(np.mean([initial[run] for run in group]))
            final_mean = float(final_ahd.loc[group].mean())
            rows.append(
                {
                    "Problem": problem_name,
                    "Algorithm": args.algorithm,
                    "Runs": f"{label} ({len(group)})",
                    "Initial AHD": initial_mean,
                    "Final AHD": final_mean,
                    "Reduction (%)": (
                        100 * (initial_mean - final_mean) / initial_mean if initial_mean else None
                    ),
                }
            )
        selections.append(
            {
                "Problem": problem_name,
                "Algorithm": args.algorithm,
                "Trial": candidate["best_trial"],
                "Training AHD": candidate["best_tuning_ahd"],
                "Visual": "Pass" if candidate["visual_accepted"] else "Fail",
                "Notes": candidate["training_visual_notes"],
            }
        )
    if not rows:
        raise ValueError("No frozen configurations found")
    table = pd.DataFrame(rows)
    table.to_csv(args.directory / "REPORT.csv", index=False)

    def markdown(records):
        frame = pd.DataFrame(records).fillna("")
        lines = [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        for record in frame.to_dict("records"):
            values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in record.values()]
            lines.append(
                "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |"
            )
        return "\n".join(lines)

    improved = sum(row["Final AHD"] < row["Initial AHD"] for row in rows if row["Runs"].startswith("All "))
    text = (
        "# MMD Newton tuning results\n\n"
        "Mean AHD = max(GD, IGD); positive reduction indicates improvement.\n\n"
        f"{improved}/{len(selections)} pairs improve their all-run mean AHD.\n\n"
        + markdown(rows)
        + "\n\n## Selected configurations\n\n"
        "Training AHD below is the original tuning score used for selection. "
        "Visual assessments describe the inspected training runs; the notes record spacing and stability limitations.\n\n"
        + markdown(selections)
        + "\n"
    )
    (args.directory / "REPORT.md").write_text(text)
    print(args.directory / "REPORT.csv")
    print(args.directory / "REPORT.md")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("review", "select", "benchmark", "report"):
        subparser = commands.add_parser(command)
        subparser.add_argument("directory", type=Path, help="Tuning output directory")
        if command != "report":
            subparser.add_argument("problem", choices=PROBLEM_NAMES)
            subparser.add_argument("--algorithm", choices=ALGORITHMS, default="NSGA-III")
            subparser.add_argument("--matching", action=argparse.BooleanOptionalAction, default=True)
        if command in {"review", "benchmark", "report"}:
            subparser.add_argument("--data-path", type=Path, default=ROOT / "data")
        if command in {"review", "benchmark"}:
            subparser.add_argument("--workers", type=int, default=1)
        if command in {"benchmark", "report"}:
            subparser.add_argument("--results-dir", type=Path, default=ROOT / "results")
        if command == "select":
            subparser.add_argument("--trial", type=int, required=True)
            subparser.add_argument(
                "--notes", required=True, help="Your visual assessment of spread/coverage/stability"
            )
            subparser.add_argument("--visual-failed", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.directory = args.directory.resolve()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("workers must be positive")
    {"review": review, "select": select, "benchmark": benchmark, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
