"""Summarize solver CPU seconds as median and 10%-90% quantile width."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--timing-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--failed-timings", type=Path)
    args = parser.parse_args()
    failures = json.loads(args.failed_timings.read_text()) if args.failed_timings else {}
    manifest = json.loads(args.manifest.read_text())
    summaries, cells, missing, models = [], {}, [], set()
    for case in manifest["cases"]:
        prefix = f"{case['problem']}-{case['algorithm']}-{case['generation']}"
        metadata_path = args.timing_dir / f"{prefix}.metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            models.update(metadata["machine"]["cpu_models"])
        for stage, name in case["outputs"].items():
            source = args.timing_dir / name
            if not source.exists():
                missing.append(name)
                continue
            with source.open() as stream:
                rows = list(csv.DictReader(stream))
            ids = [int(row["run"]) for row in rows]
            if sorted(ids) != case["runs"] or len(set(ids)) != len(ids):
                raise ValueError(f"Incomplete run IDs: {source}")
            missing_cpu = [int(row["run"]) for row in rows if row["CPU_time"] == ""]
            if sorted(missing_cpu) != sorted(failures.get(name, [])):
                raise ValueError(f"Undeclared missing CPU times: {source}")
            times = np.array([float(row["CPU_time"]) for row in rows if row["CPU_time"] != ""])
            if not np.isfinite(times).all() or np.any(times <= 0):
                raise ValueError(f"Invalid CPU times: {source}")
            q10, median, q90 = np.quantile(times, [0.1, 0.5, 0.9])
            record = dict(
                Baseline=case["algorithm"],
                Problem=case["problem"],
                Method=stage,
                n_runs=len(times),
                n_var=case["n_var"],
                n_obj=case["n_obj"],
                generation=case["generation"],
                mean_seconds=float(times.mean()),
                median_seconds=float(median),
                q10_seconds=float(q10),
                q90_seconds=float(q90),
                q90_minus_q10_seconds=float(q90 - q10),
            )
            summaries.append(record)
            cells[(case["algorithm"], case["problem"], stage)] = f"{median:.4f} ({q90-q10:.2e})" + (
                r"$^{*}$" if missing_cpu else ""
            )
    if not summaries:
        raise ValueError("No timing results are available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "solver_CPU_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    order = {name: i for i, name in enumerate(manifest["study_problems"])}
    table_rows = []
    for algorithm in ["NSGA-II", "NSGA-III", "MOEAD"]:
        cases = sorted(
            [case for case in manifest["cases"] if case["algorithm"] == algorithm],
            key=lambda c: order[c["problem"]],
        )
        if table_rows:
            table_rows.append(r"\midrule")
        for case in cases:
            values = [
                cells.get((algorithm, case["problem"], stage), "--") for stage in ["MMDN", "DpN", "MOEA"]
            ]
            table_rows.append(" & ".join([algorithm, case["problem"], *values]) + r" \\")
    cpu = ", ".join(sorted(models)).replace("_", r"\_")
    caption = (
        "CPU time in seconds on ALICE (" + cpu + "). We report the median and the "
        r"10\%--90\% quantile width in brackets across 30 runs. Each MMD Newton and DpN run executes "
        r"five iterations from the stored generation-600 warm start. MMD Newton uses the saved "
        r"MMDMatching configurations. Each baseline MOEA runs with the per-run stopping budget "
        r"$\lceil 3.215\,N_{\mathrm{Jac}}\rceil$ computed from the corresponding MMD Newton run. "
        r"The timer measures process CPU time inside each worker. Input loading, solver construction, "
        r"plotting, and final performance metrics are excluded; JIT compilation triggered inside "
        r"the optimization loop is included. Each run uses one allocated CPU with BLAS thread counts set to one. No parameter tuning is performed."
    )
    if failures:
        caption += r" $^{*}$WFG3 with MOEA/D has one failed MMD Newton run; its CPU time is unavailable, and the reported statistics use the other 29 runs."
    latex = (
        "% Requires \\usepackage{booktabs,graphicx}.\n\\begin{table*}[t]\n\\centering\n"
        "\\caption{" + caption + "}\n\\label{tab:solver-CPU-ALICE}\n\\small\n"
        "\\begin{tabular}{llrrr}\n\\toprule\nBaseline & Problem & MMDN & DpN & MOEA alone \\\\\n\\midrule\n"
        + "\n".join(table_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    (args.output_dir / "solver_CPU_table.tex").write_text(latex)
    status = dict(
        status=("complete_with_one_failed_timing" if failures else "complete") if not missing else "partial",
        completed_solver_cases=len(summaries),
        expected_solver_cases=3 * len(manifest["cases"]),
        timed_runs=sum(row["n_runs"] for row in summaries),
        cpu_models=sorted(models),
        missing=missing,
        failed_timings=failures,
        deferred_problems=manifest.get("pending_problems", []),
    )
    (args.output_dir / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
