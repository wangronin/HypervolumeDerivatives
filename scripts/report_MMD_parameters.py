"""Report initial kernel theta values from frozen MMDMatching configurations.

The means use the same population/reference preprocessing and kernel_theta
call as benchmark_MMD.py. This performs no tuning or solver iterations.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import get_run_instances, kernel_theta, read_reference_set_data, set_random_seed

ALGORITHMS = ["NSGA-II", "NSGA-III", "MOEAD"]
PROBLEMS = [f"IDTLZ{i}" for i in range(1, 5)] + [f"WFG{i}" for i in range(2, 10)]
KERNEL_NAMES = {"rbf": "Gaussian", "laplace": "Laplace", "rational_quadratic": "RQ"}


def collect_parameters(data_path: Path, config_dir: Path) -> tuple[list[dict], list[dict]]:
    """Compute theta on each initial population, then average within each pair."""
    means, per_run = [], []
    for problem in PROBLEMS:
        for algorithm in ALGORITHMS:
            config_file = config_dir / f"MMDNewton-MMDMatching-{problem}-{algorithm}-bounds-best.json"
            saved = json.loads(config_file.read_text())
            configuration = saved["best_config"]
            generation = saved["generation"]
            runs = get_run_instances(problem, algorithm, generation, data_path)
            values = []
            for run in runs:
                set_random_seed(66 + run * 1009)
                ref, _, y0, _, _ = read_reference_set_data(
                    data_path, problem, algorithm, run, generation, matching=True
                )
                theta = kernel_theta(configuration["theta_multiplier"], y0, np.vstack(list(ref.values())))
                values.append(theta)
                per_run.append(dict(Baseline=algorithm, Problem=problem, run=run, theta=theta))
            mean = float(np.mean(values))
            means.append(
                dict(
                    Baseline=algorithm,
                    Problem=problem,
                    Kernel=KERNEL_NAMES[configuration["kernel"]],
                    Mean_theta=mean,
                    # Nearest point on the refined reporting grid: three significant digits.
                    Theta_table=float(f"{mean:.3g}"),
                )
            )
            print(f"{problem}/{algorithm}: mean theta = {mean:.12g}", flush=True)
    return means, per_run


def latex_theta(value: float) -> str:
    """Format the snapped scale without requiring siunitx."""
    if value < 0.001 or value >= 1e6:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\times 10^{{{int(exponent)}}}$"
    if value >= 1 and float(value).is_integer():
        return "$" + f"{int(value):,}".replace(",", r"\,") + "$"
    return "$" + f"{value:.3g}" + "$"


def write_parameter_table(directory: Path, means: list[dict], legacy_file: Path) -> None:
    """Write 23 problem rows, each containing the three baseline settings."""
    settings = {}
    with legacy_file.open() as stream:
        for row in csv.DictReader(stream):
            settings[(row["algorithm"], row["problem"])] = (KERNEL_NAMES[row["kernel"]], float(row["param"]))
    # Follow the current manuscript supplied by the user; the repository CSV has 4000.
    settings[("MOEAD", "DTLZ7")] = ("Gaussian", 5000.0)
    for row in means:
        settings[(row["Baseline"], row["Problem"])] = (row["Kernel"], row["Theta_table"])
    problems = [f"ZDT{i}" for i in range(1, 5)] + [f"DTLZ{i}" for i in range(1, 8)] + PROBLEMS
    caption = (
        r"Kernel choices and representative values of $\theta$ for each problem and baseline MOEA. "
        r"The ZDT/DTLZ entries use the conditioning-based settings. "
        r"For IDTLZ/WFG, $\theta$ is the arithmetic mean of the population-normalized initial values "
        r"for the selected Optuna configuration, rounded to the reporting grid defined in the text. "
        r"Individual runs use the corresponding population-normalized values. "
        r"RQ denotes the rational-quadratic kernel with $\alpha=1$."
    )
    lines = [
        r"""% Requires \usepackage{booktabs}.
% DTLZ7/MOEA-D uses the supplied manuscript value 5000; the legacy CSV lists 4000.
\begin{table*}[t]
\centering
\setlength{\tabcolsep}{6pt}""",
        r"\caption{" + caption + "}",
        r"""\label{tab:hyperparameter}
\small
\begin{tabular}{@{}llrlrlr@{}}
\toprule
Problem & \multicolumn{2}{c}{NSGA-II} & \multicolumn{2}{c}{NSGA-III} & \multicolumn{2}{c}{MOEA/D} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
 & Kernel & $\theta$ & Kernel & $\theta$ & Kernel & $\theta$ \\
\midrule""",
    ]
    for problem in problems:
        cells = [problem]
        for algorithm in ALGORITHMS:
            kernel, theta = settings[(algorithm, problem)]
            cells.extend([kernel, latex_theta(theta)])
        lines.append(" & ".join(cells) + r" \\")
        if problem in {"ZDT4", "DTLZ7", "IDTLZ4"}:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (directory / "MMD_parameters_wide.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "mmd_tuning_results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/summary_stats/MMD_parameters")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means, per_run = collect_parameters(args.data_path, args.config_dir)
    pd.DataFrame(per_run).to_csv(args.output_dir / "theta_per_run.csv", index=False)
    pd.DataFrame(means).to_csv(args.output_dir / "theta_mean_and_grid.csv", index=False)
    write_parameter_table(args.output_dir, means, ROOT / "mmd_tuning_results/benchmark_MMD_param.csv")
    print(f"Saved {len(per_run)} run values and {len(means)} pair means to {args.output_dir}")


if __name__ == "__main__":
    main()
