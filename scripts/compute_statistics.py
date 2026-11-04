"""Read paired GD/IGD CSVs and write compact AHD comparison tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

pd.set_option("display.max_rows", None)
ROOT = Path(__file__).resolve().parents[1]
MOEAS = ["NSGA-II", "NSGA-III", "MOEAD"]
FAMILIES = {
    "IDTLZ1-4": [f"IDTLZ{i}" for i in range(1, 5)],
    "WFG2-9": [f"WFG{i}" for i in range(2, 10)],
}
FAMILIES["all"] = FAMILIES["IDTLZ1-4"] + FAMILIES["WFG2-9"]
RUNS = list(range(1, 31))
ALPHA = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/summary_stats")
    parser.add_argument("--generation", type=int, default=600)
    parser.add_argument("--method", choices=["MMDN", "MMDNMatching"], default="MMDNMatching")
    parser.add_argument(
        "--compare", nargs="*", default=["DpN"], help="Other algorithms; MOEA is always included"
    )
    parser.add_argument("--family", choices=FAMILIES, default="all")
    return parser


def load_ahd(path: Path, runs: list[int]) -> np.ndarray:
    """Read GD/IGD, pair by run ID, and return one AHD value per run."""
    data = pd.read_csv(path).set_index("run")
    if data.index.has_duplicates or not set(runs).issubset(data.index):
        raise ValueError(f"Missing or duplicate run IDs in {path}")
    distances = data.loc[runs, ["GD", "IGD"]].to_numpy(dtype=float)
    if not np.isfinite(distances).all():
        raise ValueError(f"Nonfinite GD or IGD in {path}")
    return np.maximum(distances[:, 0], distances[:, 1])


def paired_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Return the two-sided Wilcoxon p-value for paired AHD observations."""
    difference = x - y
    nonzero = difference[difference != 0]
    if not len(nonzero):
        return 1.0
    exact = (
        len(difference) <= 50
        and len(nonzero) == len(difference)
        and len(np.unique(np.abs(nonzero))) == len(nonzero)
    )
    return float(
        wilcoxon(
            difference, zero_method="wilcox", alternative="two-sided", method="exact" if exact else "approx"
        ).pvalue
    )


iqr = lambda v: np.quantile(v, 0.9) - np.quantile(v, 0.1)
get_summary_stats = lambda v: dict(median=np.median(v), IQR=iqr(v))
stats_formatter = lambda s: f"{s['median']:.4f}({s['IQR']:.2e})"


def main() -> None:
    args = build_parser().parse_args()
    comparators = ["MOEA", *args.compare]
    columns = ["Method", "Problem", args.method, *comparators]
    if len(columns) != len(set(columns)):
        raise ValueError("Comparison algorithms must give distinct table columns")
    records, pvalues = [], []
    # Each case uses the same 30 runs. Missing observations are never dropped.
    for moea in MOEAS:
        for problem in FAMILIES[args.family]:
            samples = {}
            row = {"Method": moea, "Problem": problem}
            for algorithm in [args.method, *comparators]:
                tag = "" if algorithm == "MOEA" else f"-{algorithm}"
                path = args.results_dir / f"{problem}{tag}-{moea}-{args.generation}.csv"
                samples[algorithm] = load_ahd(path, RUNS)
                row[algorithm] = get_summary_stats(samples[algorithm])
            records.append(row)
            pvalues.append([paired_pvalue(samples[args.method], samples[other]) for other in comparators])

    # one joint BH correction over every case and comparator, at the fixed 5% level.
    reject = multipletests(np.asarray(pvalues).ravel(), alpha=ALPHA, method="fdr_bh")[0]
    reject = reject.reshape(len(records), len(comparators))
    counts = {algorithm: {"win": 0, "tie": 0, "loss": 0} for algorithm in comparators}
    for row, rejected in zip(records, reject):
        median = row[args.method]["median"]
        row[args.method] = stats_formatter(row[args.method])
        for algorithm, significant in zip(comparators, rejected):
            other_median = row[algorithm]["median"]
            outcome, symbol = "tie", "↔"
            if significant and median != other_median:
                outcome = "win" if median < other_median else "loss"
                symbol = "↑" if outcome == "win" else "↓"
            counts[algorithm][outcome] += 1
            row[algorithm] = stats_formatter(row[algorithm]) + f" {symbol}"
    summary = {"Method": "win/tie/loss", "Problem": "", args.method: ""}
    for algorithm in comparators:
        summary[algorithm] = "/".join(str(counts[algorithm][outcome]) for outcome in ("win", "tie", "loss"))
    table = pd.DataFrame([*records, summary], columns=columns)
    print(table)
    # save the table to CSV format
    comparison_names = "-".join(args.compare) or "MOEA"
    family_suffix = "" if args.family == "all" else f"-{args.family}"
    prefix = args.output_dir / f"{args.method}-{comparison_names}{family_suffix}-{args.generation}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(prefix.with_suffix(".csv"), index=False)
    # save the table to latex format
    caption = rf"""Warmstarting {args.method} after {args.generation} MOEA iterations, we show the
    $\Delta_p$ values (median and 10% - 90% quantile range) of the final approximation set, averaged over 
    {len(RUNS)} paired runs. The {args.method} is executed for five iterations, and the MOEA baseline 
    terminates with evaluation budget equivalent to {args.generation} MOEA iterations + five {args.method}
    iterations. Two-sided Wilcoxon signed-rank tests is applied with Benjamini-Hochberg correction at 
    {ALPHA:.0%}. Comparator arrows indicate a win, non-significant tie, or loss for {args.method}.
    """
    latex = table.to_latex(index=False, escape=True, caption=caption.replace("%", r"\%"))
    for symbol, command in [("↑", r"$\uparrow$"), ("↔", r"$\leftrightarrow$"), ("↓", r"$\downarrow$")]:
        latex = latex.replace(symbol, command)
    prefix.with_suffix(".tex").write_text(latex)


if __name__ == "__main__":
    main()
