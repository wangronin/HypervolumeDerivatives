# MMD Newton tuning

The workflow is explicit: **tune → inspect → select → benchmark → report**.
There is no background controller, scheduler polling, pilot, or code/data audit.
Install Optuna in the environment used for tuning: `python -m pip install optuna`.

| File | Purpose |
| --- | --- |
| `tune_MMD.py` | Tune on training runs and export the five best COMPLETE candidates. |
| `review_MMD.py` | Generate training plots, freeze a manual selection, run the CSV benchmark, and write the report. |
| `mmd_workflow.py` | Shared data loading and Newton evaluation. |
| `plot_MMD.py` | Initial/reference/final population plots and spacing measurements. |
| `tune_MMD.sh` | Submit tuning, review, or benchmarking as Slurm array jobs. |

## One problem/algorithm pair

Inputs are the prepared population/reference files in `data/`. Defaults are
MMD-Matching, generation 600, five Newton steps, training runs 1–20, held-out
runs 21–30, and 100 cumulative attempts. COMPLETE, PRUNED, and FAIL all count
toward the cap. `--no-matching` selects original MMD.

Use a new output directory for a new experiment. Rerunning the same tuning
command resumes its journal up to the same cumulative cap. It does not retry
failed attempts. Studies with unfinished trials require their state to be
resolved before resuming; no scheduler or automatic recovery is involved.
Historical campaigns and their frozen configurations remain separate.

```sh
python scripts/tune_MMD/tune_MMD.py WFG2 --algorithm NSGA-II --workers 15 --output-dir mmd_tuning_results/new-study
python scripts/tune_MMD/review_MMD.py review mmd_tuning_results/new-study WFG2 --algorithm NSGA-II --workers 15
```

Open all PNGs under `new-study/review/`. Each candidate uses its original best,
upper-median and worst **training** run. Check final spacing, front coverage,
clusters, gaps and amplified outliers against the initial population and actual
initial reference. Review measurements support inspection; they do not select a
candidate automatically. Saved reviews are reused on subsequent invocations.

Choose the lowest original training AHD that passes inspection. If none passes,
retain the best available candidate and pass `--visual-failed`. Record the
specific limitations in `--notes`:

```sh
python scripts/tune_MMD/review_MMD.py select mmd_tuning_results/new-study WFG2 --algorithm NSGA-II --trial TRIAL_NUMBER --notes 'Findings from the training plots'
python scripts/tune_MMD/review_MMD.py benchmark mmd_tuning_results/new-study WFG2 --algorithm NSGA-II --workers 15
python scripts/tune_MMD/review_MMD.py report mmd_tuning_results/new-study
```

Selection writes `selected/STUDY-best.json` and cannot be overwritten. Benchmarking
calls the standalone `scripts/benchmark_MMD.py` on all available runs, using that
choice and its settings. Regularization comes from the selected configuration.
Held-out results are produced after selection and never choose a candidate.

## ALICE

Run numerical work through Slurm. The launcher accepts a family (`WFG`, meaning
WFG2–9, or `IDTLZ`), an output directory, and a stage (`tune`, `review`, or
`benchmark`). Submit each stage after the preceding stage is finished; perform
manual selection between review and benchmarking.

```sh
mkdir -p results/logs
sbatch scripts/tune_MMD/tune_MMD.sh WFG mmd_tuning_results/new-study tune
sbatch scripts/tune_MMD/tune_MMD.sh WFG mmd_tuning_results/new-study review
# Inspect plots and run the select command for each pair.
sbatch scripts/tune_MMD/tune_MMD.sh WFG mmd_tuning_results/new-study benchmark
```

For IDTLZ1–4, add `--array=0-11%12` and use `IDTLZ`. To process one pair, use its
array index: `3 * problem_position + algorithm_position`, with NSGA-II, NSGA-III,
MOEAD in that order. Each job receives 15 CPUs/60 GB for at most three hours.
Tuning stops starting trials after 2h30 and stops evaluations at a Newton
checkpoint after 2h50. If the cap has not been reached, resubmit the same pair.

## Outputs

The tuning directory contains the journal, completion log, full readable trial
history, saved scoring front, and candidate configurations. Numeric `*-best.json`
files describe the tuning incumbent; `selected/*-best.json` files describe the
manual frozen choice. Candidate PNGs, population NPZs and review measurements
live under `review/`.

Performance files use `results/PROBLEM-MMDNMatching-ALGORITHM-GENERATION.csv`
(or `MMDN`), with `run,HV,IGD,GD,Jac_calls`. Reporting reads those CSVs and scores
the same prepared initial populations, without Newton iterations. `REPORT.csv`
and `REPORT.md` show mean initial/final AHD and percentage reduction, separately
for training, held-out and all runs. AHD is always `max(GD, IGD)`. The Markdown
report also includes the original selected training score and visual notes.
