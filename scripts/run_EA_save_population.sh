#!/bin/bash
#SBATCH --job-name=MOEA
#SBATCH --array=0-62%20
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=60G
#SBATCH --partition=cpu-short
#SBATCH --time=03:00:00
#SBATCH --error="results/logs/%x-%A_%a.err"
#SBATCH --output="results/logs/%x-%A_%a.out"
#SBATCH --mail-type=NONE

set -e
cd "${HOME}/HypervolumeDerivatives"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg

problems=(WFG1 WFG2 WFG3 WFG4 WFG5 WFG6 WFG7 WFG8 WFG9
          IDTLZ1 IDTLZ2 IDTLZ3 IDTLZ4
          Eq1IDTLZ1 Eq1IDTLZ2 Eq1IDTLZ3 Eq1IDTLZ4
          Eq1DTLZ1 Eq1DTLZ2 Eq1DTLZ3 Eq1DTLZ4)
algorithms=(NSGA-II NSGA-III MOEAD)
index="${SLURM_ARRAY_TASK_ID}"
problem="${problems[index / 3]}"
algorithm="${algorithms[index % 3]}"
n_var=11
if [[ "${problem}" == WFG* ]]; then n_var=10; fi

srun --cpu-bind=cores venv/bin/python -u scripts/benchmark_MOEA.py "${problem}" \
    --algorithm "${algorithm}" --budget fixed --generation 600 --runs 30 \
    --n-var "${n_var}" --n-obj 3 --n-jobs "${SLURM_CPUS_PER_TASK}" --save-populations "$@"
