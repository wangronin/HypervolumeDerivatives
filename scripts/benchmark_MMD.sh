#!/bin/bash
#SBATCH --job-name=MMDNMatching
#SBATCH --array=0-23%20
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

case "${1:-WFG}" in
    WFG) problems=(WFG2 WFG3 WFG4 WFG5 WFG6 WFG7 WFG8 WFG9) ;;
    IDTLZ) problems=(IDTLZ1 IDTLZ2 IDTLZ3 IDTLZ4) ;;
    *) echo "Unknown benchmark family: $1" >&2; exit 1 ;;
esac
algorithms=(NSGA-II NSGA-III MOEAD)
index="${SLURM_ARRAY_TASK_ID}"
problem="${problems[index / 3]}"
algorithm="${algorithms[index % 3]}"
config="mmd_tuning_results/MMDNewton-MMDMatching-${problem}-${algorithm}-bounds-best.json"

srun --cpu-bind=cores venv/bin/python -u scripts/benchmark_MMD.py "${problem}" \
    --algorithm "${algorithm}" --n-jobs "${SLURM_CPUS_PER_TASK}" --generation 600 --max-iters 5 \
    --matching --config-file "${config}"
