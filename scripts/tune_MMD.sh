#!/bin/bash
#SBATCH --job-name=MMDN-tuning
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --partition=cpu-short
#SBATCH --time=03:00:00
#SBATCH --error="%x-%j-%a.err"
#SBATCH --output="%x-%j-%a.out"
#SBATCH --mail-type=END,FAIL

problems=(IDTLZ1 IDTLZ2 IDTLZ3 IDTLZ4)
problem="${problems[SLURM_ARRAY_TASK_ID]}"
workers="${SLURM_CPUS_PER_TASK:-15}"

source .bashrc
cd $HOME/HypervolumeDerivatives/
source venv/bin/activate
export PYTHONPATH=./:$PYTHONPATH

# One numerical thread per Optuna worker prevents 15 worker processes from
# each starting their own BLAS thread pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=True
# Matplotlib is imported indirectly even though this job does not create plots.
# Give each array task its own writable font/config cache.
export MPLCONFIGDIR="${HOME}/data/tmp/mmd-maplotlib-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

srun --ntasks=1 --cpus-per-task="${workers}" --cpu-bind=cores \
    python scripts/tune_MMD.py "${problem}" \
    --workers "${workers}" \
    --tune-points 0 \
    --validation-points 0 \
    --boundary-constraints \
    --output-dir "${HOME}/data/mmd-tuning" \
    "$@"
