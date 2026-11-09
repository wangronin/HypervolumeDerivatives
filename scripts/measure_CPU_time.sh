#!/bin/bash
#SBATCH --job-name=CPU-time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --partition=cpu-short
#SBATCH --constraint=AMD.Zen4
#SBATCH --time=00:45:00
#SBATCH --output=CPU-time-%j.out
#SBATCH --error=CPU-time-%j.err
#SBATCH --mail-type=NONE

set -euo pipefail
# Submit from the source snapshot directory. All 23 problems run sequentially.
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
python="${CPU_TIMING_PYTHON:-${HOME}/HypervolumeDerivatives/venv/bin/python}"
output="${CPU_TIMING_OUTPUT:-${HOME}/data/cpu_time/${SLURM_JOB_ID}}"
srun --exact --cpu-bind=threads "${python}" -u scripts/measure_CPU_time.py --output-dir "${output}" "$@"
