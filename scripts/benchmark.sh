#!/bin/bash
#SBATCH --job-name=DpN
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --partition=cpu-short
#SBATCH --time=03:00:00
#SBATCH --error="%x-%j-%a.err"
#SBATCH --output="%x-%j-%a.out"
#SBATCH --mail-type=END,FAIL

source .bashrc
cd $HOME/HypervolumeDerivatives/
source venv/bin/activate
export PYTHONPATH=./:$PYTHONPATH
problems=(ZDT1 ZDT2 ZDT3 ZDT4)

srun --ntasks=1 --cpus-per-task=15 python scripts/benchmark_ZDT.py ${problems[$SLURM_ARRAY_TASK_ID]}
srun --ntasks=1 --cpus-per-task=15 python scripts/benchmark_EA.py ${problems[$SLURM_ARRAY_TASK_ID]}