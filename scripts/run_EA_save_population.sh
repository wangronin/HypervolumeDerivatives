#!/bin/bash
#SBATCH --job-name=MOEA
#SBATCH --array=0-21
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --partition=cpu-short
#SBATCH --time=03:00:00
#SBATCH --error="%x-%j-%a.err"
#SBATCH --output="%x-%j-%a.out"
#SBATCH --mail-type=END,FAIL

cd ~ && source .bashrc
cd $HOME/HypervolumeDerivatives/
source venv/bin/activate
export PYTHONPATH=./:$PYTHONPATH

srun --ntasks=1 --cpus-per-task=15 python scripts/run_EA_save_population.py $SLURM_ARRAY_TASK_ID