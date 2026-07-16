#!/bin/bash
#SBATCH --job-name=MOEA
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=cpu-short
#SBATCH --time=04:00:00
#SBATCH --error="%x-%j-%a.err"
#SBATCH --output="%x-%j-%a.out"
#SBATCH --mail-type=END,FAIL

cd ~ && source .bashrc
cd $HOME/HypervolumeDerivatives/
source venv/bin/activate
export PYTHONPATH=./:$PYTHONPATH

srun --ntasks=1 --cpus-per-task=30 python scripts/run_EA_save_population.py $SLURM_ARRAY_TASK_ID