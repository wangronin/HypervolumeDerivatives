#!/bin/env bash

#SBATCH --job-name=DpN
#SBATCH --partition=cpu-long
#SBATCH --mem-per-cpu=1G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=h.wang@liacs.leidenuniv.nl
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --error="./err/HVI/%x-%j-%a.err"
#SBATCH --output="./out/HVI/%x-%j-%a.out"

PROBLEMS=(CF1 CF2 CF3 CF4 CF5 CF6 CF7 CF8 CF9)
cd $HOME/HypervolumeDerivatives
srun -N1 -n1 -c15 --exclusive python scripts/benchmark_hybrid_DpN.py &
